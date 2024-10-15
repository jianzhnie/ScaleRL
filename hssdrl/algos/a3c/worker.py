import argparse
import multiprocessing as mp
import time
from collections import deque
from typing import Deque, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from hssdrl.envs.gym_env import make_gym_env


def ensure_shared_grads(model: torch.nn.Module,
                        shared_model: torch.nn.Module) -> None:
    """Copies the gradients from the model to the shared_model. This ensures
    that the shared model gets updated with gradients from worker threads.

    Args:
        model (torch.nn.Module): The local model with calculated gradients.
        shared_model (torch.nn.Module): The shared model that needs updated gradients.
    """
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return  # If the shared model already has gradients, skip.
        shared_param._grad = (
            param.grad)  # Copy gradient from local model to shared model


class ActorCritic(nn.Module):

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(ActorCritic, self).__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_linear = nn.Linear(hidden_dim, action_dim)
        self.critic_linear = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        fearture = self.feature_net(obs)
        value = self.critic_linear(fearture)
        logits = self.actor_linear(fearture)
        return value, logits


def train(
    worker_id: int,
    counter: mp.Value(),
    lock: mp.Lock(),
    shared_model: torch.nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    args: argparse.Namespace = None,
) -> None:
    """Worker training function for asynchronous advantage actor-critic (A3C).

    Args:
        rank (int): The rank of the worker.
        args: Command-line arguments or parameters.
        shared_model (torch.nn.Module): The model shared among all worker processes.
        counter: A shared counter for tracking the number of updates.
        lock (Lock): A lock for updating the shared counter safely.
        optimizer (Optional[optim.Optimizer]): Optimizer for the shared model. If None, Adam is used.
    """
    # Set seed for reproducibility based on rank
    torch.manual_seed(args.seed + worker_id)

    # Create and initialize the environment
    env: gym.Env = make_gym_env(args.env_name)
    env.seed(args.seed + worker_id)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    # Flatten state and action shapes
    obs_dim = int(np.prod(state_shape))
    action_dim = int(np.prod(action_shape))

    # Create a local model
    model = ActorCritic(obs_dim=obs_dim,
                        hidden_dim=args.hidden_dim,
                        action_dim=action_dim)

    # Initialize the optimizer if not provided
    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    # Set model to training mode
    model.train()

    # Reset the environment and get the initial state
    state, info = env.reset()
    state = torch.from_numpy(state)  # Convert state to a PyTorch tensor
    done = True  # Initialize the 'done' flag

    episode_length = 0  # Track episode length
    while True:
        # Sync the local model with the shared model
        model.load_state_dict(shared_model.state_dict())
        # Prepare lists to store trajectories
        values = []
        log_probs = []
        rewards = []
        entropies = []

        # Rollout loop
        for step in range(args.num_steps):
            episode_length += 1

            # Get model output (value, policy logits, new hidden/cell states)
            value, logit = model((state.unsqueeze(0)))

            # Compute action probabilities and log probabilities
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            # Compute entropy for exploration bonus
            entropies.append(entropy)

            # Sample an action from the policy distribution
            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            # Step the environment with the selected action
            state, reward, terminal, truncated, _ = env.step(action.numpy())
            done = terminal or truncated
            # Check if episode is done
            reward = max(min(reward, 1), -1)  # Clip reward to range [-1, 1]

            with lock:
                counter.value += 1  # Increment the shared counter

            if done:
                episode_length = 0  # Reset episode length
                state, info = env.reset()  # Reset the environment

            # Convert state to a PyTorch tensor
            state = torch.from_numpy(state)

            # Store trajectories
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        # Initialize the Generalized Advantage Estimation (GAE) and the value of R
        R = torch.zeros(1, 1)
        if not done:
            value, logits = model(state.unsqueeze(0))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)

        # Compute the losses using Generalized Advantage Estimation (GAE)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(
                2)  # MSE loss for the critic

            # GAE computation
            delta_t = rewards[i] + args.gamma * values[i + 1] - values[i]
            gae = gae * args.gamma * args.gae_lambda + delta_t

            # Policy loss with entropy regularization
            policy_loss = (policy_loss - log_probs[i] * gae.detach() -
                           args.entropy_coef * entropies[i])

        # Backpropagation
        optimizer.zero_grad()  # Clear the optimizer gradients
        (policy_loss +
         args.value_loss_coef * value_loss).backward()  # Compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        # Gradient clipping
        # Ensure gradients are copied to the shared model
        ensure_shared_grads(model, shared_model)
        optimizer.step()  # Update the shared model


def test(rank: int, args, shared_model: torch.nn.Module, counter) -> None:
    """Test function for the asynchronous advantage actor-critic (A3C) agent.
    It runs a policy in a test environment and synchronizes periodically with a
    shared model to evaluate performance.

    Args:
        rank (int): The rank of the testing worker.
        args: Command-line arguments or parameters including environment, seed, and model configuration.
        shared_model (torch.nn.Module): The shared model containing the policy that is periodically synced.
        counter: A shared counter to track the number of global steps.
    """
    # Set the seed for reproducibility based on the worker rank
    torch.manual_seed(args.seed + rank)

    # Create the environment and seed it
    env = make_gym_env(args.env_name)
    env.seed(args.seed + rank)

    # Create a local copy of the model
    # Create a local model
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    # Flatten state and action shapes
    obs_dim = int(np.prod(state_shape))
    action_dim = int(np.prod(action_shape))
    model = ActorCritic(obs_dim=obs_dim,
                        hidden_dim=args.hidden_dim,
                        action_dim=action_dim)
    model.eval()
    # Set the model to evaluation mode

    state = env.reset()  # Initialize environment and get the first state
    state = torch.from_numpy(state)  # Convert state to a PyTorch tensor
    reward_sum = 0  # Track total reward per episode
    done = True  # Initialize done flag

    start_time = time.time()  # Track start time for FPS calculation

    # Prevent agent from getting stuck in repeating actions using a deque
    actions: Deque[int] = deque(maxlen=100)
    episode_length = 0  # Track the episode length

    while True:
        episode_length += 1

        # Sync local model with the shared model when the episode ends
        if done:
            model.load_state_dict(shared_model.state_dict())
            # Load shared model weights

        # No gradient tracking during testing
        with torch.no_grad():
            value, logit = model(state.unsqueeze(0))
        # Forward pass through the model
        prob = F.softmax(logit, dim=-1)  # Get action probabilities
        action = prob.max(1, keepdim=True)[1].numpy()
        # Select the action with the highest probability

        # Step the environment with the selected action
        state, reward, terminal, truncated, _ = env.step(action[0, 0])
        done = terminal or truncated
        # Check if episode is done (max steps or terminal)
        reward_sum += reward  # Accumulate the reward for the current episode

        # Prevent agent from getting stuck by tracking repeated actions
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True
        # Force reset if the agent repeats the same action for too long

        # Print performance metrics at the end of the episode
        if done:
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            fps = (counter.value / elapsed_time
                   )  # Calculate FPS based on number of steps and elapsed time
            print(
                f"Time {time.strftime('%Hh %Mm %Ss', time.gmtime(elapsed_time))}, "
                f'num steps {counter.value}, FPS {fps:.0f}, '
                f'episode reward {reward_sum}, episode length {episode_length}'
            )

            # Reset episode tracking variables
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()

            # Sleep to throttle the testing process
            time.sleep(
                60)  # Sleep for 60 seconds before starting the next episode

        # Convert the next state to a tensor for further model processing
        state = torch.from_numpy(state)
