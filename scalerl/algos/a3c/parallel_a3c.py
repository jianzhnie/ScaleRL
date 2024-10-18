from __future__ import print_function

import os
import queue
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

# Ensure paths are correctly set
sys.path.append(os.getcwd())

# Custom imports from project
from scalerl.algos.a3c.share_optim import SharedAdam
from scalerl.algos.rl_args import A3CArguments
from scalerl.envs.gym_env import make_gym_env


class ActorCriticNet(nn.Module):
    """Actor-Critic Neural Network.

    This network generates policy logits for the actor and value estimates for
    the critic from given observations.
    """

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int) -> None:
        """Initialize the Actor-Critic network.

        Args:
            obs_dim (int): Dimension of the observation space.
            hidden_dim (int): Number of neurons in the hidden layer.
            action_dim (int): Dimension of the action space.
        """
        super(ActorCriticNet, self).__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_linear = nn.Linear(hidden_dim, action_dim)
        self.critic_linear = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the network.

        Args:
            obs (torch.Tensor): Observations from the environment.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Policy logits and state-value estimate.
        """
        feature = self.feature_net(obs)
        value = self.critic_linear(feature)
        logits = self.actor_linear(feature)
        return logits, value


class ParallelA3C:
    """A3C Trainer Class to handle training and testing processes."""

    def __init__(self, args: A3CArguments) -> None:
        """Initialize the A3C trainer with the environment, shared model, and
        optimizer.

        Args:
            args (A3CArguments): Hyperparameters and arguments for A3C.
        """
        self.args = args
        self.gamma = args.gamma

        # Initialize environments for training and testing
        self.env = make_gym_env(self.args.env_name)
        self.train_env = make_gym_env(env_id=self.args.env_name)
        self.test_env = make_gym_env(env_id=self.args.env_name)

        # Get observation and action dimensions
        obs_shape = self.env.observation_space.shape or (
            self.env.observation_space.n, )
        action_shape = self.env.action_space.shape or (
            self.env.action_space.n, )
        self.obs_dim = int(np.prod(obs_shape))
        self.action_dim = int(np.prod(action_shape))

        # Initialize local and shared models
        self.local_model = ActorCriticNet(self.obs_dim, self.args.hidden_dim,
                                          self.action_dim)
        self.shared_model = ActorCriticNet(self.obs_dim, self.args.hidden_dim,
                                           self.action_dim)
        self.shared_model.share_memory(
        )  # Share the model memory across processes

        # Initialize optimizer
        if not self.args.no_shared:
            self.optimizer = SharedAdam(self.shared_model.parameters(),
                                        lr=self.args.lr)
            self.optimizer.share_memory(
            )  # Share the optimizer state across processes
        else:
            self.optimizer = None

    def get_action(self, obs: torch.Tensor) -> int:
        """Select an action using the local policy.

        Args:
            obs (torch.Tensor): The observation input.

        Returns:
            int: The selected action.
        """
        if obs.ndim == 1:
            obs = np.expand_dims(obs, axis=0)  # Ensure batch size is 1

        obs = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            logits, _ = self.local_model(obs)
        probs = F.softmax(logits, dim=-1)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def predict(self, obs: np.ndarray) -> int:
        """Predict the most probable action.

        Args:
            obs (np.ndarray): Current observation.

        Returns:
            int: Action selected by the policy.
        """
        if obs.ndim == 1:
            obs = np.expand_dims(obs, axis=0)  # Ensure batch size is 1

        obs = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            logits, _ = self.local_model(obs)
        probs = F.softmax(logits, dim=-1)
        action = probs.argmax(dim=-1)
        return action.item()

    def _sync_with_shared_model(self, local_model: nn.Module,
                                shared_model: nn.Module) -> None:
        """Synchronize local model parameters with the shared model.

        Args:
            local_model (nn.Module): Local model.
            shared_model (nn.Module): Shared global model.
        """
        local_model.load_state_dict(shared_model.state_dict())

    def ensure_shared_grads(self, local_model: nn.Module,
                            shared_model: nn.Module) -> None:
        """Ensure gradients from local model are applied to the shared model.

        Args:
            local_model (nn.Module): Local worker's model.
            shared_model (nn.Module): Shared global model.
        """
        for param, shared_param in zip(local_model.parameters(),
                                       shared_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad.clone()

    def compute_loss(
            self, transition_dict: Dict[str,
                                        List[np.ndarray]]) -> torch.Tensor:
        """Compute loss for both Actor and Critic.

        Args:
            transition_dict (Dict[str, List[np.ndarray]]): Transitions containing states, actions, rewards, etc.

        Returns:
            torch.Tensor: Total loss for the Actor and Critic.
        """
        # Convert transition arrays to tensors
        obs = torch.tensor(np.array(transition_dict['obs']),
                           dtype=torch.float32)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.long).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float32).view(-1, 1)
        next_obs = torch.tensor(np.array(transition_dict['next_obs']),
                                dtype=torch.float32)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float32).view(-1, 1)

        # Get current and next policy and value estimates
        curr_policy, curr_value = self.local_model(obs)
        _, next_value = self.local_model(next_obs)
        curr_probs = F.softmax(curr_policy, dim=-1)

        # Compute TD target and error
        td_target = rewards + self.gamma * next_value * (1 - dones)
        td_error = td_target - curr_value

        # Compute Actor and Critic losses
        log_probs = torch.log(curr_probs).gather(1, actions)
        actor_loss = torch.mean(-log_probs * td_error.detach())
        critic_loss = F.mse_loss(td_target.detach(), curr_value)

        total_loss = actor_loss + critic_loss
        return total_loss

    def train(
        self,
        worker_id: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        global_ep_counter: Optional[mp.Value] = None,
        global_reward_queue: Optional[mp.Queue] = None,
    ) -> None:
        """Train the model using asynchronous advantage actor-critic.

        Args:
            worker_id (int): Worker process ID.
            optimizer (Optional[torch.optim.Optimizer]): Optimizer for updating shared model.
            global_ep_counter (Optional[mp.Value]): Global episode counter.
            global_reward_queue (Optional[mp.Queue]): Queue to store rewards.
        """
        seed = self.args.seed + worker_id
        torch.manual_seed(seed)
        self.local_model.train()

        if optimizer is None:
            optimizer = torch.optim.Adam(self.shared_model.parameters(),
                                         lr=self.args.lr)

        # Sync local model with the shared model
        self._sync_with_shared_model(self.local_model, self.shared_model)

        while global_ep_counter.value < self.args.max_episode_size:
            transition_dict = {
                'obs': [],
                'actions': [],
                'next_obs': [],
                'rewards': [],
                'dones': [],
            }
            episode_reward = 0

            obs, _ = self.train_env.reset(seed=seed)
            done = False

            while not done:
                action = self.get_action(obs)
                next_obs, reward, terminal, truncated, _ = self.train_env.step(
                    action)
                done = terminal or truncated

                transition_dict['obs'].append(obs)
                transition_dict['actions'].append(action)
                transition_dict['next_obs'].append(next_obs)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)

                obs = next_obs
                episode_reward += reward

            total_loss = self.compute_loss(transition_dict)
            print(
                f'Total loss: {total_loss.item()}, Episode reward: {episode_reward}'
            )

            # Backpropagation and optimization
            self.optimizer.zero_grad()
            total_loss.backward()
            self.ensure_shared_grads(self.local_model, self.shared_model)
            self.optimizer.step()

            # Update global episode counter
            with global_ep_counter.get_lock():
                global_ep_counter.value += 1
                global_reward_queue.put(episode_reward)

    def test(self, worker_id: int) -> Tuple[float, int]:
        """Test the model by running it in the environment and return episode
        reward and length.

        Args:
            worker_id (int): Worker process ID.

        Returns:
            Tuple[float, int]: Episode reward and episode length.
        """
        self.local_model.eval()
        seed = self.args.seed + worker_id
        torch.manual_seed(seed=seed)

        self._sync_with_shared_model(self.local_model, self.shared_model)
        obs, info = self.test_env.reset(seed=seed)

        done = False
        episode_reward = 0.0
        episode_length = 0

        while not done:
            action = self.predict(obs)
            next_obs, reward, terminal, truncated, _ = self.test_env.step(
                action)
            done = terminal or truncated
            obs = next_obs

            if info and 'episode' in info:
                info_item = {k: v.item() for k, v in info['episode'].items()}
                episode_reward = info_item['r']
                episode_length = info_item['l']

        print(
            f'Episode reward: {episode_reward}, Episode length: {episode_length}'
        )
        return episode_reward, episode_length

    def run(self) -> None:
        """Run the agent by spawning processes for training and testing."""
        processes: List[mp.Process] = []

        # Create shared counters and reward queue
        global_ep_counter = mp.Value('i', 0)
        global_reward_queue = queue.Queue()

        # Start the training processes
        for rank in range(self.args.num_processes):
            p = mp.Process(
                target=self.train,
                args=(rank, self.optimizer, global_ep_counter,
                      global_reward_queue),
            )
            p.start()
            processes.append(p)

        # Wait for all processes to finish
        for p in processes:
            p.join()


if __name__ == '__main__':
    args = A3CArguments()
    a3c = ParallelA3C(args)
    a3c.run()
