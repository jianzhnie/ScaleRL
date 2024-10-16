from __future__ import print_function

import multiprocessing as mp
import os
import time
from collections import deque
from typing import Deque, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scalerl.algos.a3c.share_optim import SharedAdam
from scalerl.algos.rl_args import A3CArguments
from scalerl.envs.gym_env import make_gym_env


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


class A3CAgent:

    def __init__(self, args: A3CArguments) -> None:
        """Initialize the A3CAgent with shared model, environment and
        optimizer."""
        self.args = args
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

        torch.manual_seed(self.args.seed)

        # Initialize environment
        self.env = make_gym_env(self.args.env_name)
        self.env.seed(self.args.seed)

        # Observation and action dimensions
        state_shape = self.env.observation_space.shape or self.env.observation_space.n
        action_shape = self.env.action_space.shape or self.env.action_space.n
        self.obs_dim = int(np.prod(state_shape))
        self.action_dim = int(np.prod(action_shape))

        # Initialize shared model
        self.shared_model = ActorCritic(
            obs_dim=self.obs_dim,
            hidden_dim=self.args.hidden_dim,
            action_dim=self.action_dim,
        )
        self.shared_model.share_memory(
        )  # Ensure model is shared across processes

        # Initialize optimizer
        if not self.args.no_shared:
            self.optimizer = SharedAdam(self.shared_model.parameters(),
                                        lr=self.args.lr)
            self.optimizer.share_memory()
        else:
            self.optimizer = torch.optim.Adam(self.shared_model.parameters,
                                              lr=self.args.lr)

        # Create shared counter for processes
        self.counter = mp.Value('i', 0)
        self.lock = mp.Lock()

    def _sync_with_shared_model(self, local_model: torch.nn.Module):
        """Load the shared model parameters into the local model."""
        local_model.load_state_dict(self.shared_model.state_dict())

    def ensure_shared_grads(self, model: torch.nn.Module) -> None:
        """Ensure that gradients from the local model are copied to the shared
        model."""
        for param, shared_param in zip(model.parameters(),
                                       self.shared_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad.clone()

    def train(self, worker_id: int) -> None:
        """Worker training function."""
        torch.manual_seed(self.args.seed + worker_id)

        # Local model for the worker
        model = ActorCritic(
            obs_dim=self.obs_dim,
            hidden_dim=self.args.hidden_dim,
            action_dim=self.action_dim,
        )
        model.train()

        state, info = self.env.reset()
        state = torch.from_numpy(state).float()
        done = True
        episode_length = 0

        while True:
            self._sync_with_shared_model(
                model)  # Sync local model with shared model
            values, log_probs, rewards, entropies = [], [], [], []

            for step in range(self.args.num_steps):
                episode_length += 1
                value, logit = model(state.unsqueeze(0))
                prob = F.softmax(logit, dim=-1)
                log_prob = F.log_softmax(logit, dim=-1)
                entropy = -(log_prob * prob).sum(1, keepdim=True)
                entropies.append(entropy)

                action = prob.multinomial(num_samples=1).detach()
                log_prob = log_prob.gather(1, action)

                state, reward, terminal, truncated, info = self.env.step(
                    action.numpy())
                done = terminal or truncated
                reward = max(min(reward, 1), -1)

                with self.lock:
                    self.counter.value += 1

                if done:
                    episode_length = 0
                    state, info = self.env.reset()

                state = torch.from_numpy(state).float()

                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)

                if done:
                    break

            R = torch.zeros(1, 1)
            if not done:
                value, _ = model(state.unsqueeze(0))
                R = value.detach()

            values.append(R)
            policy_loss, value_loss, gae = 0, 0, torch.zeros(1, 1)

            for i in reversed(range(len(rewards))):
                R = self.args.gamma * R + rewards[i]
                advantage = R - values[i]
                value_loss += 0.5 * advantage.pow(2)

                delta_t = rewards[i] + self.args.gamma * values[i +
                                                                1] - values[i]
                gae = gae * self.args.gamma * self.args.gae_lambda + delta_t

                policy_loss -= (log_probs[i] * gae.detach() +
                                self.args.entropy_coef * entropies[i])

            self.optimizer.zero_grad()
            (policy_loss + self.args.value_loss_coef * value_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           self.args.max_grad_norm)
            self.ensure_shared_grads(model)
            self.optimizer.step()

    def test(self, rank: int) -> None:
        """Test worker function to evaluate the performance of the agent."""
        torch.manual_seed(self.args.seed + rank)

        model = ActorCritic(
            obs_dim=self.obs_dim,
            hidden_dim=self.args.hidden_dim,
            action_dim=self.action_dim,
        )
        model.eval()

        state, info = self.env.reset()
        state = torch.from_numpy(state).float()
        reward_sum = 0
        done = True
        episode_length = 0
        start_time = time.time()

        actions: Deque[int] = deque(maxlen=100)

        while True:
            episode_length += 1

            if done:
                self._sync_with_shared_model(model)

            with torch.no_grad():
                value, logit = model(state.unsqueeze(0))
            prob = F.softmax(logit, dim=-1)
            action = prob.max(1, keepdim=True)[1].numpy()

            state, reward, terminal, truncated, _ = self.env.step(action[0, 0])
            done = terminal or truncated
            reward_sum += reward

            actions.append(action[0, 0])
            if actions.count(actions[0]) == actions.maxlen:
                done = True

            if done:
                elapsed_time = time.time() - start_time
                fps = self.counter.value / elapsed_time
                print(
                    f"Time {time.strftime('%Hh %Mm %Ss', time.gmtime(elapsed_time))}, "
                    f'num steps {self.counter.value}, FPS {fps:.0f}, '
                    f'episode reward {reward_sum}, episode length {episode_length}'
                )

                reward_sum = 0
                episode_length = 0
                actions.clear()
                state, info = self.env.reset()

                time.sleep(60)

            state = torch.from_numpy(state).float()

    def run(self) -> None:
        """Run the agent, spawning processes for training and testing."""
        processes: List[mp.Process] = []

        # Start the testing process
        test_process = mp.Process(target=self.test,
                                  args=(self.args.num_processes, ))
        test_process.start()
        processes.append(test_process)

        # Start training processes
        for rank in range(self.args.num_processes):
            p = mp.Process(target=self.train, args=(rank, ))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
