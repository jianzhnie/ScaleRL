import copy
import random

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.optim as optim

from hssdrl.algos.apex.memory import PrioritizedReplayBuffer
from hssdrl.algos.apex.network import QNet


class Actor(mp.Process):

    def __init__(
        self,
        actor_id,
        env_name: str,
        replay_buffer: PrioritizedReplayBuffer,
        global_step: int,
        batch_size: int,
        eps: float = 0.4,
        gamma: float = 0.99,
        eps_greedy: float = 1.0,
    ) -> None:
        self.actor_id = actor_id
        self.eps = eps
        self.gamma = gamma
        self.global_step = global_step
        self.batch_size = batch_size
        self.eps_greedy = eps_greedy
        self.replay_buffer = replay_buffer
        self.env = gym.make(env_name)
        self.action_dim = self.env.action_space.n
        self.obs_dim = self.env.action_space.shape[0]
        self.q_net = QNet(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=128,
        )
        self.q_target = copy.deepcopy(self.q_net)

    def get_info(self) -> None:
        with self.global_step.get_lock():
            self.global_step.value += 1

    def run(self) -> None:
        obs, info = self.env.reset()
        done = False
        while not done:
            action = self.get_action(obs)
            next_obs, reward, terminal, truncated, info = self.env.step(action)
            if terminal or truncated:
                done = True
            self.replay_buffer.add((obs, action, reward, next_obs, done))
            obs = next_obs

            if len(self.replay_buffer) > self.batch_size:
                batch = self.replay_buffer.sample(self.batch_size)

    def compute_prior(self, transitions):  # -> Any:
        obs, next_obs, actions, rewards, masks, steps = transitions

        obs = torch.stack(obs)
        next_obs = torch.stack(next_obs)
        actions = torch.Tensor(actions)
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        steps = torch.Tensor(steps)

        pred = self.local_model(obs).squeeze(1)
        next_pred = self.local_model(next_obs).squeeze(1)

        pred_action = (pred * actions).sum(dim=1)

        target = rewards + masks * pow(gamma, steps) * next_pred.max(1)[0]

        td_error = pred_action - target
        prior = abs(td_error.detach())

        return prior

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Get action from the actor network.

        Args:
            obs (np.ndarray): Current observation.

        Returns:
            np.ndarray: Selected action.
        """
        # epsilon greedy policy
        if np.random.rand() <= self.eps_greedy:
            action = random.randint(0, self.action_dim - 1)
        else:
            action = self.predict(obs)

        return action

    def predict(self, obs: np.ndarray) -> int:
        """Predict an action given an observation.

        Args:
            obs (np.ndarray): Current observation.

        Returns:
            int: Selected action.
        """
        if obs.ndim == 1:
            # Expand to have batch_size = 1
            obs = np.expand_dims(obs, axis=0)

        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        with torch.no_grad():
            q_values = self.q_net(obs)
        action = torch.argmax(q_values, dim=1).item()
        return action


class Learner:

    def __init__(self,
                 model,
                 target_model,
                 replay_buffer,
                 batch_size=32,
                 gamma=0.99,
                 lr=1e-3):
        self.model = model
        self.target_model = target_model
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        indices, samples, weights = self.replay_buffer.sample(self.batch_size)
        obs, actions, rewards, next_obs, dones = zip(*samples)

        obs = torch.FloatTensor(obs)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_obs = torch.FloatTensor(next_obs)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        weights = torch.FloatTensor(weights).unsqueeze(1)

        current_q_values = self.model(obs).gather(1, actions)
        next_q_values = self.target_model(next_obs).max(1, keepdim=True)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        td_error = torch.abs(current_q_values -
                             target_q_values).detach().numpy()
        self.replay_buffer.update_priorities(indices, td_error)

        loss = (weights *
                (current_q_values - target_q_values.detach())**2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_obs_dict(self.model.obs_dict())
