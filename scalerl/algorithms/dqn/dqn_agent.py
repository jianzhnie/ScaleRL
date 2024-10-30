import copy
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator

from scalerl.algorithms.base import BaseAgent
from scalerl.algorithms.utils.network import QNet
from scalerl.utils.algo_utils import unwrap_optimizer
from scalerl.utils.model_utils import soft_target_update


class DQN(BaseAgent):
    """The DQN algorithm class.

    DQN paper: https://arxiv.org/abs/1312.5602
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        learn_step: int = 5,
        gamma: float = 0.99,
        tau: float = 1e-3,
        double_dqn: bool = False,
        device: str = 'cpu',
        accelerator: Optional[Accelerator] = None,
        wrap: bool = True,
    ) -> None:
        """Initializes the DQN agent."""
        super().__init__()
        self.algo = 'DQN'
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learn_step = learn_step
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.accelerator = accelerator
        self.double_dqn = double_dqn

        # Create the actor network and target network
        self.actor = QNet(obs_dim=self.obs_dim, action_dim=self.action_dim)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.optimizer = optim.Adam(self.actor.parameters(),
                                    lr=self.learning_rate)

        if self.accelerator is not None:
            if wrap:
                self.wrap_models()
        else:
            self.actor = self.actor.to(self.device)
            self.actor_target = self.actor_target.to(self.device)

        self.criterion = nn.MSELoss()

    def get_action(
        self,
        state: np.ndarray,
        epsilon: float = 0,
        action_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Returns the next action to take in the environment.

        Epsilon is the probability of taking a random action, used for
        exploration. For epsilon-greedy behavior, set epsilon to 0.
        """
        state = torch.from_numpy(state).float()
        if self.accelerator is None:
            state = state.to(self.device)
        else:
            state = state.to(self.accelerator.device)

        state = state.unsqueeze(0)

        # epsilon-greedy
        if random.random() < epsilon:
            if action_mask is None:
                action = np.random.randint(0, self.action_dim, size=len(state))
            else:
                action = np.argmax(
                    (np.random.uniform(0, 1, (len(state), self.action_dim)) *
                     action_mask),
                    axis=1,
                )
        else:
            self.actor.eval()
            with torch.no_grad():
                action_values = self.actor(state).cpu().data.numpy()
            self.actor.train()

            if action_mask is None:
                action = np.argmax(action_values, axis=-1)
            else:
                inv_mask = 1 - action_mask
                masked_action_values = np.ma.array(action_values,
                                                   mask=inv_mask)
                action = np.argmax(masked_action_values, axis=-1)

        return action

    def predict(
        self,
        state: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
    ) -> np.array:
        state = torch.from_numpy(state).float()
        if self.accelerator is None:
            state = state.to(self.device)
        else:
            state = state.to(self.accelerator.device)

        state = state.unsqueeze(0)
        with torch.no_grad():
            action_values = self.actor(state).cpu().data.numpy()

        if action_mask is None:
            action = np.argmax(action_values, axis=-1)
        else:
            inv_mask = 1 - action_mask
            masked_action_values = np.ma.array(action_values, mask=inv_mask)
            action = np.argmax(masked_action_values, axis=-1)
        return action

    def learn(self, experiences: List[torch.Tensor]) -> float:
        """Updates agent network parameters to learn from experiences."""
        states, actions, rewards, next_states, dones = experiences
        if self.accelerator is not None:
            states = states.to(self.accelerator.device)
            actions = actions.to(self.accelerator.device)
            rewards = rewards.to(self.accelerator.device)
            next_states = next_states.to(self.accelerator.device)
            dones = dones.to(self.accelerator.device)

        if self.double_dqn:  # Double Q-learning
            q_idx = self.actor_target(next_states).argmax(dim=1).unsqueeze(1)
            q_target = self.actor(next_states).gather(dim=1,
                                                      index=q_idx).detach()
        else:
            q_target = (self.actor_target(next_states).detach().max(
                axis=1)[0].unsqueeze(1))

        # target, if terminal then y_j = rewards
        y_j = rewards + self.gamma * q_target * (1 - dones)
        q_eval = self.actor(states).gather(1, actions.long())

        # loss backprop
        loss = self.criterion(q_eval, y_j)
        self.optimizer.zero_grad()
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()
        self.optimizer.step()

        # soft update target network
        soft_target_update(src_model=self.actor,
                           tgt_model=self.actor_target,
                           tau=self.tau)
        return loss.item()

    def wrap_models(self) -> None:
        """Wraps models with the accelerator."""
        if self.accelerator is not None:
            self.actor, self.actor_target, self.optimizer = self.accelerator.prepare(
                self.actor, self.actor_target, self.optimizer)

    def unwrap_models(self) -> None:
        """Unwraps models from the accelerator."""
        if self.accelerator is not None:
            self.actor = self.accelerator.unwrap_model(self.actor)
            self.actor_target = self.accelerator.unwrap_model(
                self.actor_target)
            self.optimizer = unwrap_optimizer(self.optimizer,
                                              self.actor,
                                              lr=self.learning_rate)

    def save_checkpoint(self, path: str) -> None:
        """Saves a checkpoint of agent properties and network weights to
        path."""
        network_info = {
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        torch.save(network_info, path)

    def load_checkpoint(self, path: str) -> None:
        """Loads saved agent properties and network weights from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.optimizer = optim.Adam(self.actor.parameters(),
                                    lr=self.learning_rate)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(
            checkpoint['actor_target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
