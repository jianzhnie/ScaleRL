import copy
import random
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator

from scalerl.algorithms.base import BaseAgent
from scalerl.algorithms.rl_args import DQNArguments
from scalerl.algorithms.utils.network import QNet
from scalerl.utils import LinearDecayScheduler
from scalerl.utils.algo_utils import unwrap_optimizer
from scalerl.utils.model_utils import soft_target_update


class DQNAgent(BaseAgent):
    """The DQN algorithm class.

    DQN paper: https://arxiv.org/abs/1312.5602
    """

    def __init__(
        self,
        args: DQNArguments,
        state_shape: Union[int, List[int]],
        action_shape: Union[int, List[int]],
        accelerator: Optional[Accelerator] = None,
        device: Optional[Union[str, torch.device]] = 'auto',
    ) -> None:
        """Initializes the DQN agent."""
        super().__init__(args)

        self.args: DQNArguments = args
        self.device = device
        self.accelerator = accelerator

        # Default to CPU if not specified
        self.learner_update_step = 0
        self.target_model_update_step = 0
        self.eps_greedy = args.eps_greedy_start
        self.learning_rate = args.learning_rate

        # Flatten state and action shapes
        self.obs_dim = int(np.prod(state_shape))
        self.action_dim = int(np.prod(action_shape))

        # Create the actor network and target network
        self.actor = QNet(obs_dim=self.obs_dim, action_dim=self.action_dim)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.optimizer = optim.Adam(self.actor.parameters(),
                                    lr=self.learning_rate)

        if self.accelerator is not None:
            self.wrap_models()
        else:
            self.actor = self.actor.to(self.device)
            self.actor_target = self.actor_target.to(self.device)

        self.criterion = nn.MSELoss()
        # Epsilon-greedy scheduler for exploration
        self.eps_greedy_scheduler = LinearDecayScheduler(
            start_value=args.eps_greedy_start,
            end_value=args.eps_greedy_end,
            max_steps=int(args.max_timesteps * 0.8),
        )

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Returns the next action to take in the environment.

        Epsilon is the probability of taking a random action, used for
        exploration. For epsilon-greedy behavior, set epsilon to 0.
        """
        obs = torch.from_numpy(obs).float()
        if self.accelerator is None:
            obs = obs.to(self.device)
        else:
            obs = obs.to(self.accelerator.device)
        obs = obs.unsqueeze(0)
        # epsilon-greedy
        if random.random() < self.eps_greedy:
            action = np.argmax(
                np.random.uniform(0, 1, (len(obs), self.action_dim)),
                axis=1,
            )
        else:
            action = self.predict(obs)  # Exploit
        return action

    def predict(self, obs: np.ndarray) -> np.array:
        obs = torch.from_numpy(obs).float()
        if self.accelerator is None:
            obs = obs.to(self.device)
        else:
            obs = obs.to(self.accelerator.device)

        obs = obs.unsqueeze(0)
        with torch.no_grad():
            action_values = self.actor(obs).cpu().data.numpy()
            action = np.argmax(action_values, axis=-1)
        return action

    def learn(self, experiences: List[torch.Tensor]) -> float:
        """Updates agent network parameters to learn from experiences."""
        obs, actions, rewards, next_obs, dones = experiences
        if self.accelerator is not None:
            obs = obs.to(self.accelerator.device)
            actions = actions.to(self.accelerator.device)
            rewards = rewards.to(self.accelerator.device)
            next_obs = next_obs.to(self.accelerator.device)
            dones = dones.to(self.accelerator.device)

        with torch.no_grad():
            if self.args.double_dqn:  # Double Q-learning
                greedy_action = self.actor(next_obs).max(dim=1,
                                                         keepdim=True)[1]
                next_q_values = self.actor_target(next_obs).gather(
                    dim=1, index=greedy_action)
            else:
                next_q_values = self.actor_target(next_obs).max(
                    dim=1, keepdim=True)[0]

        # target, if terminal then y_j = rewards
        target_q_values = rewards + self.args.gamma * next_q_values * (1 -
                                                                       dones)
        current_q_values = self.actor(obs).gather(1, actions.long())

        # Compute loss (Mean Squared Error between current and target Q-values)
        loss = self.criterion(target_q_values, current_q_values)
        self.optimizer.zero_grad()
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()
        # Gradient clipping (if enabled)
        if self.args.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                           self.args.max_grad_norm)
        self.optimizer.step()

        # Soft update of the target network at regular intervals
        if self.learner_update_step % self.args.target_update_frequency == 0:
            soft_target_update(self.actor, self.actor_target,
                               self.args.soft_update_tau)
            self.target_model_update_step += 1

        self.learner_update_step += 1

        learn_result = {
            'loss': loss.item(),
        }
        # Return the loss as a dictionary for logging
        return learn_result

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
