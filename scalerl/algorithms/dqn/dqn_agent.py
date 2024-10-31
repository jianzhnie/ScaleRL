import copy
import random
from typing import Dict, List, Optional, Union

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
    """Deep Q-Network (DQN) Agent for Reinforcement Learning.

    Implements DQN and Double DQN methods for value-based RL.

    Attributes:
        args (DQNArguments): Hyperparameters for DQN.
        device (Union[str, torch.device]): Device used for computation.
        accelerator (Optional[Accelerator]): Accelerator for distributed training.
        actor (QNet): Main network for estimating Q-values.
        actor_target (QNet): Target network to provide stable learning targets.
        optimizer (torch.optim.Optimizer): Optimizer for training the network.
        eps_greedy_scheduler (LinearDecayScheduler): Schedules epsilon decay.
        criterion (nn.Module): Loss function, Mean Squared Error.
    """

    def __init__(
        self,
        args: DQNArguments,
        state_shape: Union[int, List[int]],
        action_shape: Union[int, List[int]],
        accelerator: Optional[Accelerator] = None,
        device: Optional[Union[str, torch.device]] = 'auto',
    ) -> None:
        """Initializes the DQN agent.

        Args:
            args (DQNArguments): Configuration parameters for the DQN.
            state_shape (Union[int, List[int]]): Dimension(s) of state space.
            action_shape (Union[int, List[int]]): Dimension(s) of action space.
            accelerator (Optional[Accelerator], optional): Used for model wrapping in distributed setups.
            device (Optional[Union[str, torch.device]], optional): Device for model computation, default is 'auto'.
        """
        super().__init__(args)

        self.args: DQNArguments = args
        self.device = device
        self.accelerator = accelerator

        # Initialize counters and parameters
        self.learner_update_step = 0
        self.target_model_update_step = 0
        self.eps_greedy = args.eps_greedy_start
        self.learning_rate = args.learning_rate

        # Flatten state and action shapes
        self.obs_dim = int(np.prod(state_shape))
        self.action_dim = int(np.prod(action_shape))

        # Set up main (actor) and target networks
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
        """Chooses the next action using an epsilon-greedy policy.

        Args:
            obs (np.ndarray): The current observation/state.

        Returns:
            np.ndarray: The chosen action(s) based on epsilon-greedy exploration.
        """
        if random.random() < self.eps_greedy:
            # Random action for exploration
            action = np.argmax(np.random.uniform(0, 1,
                                                 (len(obs), self.action_dim)),
                               axis=1)
        else:
            # Exploitative action from Q-network
            action = self.predict(obs)
        return action

    def predict(self, obs: np.array) -> np.ndarray:
        """Computes action from the Q-network.

        Args:
            obs (torch.Tensor): Observation tensor.

        Returns:
            np.ndarray: Computed action(s) from the network.
        """
        obs_tensor = torch.from_numpy(obs).float()
        if self.accelerator is None:
            obs_tensor = obs_tensor.to(self.device)
        else:
            obs_tensor = obs_tensor.to(self.accelerator.device)

        if len(obs_tensor.size()) < 2:
            obs_tensor = obs_tensor.unsqueeze(0)
        with torch.no_grad():
            action_values = self.actor(obs_tensor).cpu().data.numpy()
            action = np.argmax(action_values, axis=-1)
        return action

    def learn(self, experiences: List[torch.Tensor]) -> Dict[str, float]:
        """Trains the network on a batch of experiences.

        Args:
            experiences (List[torch.Tensor]): Batch of experience tensors.

        Returns:
            Dict[str, float]: Training statistics, including loss.
        """
        obs, actions, rewards, next_obs, dones = experiences
        if self.accelerator is not None:
            obs, actions, rewards, next_obs, dones = (
                obs.to(self.accelerator.device),
                actions.to(self.accelerator.device, dtype=torch.long),
                rewards.to(self.accelerator.device),
                next_obs.to(self.accelerator.device),
                dones.to(self.accelerator.device),
            )

        with torch.no_grad():
            if self.args.double_dqn:
                greedy_action = self.actor(next_obs).max(dim=1,
                                                         keepdim=True)[1]
                next_q_values = self.actor_target(next_obs).gather(
                    dim=1, index=greedy_action)
            else:
                next_q_values = self.actor_target(next_obs).max(
                    dim=1, keepdim=True)[0]

        # Calculate target Q-value: reward if terminal, else discounted future value
        target_q_values = rewards + self.args.gamma * next_q_values * (1 -
                                                                       dones)
        current_q_values = self.actor(obs).gather(1, actions.long())

        # Loss computation and backward pass
        loss = self.criterion(target_q_values, current_q_values)
        self.optimizer.zero_grad()
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()

        # Optional gradient clipping
        if self.args.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                           self.args.max_grad_norm)
        self.optimizer.step()

        # Periodic target network update
        if self.learner_update_step % self.args.target_update_frequency == 0:
            soft_target_update(self.actor, self.actor_target,
                               self.args.soft_update_tau)
            self.target_model_update_step += 1

        self.learner_update_step += 1

        return {'loss': loss.item()}

    def wrap_models(self) -> None:
        """Wraps models and optimizer with the accelerator."""
        if self.accelerator:
            self.actor, self.actor_target, self.optimizer = self.accelerator.prepare(
                self.actor, self.actor_target, self.optimizer)

    def unwrap_models(self) -> None:
        """Unwraps models from the accelerator for standard usage."""
        if self.accelerator:
            self.actor = self.accelerator.unwrap_model(self.actor)
            self.actor_target = self.accelerator.unwrap_model(
                self.actor_target)
            self.optimizer = unwrap_optimizer(self.optimizer,
                                              self.actor,
                                              lr=self.learning_rate)

    def save_checkpoint(self, path: str) -> None:
        """Saves agent state and weights to file.

        Args:
            path (str): Path to save checkpoint.
        """
        network_info = {
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(network_info, path)

    def load_checkpoint(self, path: str) -> None:
        """Loads saved agent properties and network weights from checkpoint.

        Args:
            path (str): Path to load checkpoint from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(
            checkpoint['actor_target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
