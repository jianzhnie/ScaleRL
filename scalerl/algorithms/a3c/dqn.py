import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.optim as optim
from agents.algorithms.base import Agent
from networks.network import Actor
from utils.replaybuffer import ReplayBuffer
from utils.utils import convert_to_tensor, make_transition


class DQN(Agent):
    """Deep Q-Network (DQN) agent implementation.

    This class extends the Agent class and provides specific implementations
    for the DQN algorithm.
    """

    def __init__(
        self,
        writer,
        device: torch.device,
        state_dim: int,
        action_dim: int,
        args: Dict[str, Any],
        epsilon: float,
    ):
        """Initialize the DQN agent.

        Args:
            writer: TensorBoard writer for logging.
            device (torch.device): Device to run the computations on (CPU or GPU).
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            args (Dict[str, Any]): Additional arguments required for initialization.
            epsilon (float): Exploration rate for epsilon-greedy policy.
        """
        super(DQN, self).__init__(state_dim, action_dim, args)
        self.args = args
        self.device = device
        self.epsilon = epsilon
        self.action_dim = action_dim

        # Initialize Q-network and target Q-network
        self.q_network = Actor(
            self.args['layer_num'],
            state_dim,
            action_dim,
            self.args['hidden_dim'],
            self.args['activation_function'],
            self.args['last_activation'],
            self.args['trainable_std'],
        )
        self.target_q_network = Actor(
            self.args['layer_num'],
            state_dim,
            action_dim,
            self.args['hidden_dim'],
            self.args['activation_function'],
            self.args['last_activation'],
            self.args['trainable_std'],
        )
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(),
                                    lr=self.args['lr'])

        # Initialize replay buffer
        if self.args['discrete']:
            action_dim = 1
        self.data = ReplayBuffer(
            max_size=(self.args['traj_length'] - self.args['n_step'] + 1),
            state_dim=state_dim,
            num_action=action_dim,
            n_step=self.args['n_step'],
            args=self.args,
        )
        self.update_num = 0

    def get_q(self, x: torch.Tensor) -> torch.Tensor:
        """Get the Q-values for the given state.

        Args:
            x (torch.Tensor): The current state.

        Returns:
            torch.Tensor: The Q-values for the given state.
        """
        x, _ = self.q_network(x)
        return x

    def get_td_error(self,
                     data: Dict[str, Any],
                     weights: Optional[np.ndarray] = None) -> torch.Tensor:
        """Compute the temporal difference (TD) error for the given data.

        Args:
            data (Dict[str, Any]): Dictionary containing state, action, reward, next_state, and done.
            weights (Optional[np.ndarray]): Optional weights for prioritized experience replay.

        Returns:
            torch.Tensor: The TD error.
        """
        state, action, reward, next_state, done = convert_to_tensor(
            self.device,
            data['state'],
            data['action'],
            data['reward'],
            data['next_state'],
            data['done'],
        )
        action = action.type(torch.int64)
        q = self.get_q(state)
        q_action = q.gather(1, action)
        target = reward + (1 -
                           done) * self.args['gamma'] * self.target_q_network(
                               next_state)[0].max(1)[0].unsqueeze(1)

        beta = 1
        n = torch.abs(q_action - target.detach())
        cond = n < beta
        loss = torch.where(cond, 0.5 * n**2 / beta, n - 0.5 * beta)

        if isinstance(weights, np.ndarray):
            return torch.tensor(weights).to(self.device) * loss
        else:
            return loss

    def get_action(self, x: Union[torch.Tensor, np.ndarray]) -> int:
        """Get the action to take given the current state using an epsilon-
        greedy policy.

        Args:
            x (Union[torch.Tensor, np.ndarray]): The current state.

        Returns:
            int: The action to take.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32).to(self.device)
            return self.get_q(x).argmax().item()

    def get_buffer_size(self) -> int:
        """Get the current size of the replay buffer.

        Returns:
            int: The current size of the replay buffer.
        """
        return self.data.data_idx

    def get_trajectories(self) -> Dict[str, Any]:
        """Get trajectories from the replay buffer.

        Returns:
            Dict[str, Any]: Dictionary containing the sampled trajectories.
        """
        data = self.data.sample(False)
        return data

    def train_network(
        self, data: Tuple[List[Any], np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Train the Q-network using the given data.

        Args:
            data (Tuple[List[Any], np.ndarray, np.ndarray]): Tuple containing mini-batch, indexes, and weights.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing indexes and TD errors.
        """
        mini_batch, idxs, is_weights = data
        mini_batch = np.array(mini_batch, dtype=object).transpose()
        state = np.vstack(mini_batch[0])
        action = np.vstack(mini_batch[1])
        reward = np.vstack(mini_batch[2])
        next_state = np.vstack(mini_batch[3])
        done = np.vstack(mini_batch[4])
        log_prob = np.zeros((1, 1))  # Placeholder for log probabilities

        data = make_transition(state, action, reward, next_state, done,
                               log_prob)
        td_error = self.get_td_error(data, is_weights.reshape(-1, 1))
        self.optimizer.zero_grad()
        td_error.mean().backward()
        self.optimizer.step()
        self.update_num += 1

        if self.update_num % self.args['target_update_cycle'] == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        return idxs, td_error.detach().cpu().numpy()
