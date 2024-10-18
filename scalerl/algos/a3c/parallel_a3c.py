from __future__ import print_function

import os
import random
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

# Ensure paths are correctly set
sys.path.append(os.getcwd())

# Custom imports from project
from scalerl.algos.a3c.share_optim import SharedAdam
from scalerl.envs.gym_env import make_gym_env
from scalerl.utils.logger_utils import get_logger
from scalerl.utils.utils import get_device

logger = get_logger('a3c')


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

    def __init__(
        self,
        env_name: str = 'CartPole-v0',
        num_workers: int = 4,
        hidden_dim: int = 8,
        max_episode_size: int = 1000,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 50.0,
        rollout_steps: int = 50,
        no_shared: bool = True,
        eval_interval: int = 10,
        train_log_interval: int = 10,
        target_update_frequency: int = 10,
        device: Union[torch.device, str] = 'auto',
    ) -> None:
        """Initialize the A3C trainer with the environment, shared model, and
        optimizer.

        Args:
            args (A3CArguments): Hyperparameters and arguments for A3C.
        """
        self.seed = random.randint(0, 100)
        self.env_name = env_name
        self.num_workers = num_workers
        self.hidden_dim = hidden_dim
        self.max_episode_size = max_episode_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_steps = rollout_steps
        self.no_shared = no_shared
        self.eval_interval = eval_interval
        self.train_log_interval = train_log_interval
        self.target_update_frequency = target_update_frequency

        self.device = get_device(device)
        print(f'Using {self.device} device')

        # Create a shared queue for storing experiences
        self.exp_queue = mp.Queue(maxsize=self.max_episode_size)
        self.exp_queue_lock = mp.Lock()

        # Initialize environments for training and testing
        self.env = make_gym_env(self.env_name)
        self.train_env = make_gym_env(env_id=self.env_name)
        self.test_env = make_gym_env(env_id=self.env_name)

        # Get observation and action dimensions
        obs_shape = self.env.observation_space.shape or (
            self.env.observation_space.n, )
        action_shape = self.env.action_space.shape or (
            self.env.action_space.n, )
        self.obs_dim = int(np.prod(obs_shape))
        self.action_dim = int(np.prod(action_shape))

        # Initialize local and shared models
        self.local_model = ActorCriticNet(self.obs_dim, self.hidden_dim,
                                          self.action_dim).to(self.device)
        self.shared_model = ActorCriticNet(self.obs_dim, self.hidden_dim,
                                           self.action_dim).to(self.device)
        self.local_model.share_memory()
        self.shared_model.share_memory(
        )  # Share the model memory across processes

        # Initialize optimizer
        if not self.no_shared:
            self.optimizer = SharedAdam(self.shared_model.parameters(),
                                        lr=self.learning_rate)
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

        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
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

        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        with torch.no_grad():
            logits, _ = self.local_model(obs)
        probs = F.softmax(logits, dim=-1)
        action = probs.argmax(dim=-1)
        return action.item()

    def sync_with_shared_model(self, local_model: nn.Module,
                               shared_model: nn.Module) -> None:
        """Synchronize local model parameters with the shared model.

        Args:
            local_model (nn.Module): Local model.
            shared_model (nn.Module): Shared global model.
        """
        local_model.load_state_dict(shared_model.state_dict(), strict=False)

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
        device = self.device
        obs = torch.tensor(np.array(transition_dict['obs']),
                           dtype=torch.float32,
                           device=device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.long,
                               device=device).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float32,
                               device=device).view(-1, 1)
        next_obs = torch.tensor(np.array(transition_dict['next_obs']),
                                dtype=torch.float32,
                                device=device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float32,
                             device=device).view(-1, 1)

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
            global_episode: Optional[mp.Value] = None,
            mp_lock: mp.Lock = mp.Lock(),
    ) -> None:
        """Train the model using asynchronous advantage actor-critic.

        Args:
            worker_id (int): Worker process ID.
            optimizer (Optional[torch.optim.Optimizer]): Optimizer for updating shared model.
            global_episode (Optional[mp.Value]): Global episode counter.
            global_reward_queue (Optional[mp.Queue]): Queue to store rewards.
        """
        seed = self.seed + worker_id
        torch.manual_seed(seed)
        self.local_model.train()

        # Update global episode counter
        with global_episode.get_lock():
            global_episode.value += 1

        if optimizer is None:
            optimizer = torch.optim.Adam(self.shared_model.parameters(),
                                         lr=self.learning_rate)

        # Sync local model with the shared model

        with mp_lock:
            self.sync_with_shared_model(self.local_model, self.shared_model)

        while global_episode.value < self.max_episode_size:
            transition_dict = {
                'obs': [],
                'actions': [],
                'next_obs': [],
                'rewards': [],
                'dones': [],
            }
            obs, _ = self.train_env.reset(seed=seed)
            done = False
            episode_reward = 0
            episode_length = 0
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
                episode_length += 1

            total_loss = self.compute_loss(transition_dict)

            if self.global_episode.value % self.train_log_interval == 0:
                log_message = (
                    '[Train], gobal_episode:{}, episode_length: {}, episode_reward:{}, loss:{}'
                ).format(
                    self.global_episode.value,
                    episode_length,
                    episode_reward,
                    total_loss.item(),
                )
                logger.info(log_message)

            # Backpropagation and optimization
            self.optimizer.zero_grad()
            total_loss.backward()
            self.ensure_shared_grads(self.local_model, self.shared_model)
            self.optimizer.step()

    def test(self, worker_id: int) -> Tuple[float, int]:
        """Test the model by running it in the environment and return episode
        reward and length.

        Args:
            worker_id (int): Worker process ID.

        Returns:
            Tuple[float, int]: Episode reward and episode length.
        """
        self.local_model.eval()
        seed = self.seed + worker_id
        torch.manual_seed(seed=seed)

        self.sync_with_shared_model(self.local_model, self.shared_model)
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

        log_message = (
            '[Test], gobal_episode:{}, episode_length: {}, episode_reward:{}'
        ).format(self.global_episode.value, episode_length, episode_reward)
        logger.info(log_message)
        return episode_reward, episode_length

    def run(self) -> None:
        """Run the agent by spawning processes for training and testing."""
        processes: List[mp.Process] = []

        # Create shared counters and reward queue
        self.global_episode = mp.Value('i', 0)
        self.result_queue = mp.Queue()
        self.global_reward_queue = mp.Value('d', 0)

        # Start the training processes
        for worker_id in range(self.num_workers):
            p = mp.Process(
                target=self.train,
                args=(worker_id, self.optimizer, self.global_episode,
                      mp.Lock()),
            )
            p.start()
            processes.append(p)

        # Wait for all processes to finish
        for p in processes:
            p.join()


if __name__ == '__main__':
    a3c = ParallelA3C(num_workers=4)
    a3c.run()
