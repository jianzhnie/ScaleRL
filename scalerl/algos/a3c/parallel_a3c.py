from __future__ import print_function

import os
import random
import sys
import traceback
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
    """Actor-Critic Neural Network for A3C.

    This network computes policy logits (for the actor) and value estimates
    (for the critic) from input observations.
    """

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int) -> None:
        """Initialize the Actor-Critic network.

        Args:
            obs_dim (int): Dimension of the observation space.
            hidden_dim (int): Number of hidden units in the network.
            action_dim (int): Dimension of the action space.
        """
        super(ActorCriticNet, self).__init__()

        # Shared feature extraction layers
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Separate heads for actor and critic
        self.actor_linear = nn.Linear(hidden_dim, action_dim)
        self.critic_linear = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the network.

        Args:
            obs (torch.Tensor): Observations from the environment.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Logits for policy (actor) and value estimate (critic).
        """
        feature = self.feature_net(obs)
        logits = self.actor_linear(feature)  # Policy logits for actor
        value = self.critic_linear(feature)  # Value estimation for critic
        return logits, value


class ParallelA3C:
    """A3C Trainer Class to manage asynchronous training and testing.

    This class manages multiple workers for parallel training using the A3C
    algorithm.
    """

    def __init__(
        self,
        env_name: str = 'CartPole-v0',
        num_workers: int = 10,
        hidden_dim: int = 64,
        max_episode_size: int = 1000,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 50.0,
        rollout_steps: int = 200,
        no_shared: bool = True,
        eval_interval: int = 10,
        num_episodes_eval: int = 5,
        train_log_interval: int = 10,
        eval_log_interval: int = 10,
        device: Union[torch.device, str] = 'auto',
    ) -> None:
        """Initialize the A3C trainer with environments, shared models, and
        optimizer.

        Args:
            env_name (str): Name of the Gym environment.
            num_workers (int): Number of worker processes.
            hidden_dim (int): Number of hidden units in the network.
            max_episode_size (int): Maximum number of episodes for training.
            learning_rate (float): Learning rate for optimization.
            gamma (float): Discount factor.
            entropy_coef (float): Coefficient for entropy regularization.
            value_loss_coef (float): Coefficient for critic value loss.
            max_grad_norm (float): Maximum gradient norm for clipping.
            rollout_steps (int): Number of steps per rollout.
            no_shared (bool): Whether to use shared models across processes.
            eval_interval (int): Interval for evaluation.
            train_log_interval (int): Interval for logging training information.
            device (Union[torch.device, str]): Device for model (CPU, GPU).
        """
        # Initialize parameters and settings
        self.seed = random.randint(0, 100)
        self.env_name = env_name
        self.num_workers = num_workers
        self.hidden_dim = hidden_dim
        self.max_episode_size = max_episode_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_steps = rollout_steps
        self.no_shared = no_shared
        self.eval_interval = eval_interval
        self.num_episodes_eval = num_episodes_eval
        self.train_log_interval = train_log_interval
        self.eval_log_interval = eval_log_interval

        # Get the appropriate device (CPU/GPU)
        self.device = get_device(device)
        print(f'Using {self.device} device')

        # Create shared counters and reward queue
        self.global_episode = mp.Value('i', 0)
        self.result_queue = mp.Queue()
        self.global_reward_queue = mp.Value('d', 0)

        # Create environment instances for training and testing
        self.env = make_gym_env(self.env_name)
        self.train_env = make_gym_env(env_id=self.env_name)
        self.test_env = make_gym_env(env_id=self.env_name)

        # Get dimensions of observation and action spaces
        obs_shape = self.env.observation_space.shape or (
            self.env.observation_space.n, )
        action_shape = self.env.action_space.shape or (
            self.env.action_space.n, )
        self.obs_dim = int(np.prod(obs_shape))
        self.action_dim = int(np.prod(action_shape))

        # Initialize local and shared models for actor-critic
        self.local_model = ActorCriticNet(self.obs_dim, self.hidden_dim,
                                          self.action_dim).to(self.device)
        self.shared_model = ActorCriticNet(self.obs_dim, self.hidden_dim,
                                           self.action_dim).to(self.device)
        self.local_model.share_memory()  # Share the model across processes
        self.shared_model.share_memory()

        # Optimizer setup
        if not self.no_shared:
            self.optimizer = SharedAdam(self.shared_model.parameters(),
                                        lr=self.learning_rate)
            self.optimizer.share_memory()
        else:
            self.optimizer = None

    def get_action(self, obs: torch.Tensor) -> int:
        """Select an action using the local policy model.

        Args:
            obs (torch.Tensor): The observation input.

        Returns:
            int: The action selected by the policy.
        """
        if obs.ndim == 1:
            obs = np.expand_dims(obs, axis=0)  # Add batch dimension

        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, _ = self.local_model(obs)
        probs = F.softmax(logits, dim=-1)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def predict(self, obs: np.ndarray) -> int:
        """Predict the most likely action from the policy.

        Args:
            obs (np.ndarray): Observation from the environment.

        Returns:
            int: Predicted action based on the highest probability.
        """
        if obs.ndim == 1:
            obs = np.expand_dims(obs, axis=0)

        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, _ = self.local_model(obs)
        probs = F.softmax(logits, dim=-1)
        action = probs.argmax(dim=-1)
        return action.item()

    def sync_with_shared_model(self, local_model: nn.Module,
                               shared_model: nn.Module) -> None:
        """Synchronize local model weights with the shared global model.

        Args:
            local_model (nn.Module): Local model to sync.
            shared_model (nn.Module): Shared global model.
        """
        local_model.load_state_dict(shared_model.state_dict(), strict=False)

    def ensure_shared_grads(self, local_model: nn.Module,
                            shared_model: nn.Module) -> None:
        """Ensure that gradients from the local model are applied to the shared
        model.

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
        """Compute loss for both the actor and critic based on transitions.

        Args:
            transition_dict (Dict[str, List[np.ndarray]]): Transitions containing obs, actions, rewards, etc.

        Returns:
            torch.Tensor: Total loss for the actor and critic.
        """
        device = self.device

        # Convert transition data to tensors
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

        # Model outputs for current obs
        logits, values = self.local_model(obs)
        _, next_values = self.local_model(next_obs)

        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)

        # Calculate advantages and returns
        td_target = rewards + self.gamma * next_values * (1 - dones)
        delta = td_target - values
        advantages = delta.detach()

        # Actor loss
        actor_loss = -(log_probs * advantages).mean()

        # Critic loss
        critic_loss = F.mse_loss(values, td_target.detach())

        # Entropy regularization (for exploration)
        entropy = dist.entropy().mean()
        total_loss = (actor_loss + self.value_loss_coef * critic_loss -
                      self.entropy_coef * entropy)
        return total_loss

    def rollout(self, seed: int,
                rollout_steps: int) -> Dict[str, List[np.ndarray]]:
        """Perform a rollout to collect experiences for training.

        Args:
            env: The environment in which to collect the rollout.
            rollout_steps (int): Maximum number of steps to take in the environment.

        Returns:
            Dict[str, List[np.ndarray]]: A dictionary containing the rollout's transitions.
        """
        transition_dict = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'next_obs': [],
            'dones': [],
        }
        obs, _ = self.train_env.reset(seed=seed)
        for step in range(rollout_steps):
            action = self.get_action(obs)
            next_obs, reward, terminal, truncated, info = self.train_env.step(
                action)
            done = terminal or truncated

            # Append transition data
            transition_dict['obs'].append(obs)
            transition_dict['actions'].append(action)
            transition_dict['rewards'].append(reward)
            transition_dict['next_obs'].append(next_obs)
            transition_dict['dones'].append(done)

            obs = next_obs
            if done:
                break
        return transition_dict

    def train(
            self,
            worker_id: int,
            optimizer: Optional[torch.optim.Optimizer] = None,
            mp_lock: mp.Lock = mp.Lock(),
            stop_event: mp.Event = None,
    ) -> None:
        """Train the A3C model using parallel workers.

        Args:
            max_episodes (int): Maximum number of training episodes.
        """
        logger.info(f'Train Worker {worker_id} starting...')

        seed = self.seed + worker_id
        torch.manual_seed(seed)
        self.local_model.train()
        try:
            while (self.global_episode.value < self.max_episode_size
                   and not stop_event.is_set()):
                # Update global episode counter
                with self.global_episode.get_lock():
                    self.global_episode.value += 1

                if optimizer is None:
                    optimizer = torch.optim.Adam(
                        self.shared_model.parameters(), lr=self.learning_rate)

                # Sync local model with the shared model
                with mp_lock:
                    self.sync_with_shared_model(self.local_model,
                                                self.shared_model)

                transition_dict = self.rollout(seed, self.rollout_steps)

                # Compute loss and backpropagate
                loss = self.compute_loss(transition_dict)
                # Zero the gradients
                optimizer.zero_grad()
                loss.backward()

                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(),
                                               self.max_grad_norm)

                # Ensure the gradients are applied to the shared model if shared mode is enabled
                if not self.no_shared:
                    self.ensure_shared_grads(self.local_model,
                                             self.shared_model)
                    optimizer.step()

                # Log training information at specified intervals
                if self.global_episode.value % self.train_log_interval == 0:
                    log_message = (
                        '[Train], gobal_episode:{}, loss:{}').format(
                            self.global_episode.value, loss.item())
                    if worker_id == 0:
                        logger.info(log_message)
        except Exception as e:
            logger.error(f'Exception in worker process {worker_id}: {e}')
            traceback.print_exc()

        logger.info(f'Worker {worker_id} completed training.')

    def evaluate(
            self,
            worker_id: int,
            num_episodes: int = 5,
            mp_lock: mp.Lock = mp.Lock(),
            stop_event: mp.Event = None,
    ) -> float:
        """Evaluate the policy by running it in the environment.

        Args:
            env: The environment for evaluation.
            num_episodes (int): Number of episodes to evaluate.

        Returns:
            float: The average reward across all evaluation episodes.
        """
        logger.info(f'Eval Worker {worker_id} starting...')
        try:
            while (self.global_episode.value < self.max_episode_size
                   and not stop_event.is_set()):
                self.local_model.eval()
                seed = self.seed + worker_id
                # Sync local model with the shared model
                with mp_lock:
                    self.sync_with_shared_model(self.local_model,
                                                self.shared_model)

                total_reward = 0.0
                total_len = 0.0
                for _ in range(num_episodes):
                    obs, info = self.test_env.reset(seed=seed)
                    done = False
                    episode_len = 0
                    episode_reward = 0
                    while not done:
                        action = self.predict(obs)
                        next_obs, reward, terminal, truncated, info = (
                            self.test_env.step(action))
                        done = terminal or truncated
                        episode_reward += reward
                        episode_len += 1
                        obs = next_obs
                    total_reward += episode_reward
                    total_len += episode_len

                avg_reward = total_reward / num_episodes
                avg_len = total_len / num_episodes
                # Log training information at specified intervals
                if self.global_episode.value % self.eval_log_interval == 0:
                    log_message = (
                        '[Eval], gobal_episode:{}, episode_length:{}, episode_rewards:{}'
                    ).format(self.global_episode.value, avg_len, avg_reward)
                    logger.info(log_message)
        except Exception as e:
            logger.error(f'Exception in worker process {worker_id}: {e}')
            traceback.print_exc()
        return avg_reward, avg_len

    def save_model(self, path: str) -> None:
        """Save the trained model to a file.

        Args:
            path (str): Path where the model will be saved.
        """
        torch.save(self.local_model.state_dict(), path)
        logger.info(f'Model saved to {path}')

    def load_model(self, path: str) -> None:
        """Load a pre-trained model from a file.

        Args:
            path (str): Path from where the model will be loaded.
        """
        self.local_model.load_state_dict(
            torch.load(path, map_location=self.device))
        logger.info(f'Model loaded from {path}')

    def run(self) -> None:
        """Run the agent by spawning processes for training and testing."""
        stop_event = mp.Event()
        train_proc: List[mp.Process] = []
        # Start the training processes
        for worker_id in range(self.num_workers):
            p = mp.Process(
                target=self.train,
                args=(worker_id, self.optimizer, mp.Lock(), stop_event),
            )
            p.start()
            train_proc.append(p)

        eval_proc = mp.Process(
            target=self.evaluate,
            args=(self.num_workers, self.num_episodes_eval, mp.Lock(),
                  stop_event),
        )
        eval_proc.start()

        try:
            eval_proc.join()
        except KeyboardInterrupt:
            logger.info(
                'Keyboard interrupt received, stopping all processes...')
        finally:
            stop_event.set()
            for worker in train_proc:
                worker.join(timeout=1)  # 给予一定的时间让进程正常结束
                if worker.is_alive():
                    logger.warning(
                        f'Actor process {worker.pid} did not terminate, force terminating...'
                    )
                    worker.terminate()
            eval_proc.join(timeout=1)
            if eval_proc.is_alive():
                logger.warning(
                    'Learner process did not terminate, force terminating...')
                eval_proc.terminate()
            logger.info('All processes have been stopped.')


if __name__ == '__main__':
    a3c = ParallelA3C(num_workers=10, device='cpu')
    a3c.run()
