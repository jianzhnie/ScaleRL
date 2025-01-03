import math
import multiprocessing as mp
import os
import random
import sys
import traceback
from collections import deque
from typing import Dict, List, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Ensure paths are correctly set
sys.path.append(os.getcwd())
from scalerl.envs.gym_env import make_gym_env
from scalerl.utils.logger_utils import get_logger
from scalerl.utils.lr_scheduler import LinearDecayScheduler
from scalerl.utils.utils import get_device

logger = get_logger("impala")


def ceil_to_nearest_hundred(num: int):
    return math.ceil(num / 100) * 100


class QNetwork(nn.Module):
    """A simple feedforward neural network for Q-learning.

    Args:
        obs_dim (int): Dimension of the obs space.
        action_dim (int): Dimension of the action space.
    """

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor representing the obs.

        Returns:
            torch.Tensor: Output tensor representing the Q-values for each action.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """A replay buffer to store and sample experiences.

    Args:
        buffer_size (int): Maximum size of the buffer.
    """

    def __init__(self, buffer_size: int):
        self.memory = deque(maxlen=buffer_size)

    def add(self, experience: Tuple[np.ndarray, int, float, np.ndarray, bool]):
        """Add an experience to the buffer.

        Args:
            experience (Tuple[np.ndarray, int, float, np.ndarray, bool]): A tuple containing (obs, action, reward, next_obs, done).
        """
        self.memory.append(experience)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences from the buffer.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing obs, actions, rewards, next_obs, and dones.
        """
        experiences = random.sample(self.memory, k=batch_size)
        obs, actions, rewards, next_obs, dones = zip(*experiences)
        batch = dict(
            obs=np.array(obs),
            actions=np.array(actions),
            rewards=np.array(rewards),
            next_obs=np.array(next_obs),
            dones=np.array(dones),
        )
        return batch

    def __len__(self) -> int:
        """Get the current size of the buffer.

        Returns:
            int: Current size of the buffer.
        """
        return len(self.memory)


class ParallelDQNv2:
    """Implements the IMPALA (Importance Weighted Actor-Learner Architecture)
    with DQN.

    Args:
        obs_dim (int): Dimension of the obs space.
        action_dim (int): Dimension of the action space.
        num_actors (int): Number of actor processes.
        buffer_size (int): Maximum size of the replay buffer.
        gamma (float): Discount factor for future rewards.
        batch_size (int): Number of experiences to sample for training.
        learning_rate (float): Learning rate for the optimizer.
        device (str): Device to use for training (e.g., 'cpu' or 'cuda').
    """

    def __init__(
        self,
        env_name: str = None,
        num_actors: int = 4,
        hidden_dim: int = 64,
        max_timesteps: int = 50000,
        buffer_size: int = 10000,
        eps_greedy_start: float = 1.0,
        eps_greedy_end: float = 0.01,
        eval_interval: int = 1000,
        train_log_interval: int = 1000,
        target_update_frequency: int = 2000,
        double_dqn: bool = True,
        gamma: float = 0.99,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        device: Union[torch.device, str] = "auto",
    ) -> None:
        self.num_actors = num_actors
        self.max_timesteps = max_timesteps
        self.buffer_size = buffer_size
        self.eval_interval = eval_interval
        self.train_log_interval = train_log_interval
        self.target_update_frequency = target_update_frequency
        self.double_dqn = double_dqn
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = get_device(device)

        print(f'Using {self.device} device')

        self.train_env = make_gym_env(env_id=env_name)
        self.test_env = make_gym_env(env_id=env_name)
        # Get observation and action dimensions
        obs_shape = self.test_env.observation_space.shape or (
            self.test_env.observation_space.n, )
        action_shape = self.test_env.action_space.shape or (
            self.test_env.action_space.n, )
        self.obs_dim = int(np.prod(obs_shape))
        self.action_dim = int(np.prod(action_shape))

        self.q_network = QNetwork(self.obs_dim, hidden_dim,
                                  self.action_dim).to(device)
        self.target_network = QNetwork(self.obs_dim, hidden_dim,
                                       self.action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(),
                                    lr=learning_rate)
        self.eps_greedy_start = eps_greedy_start
        self.eps_greedy_end = eps_greedy_end
        self.eps_greedy_scheduler = LinearDecayScheduler(
            eps_greedy_start,
            eps_greedy_end,
            max_steps=max_timesteps * 0.9,
        )
        self.eps_greedy = eps_greedy_start

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

        self.eps_greedy = max(self.eps_greedy_scheduler.step(), self.eps_greedy_end)

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
            q_values = self.q_network(obs)
        action = torch.argmax(q_values, dim=1).item()
        return action

    def actor_process(
        self, actor_id: int, env: gym.Env, data_queue: mp.Queue, stop_event: mp.Event
    ) -> None:
        """Actor process that interacts with the environment and collects
        experiences.

        Args:
            actor_id (int): ID of the actor.
            env (gym.Env): Environment to interact with.
            data_queue (mp.Queue): Queue to send collected experiences to the learner.
            stop_event (mp.Event): Event to signal the actor to stop.
        """
        logger.info(f'Actor {actor_id} started')
        try:
            while not stop_event.is_set():
                obs, _ = self.train_env.reset(seed=actor_id)
                buffer: List[Tuple[np.ndarray, int, float, np.ndarray,
                                   bool]] = []
                done = False
                while not done:
                    action = self.get_action(obs)
                    next_obs, reward, terminal, truncated, info = self.train_env.step(
                        action)
                    done = terminal or truncated

                    if info and 'episode' in info:
                        info_item = {
                            k: v.item()
                            for k, v in info['episode'].items()
                        }
                        episode_reward = info_item['r']
                        episode_step = info_item["l"]

                    with self.global_step.get_lock():
                        self.global_step.value += 1
                    buffer.append((obs, action, reward, next_obs, done))
                    obs = next_obs
                if buffer:
                    data_queue.put(buffer)

                global_step = ceil_to_nearest_hundred(self.global_step.value)
                if actor_id == 0 and global_step % self.train_log_interval == 0:
                    logger.info(
                        "Actor {}: , episode step: {}, episode reward: {}".format(
                            actor_id, episode_step, episode_reward
                        ),
                    )

        except Exception as e:
            logger.error(f'Exception in actor process {actor_id}: {e}')
            traceback.print_exc()

    def learn(self, batch: Dict[str, np.array]) -> None:
        obs = torch.tensor(batch['obs'], dtype=torch.float32).to(self.device)
        actions = (torch.tensor(batch['actions'],
                                dtype=torch.long).unsqueeze(1).to(self.device))
        rewards = (torch.tensor(batch['rewards'],
                                dtype=torch.float32).unsqueeze(1).to(
                                    self.device))
        next_obs = torch.tensor(batch['next_obs'],
                                dtype=torch.float32).to(self.device)
        dones = (torch.tensor(
            batch['dones'], dtype=torch.float32).unsqueeze(1).to(self.device))

        # Compute current Q values
        current_q_values = self.q_network(obs).gather(1, actions)
        # Compute target Q values
        if self.double_dqn:
            with torch.no_grad():
                next_action = self.q_network(next_obs).max(dim=1,
                                                           keepdim=True)[1]
                next_q_values = self.target_network(next_obs).gather(
                    1, next_action)
        else:
            with torch.no_grad():
                next_q_values = self.target_network(next_obs).max(
                    dim=1, keepdim=True)[0]

        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values, reduction='mean')

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        learn_result = {
            'loss': loss.item(),
        }
        return learn_result

    def learner_process(self, data_queue: mp.Queue, stop_event: mp.Event):
        """Learner process that trains the Q-network using experiences from the
        actors.

        Args:
            data_queue (mp.Queue): Queue to receive experiences from actors.
            stop_event (mp.Event): Event to signal the learner to stop.
        """

        try:
            while (
                self.global_step.value < self.max_timesteps and not stop_event.is_set()
            ):
                try:
                    # Non-blocking with timeout
                    data = data_queue.get()
                except data_queue.Empty:
                    continue  # 如果队列为空，继续循环

                for experience in data:
                    self.replay_buffer.add(experience)

                global_step = ceil_to_nearest_hundred(self.global_step.value)

                if len(self.replay_buffer) >= self.batch_size:
                    batch = self.replay_buffer.sample(self.batch_size)
                    learn_result = self.learn(batch)

                    if global_step % self.target_update_frequency == 0:
                        self.target_network.load_state_dict(
                            self.q_network.state_dict())

                    if global_step % self.train_log_interval == 0:
                        logger.info(
                            f"Step {global_step}: Train results: {learn_result}"
                        )

                if global_step % self.eval_interval == 0:
                    eval_results = self.evaluate()
                    logger.info(
                        f"Step {global_step}: Evaluation results: {eval_results}"
                    )

        except Exception as e:
            logger.error(f'Exception in learner process: {e}')
        finally:
            logger.info('Learner process is shutting down')

    def evaluate(self, n_eval_episodes: int = 5) -> dict[str, float]:
        """Evaluate the model on the test environment.

        Args:
            n_eval_episodes (int): Number of episodes to evaluate.

        Returns:
            dict[str, float]: Evaluation results.
        """
        eval_rewards = []
        eval_steps = []
        for _ in range(n_eval_episodes):
            obs, info = self.test_env.reset(seed=random.randint(0, 100))
            done = False
            episode_reward = 0.0
            episode_step = 0
            while not done:
                action = self.predict(obs)
                next_obs, reward, terminated, truncated, info = self.test_env.step(
                    action)
                obs = next_obs
                done = terminated or truncated
                if info and 'episode' in info:
                    info_item = {
                        k: v.item()
                        for k, v in info['episode'].items()
                    }
                    episode_reward = info_item['r']
                    episode_step = info_item["l"]
            eval_rewards.append(episode_reward)
            eval_steps.append(episode_step)

        return {
            'reward_mean': np.mean(eval_rewards),
            'reward_std': np.std(eval_rewards),
            'length_mean': np.mean(eval_steps),
            'length_std': np.std(eval_steps),
        }

    def run(self) -> None:
        """Run the IMPALA DQN algorithm."""

        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.global_step = mp.Value('i', 0)
        self.data_queue = mp.Queue(maxsize=500)

        stop_event = mp.Event()
        actor_processes = []
        for actor_id in range(self.num_actors):
            train_env = make_gym_env(env_id="CartPole-v0")
            actor = mp.Process(
                target=self.actor_process,
                args=(actor_id, train_env, self.data_queue, stop_event),
            )
            actor.start()
            actor_processes.append(actor)

        learner = mp.Process(
            target=self.learner_process, args=(self.data_queue, stop_event)
        )
        learner.start()

        try:
            learner.join()
        except KeyboardInterrupt:
            logger.info(
                'Keyboard interrupt received, stopping all processes...')
        finally:
            stop_event.set()
            for actor in actor_processes:
                actor.join(timeout=1)  # 给予一定的时间让进程正常结束
                if actor.is_alive():
                    logger.warning(
                        f'Actor process {actor.pid} did not terminate, force terminating...'
                    )
                    actor.terminate()
            learner.join(timeout=1)
            if learner.is_alive():
                logger.warning(
                    'Learner process did not terminate, force terminating...')
                learner.terminate()
            logger.info('All processes have been stopped.')


if __name__ == '__main__':
    impala_dqn = ParallelDQNv2(env_name="CartPole-v0", num_actors=10)
    impala_dqn.run()
