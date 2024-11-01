from dataclasses import dataclass
from typing import Any, Dict

import gymnasium as gym
import numpy as np

from scalerl.envs.vector.pz_async_vec_env import AsyncPettingZooVecEnv


@dataclass
class EpisodeMetrics:
    """Container for episode-level metrics in vectorized environments."""

    num_envs: int

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset metrics for new episodes."""
        self.returns = np.zeros(self.num_envs, dtype=np.float32)
        self.lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.completed_returns = []
        self.completed_lengths = []
        self.episode_count = 0

    def update(self, rewards: np.ndarray, terminated: np.ndarray,
               truncated: np.ndarray) -> None:
        """Update metrics based on environment step results.

        Args:
            rewards: Reward received
            terminated: Environment termination flags
            truncated: Environment truncation flags
        """
        if not isinstance(rewards, np.ndarray):
            rewards = np.array([rewards])
        if not isinstance(terminated, np.ndarray):
            terminated = np.array([terminated])
        if not isinstance(truncated, np.ndarray):
            truncated = np.array([truncated])

        # Update cumulative metrics
        self.returns += rewards
        self.lengths += 1

        done = np.logical_or(terminated, truncated)
        for i in range(self.num_envs):
            if done[i]:
                self.completed_returns.append(float(self.returns[i]))
                self.completed_lengths.append(int(self.lengths[i]))
                self.returns[i] = 0
                self.lengths[i] = 0
                self.episode_count += 1

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current episode metrics for active (incomplete) episodes."""
        return {
            'current_returns': self.returns.copy(),
            'current_lengths': self.lengths.copy(),
        }

    def get_episode_info(self) -> Dict[str, float]:
        """Get current episode metrics summary."""
        if not self.completed_returns:
            return {
                'episode_cnt': 0,
                'episode_return': 0.0,
                'episode_length': 0.0
            }
        return {
            'episode_cnt': self.episode_count,
            'episode_return': float(np.mean(self.completed_returns)),
            'episode_length': float(np.mean(self.completed_lengths)),
        }

    def __str__(self) -> str:
        """Return a human-readable string representation of current metrics."""
        metrics = self.get_episode_info()
        return (f"Episodes: {metrics['episode_count']}\n"
                f"Mean Return: {metrics['episode_return']:.2f}\n"
                f"Mean Length: {metrics['episode_length']:.1f}")


def make_vect_envs(env_name, num_envs=1):
    """Returns async-vectorized gym environments.

    :param env_name: Gym environment name
    :type env_name: str
    :param num_envs: Number of vectorized environments, defaults to 1
    :type num_envs: int, optional
    """
    return gym.vector.AsyncVectorEnv(
        [lambda: gym.make(env_name) for i in range(num_envs)])


def make_multi_agent_vect_envs(env, num_envs=1, **env_kwargs):
    """Returns async-vectorized PettingZoo parallel environments.

    :param env: PettingZoo parallel environment object
    :type env: pettingzoo.utils.env.ParallelEnv
    :param num_envs: Number of vectorized environments, defaults to 1
    :type num_envs: int, optional
    """
    env_fns = [lambda: env(**env_kwargs) for _ in range(num_envs)]
    return AsyncPettingZooVecEnv(env_fns=env_fns)


def make_skill_vect_envs(env_name, skill, num_envs=1):
    """Returns async-vectorized gym environments.

    :param env_name: Gym environment name
    :type env_name: str
    :param skill: Skill wrapper to apply to environment
    :type skill: agilerl.wrappers.learning.Skill
    :param num_envs: Number of vectorized environments, defaults to 1
    :type num_envs: int, optional
    """
    return gym.vector.AsyncVectorEnv(
        [lambda: skill(gym.make(env_name)) for i in range(num_envs)])


def calculate_vectorized_scores(rewards,
                                terminations,
                                include_unterminated=False,
                                only_first_episode=True):
    episode_returns = []
    num_envs, _ = rewards.shape

    for env_index in range(num_envs):
        # Find the indices where episodes terminate for the current environment
        termination_indices = np.where(terminations[env_index] == 1)[0]

        # If no terminations, sum the entire reward array for this environment
        if len(termination_indices) == 0:
            episode_return = np.sum(rewards[env_index])
            episode_returns.append(episode_return)
            continue  # Skip to the next environment

        # Initialize the starting index for segmenting
        start_index = 0

        for termination_index in termination_indices:
            # Sum the rewards for the current episode
            episode_return = np.sum(rewards[env_index,
                                            start_index:termination_index + 1])

            # Store the episode reward
            episode_returns.append(episode_return)

            # If only the first episode is required, break after processing it
            if only_first_episode:
                break

            # Update the starting index for segmenting
            start_index = termination_index + 1

        # If include_unterminated is True, sum the rewards from the last termination index to the end
        if (not only_first_episode and include_unterminated
                and start_index < len(rewards[env_index])):
            episode_return = np.sum(rewards[env_index, start_index:])
            episode_returns.append(episode_return)

    return episode_returns
