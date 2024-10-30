import gymnasium as gym
import numpy as np

from scalerl.envs.vector.pz_async_vec_env import AsyncPettingZooVecEnv


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
    episode_rewards = []
    num_envs, _ = rewards.shape

    for env_index in range(num_envs):
        # Find the indices where episodes terminate for the current environment
        termination_indices = np.where(terminations[env_index] == 1)[0]

        # If no terminations, sum the entire reward array for this environment
        if len(termination_indices) == 0:
            episode_reward = np.sum(rewards[env_index])
            episode_rewards.append(episode_reward)
            continue  # Skip to the next environment

        # Initialize the starting index for segmenting
        start_index = 0

        for termination_index in termination_indices:
            # Sum the rewards for the current episode
            episode_reward = np.sum(rewards[env_index,
                                            start_index:termination_index + 1])

            # Store the episode reward
            episode_rewards.append(episode_reward)

            # If only the first episode is required, break after processing it
            if only_first_episode:
                break

            # Update the starting index for segmenting
            start_index = termination_index + 1

        # If include_unterminated is True, sum the rewards from the last termination index to the end
        if (not only_first_episode and include_unterminated
                and start_index < len(rewards[env_index])):
            episode_reward = np.sum(rewards[env_index, start_index:])
            episode_rewards.append(episode_reward)

    return episode_rewards
