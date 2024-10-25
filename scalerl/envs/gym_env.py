import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics


def make_gym_env(
    env_id: str,
    seed: int = 42,
    capture_video: bool = False,
    save_video_dir: str = 'work_dir',
    save_video_name: str = 'test',
) -> RecordEpisodeStatistics:
    """Create and wrap the environment with necessary wrappers.

    Args:
        env_id (str): ID of the environment.
        seed (int): Random seed.
        capture_video (bool): Whether to capture video.
        save_video_dir (str): Directory to save video.
        save_video_name (str): Name of the video file.

    Returns:
        RecordEpisodeStatistics: Wrapped environment.
    """
    if capture_video:
        env = gym.make(env_id, render_mode='rgb_array')
        env = gym.wrappers.RecordVideo(env,
                                       f'{save_video_dir}/{save_video_name}')
    else:
        env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    return env


class GymEnvWrapper(gym.Env):
    """A wrapper class for the Gymnasium environment.

    Attributes:
        obs_dim (int): The dimension of the obs space.
        action_dim (int): The dimension of the action space.
        is_discrete (bool): Whether the action space is discrete.
        max_episode_steps (int): The maximum number of steps per episode.
        can_run (bool): Whether the environment is ready to run.
        obs (Optional[Union[int, float, Tuple[float]]]): The current obs of the environment.
    """

    def __init__(self, env: gym.Env) -> None:
        """Initializes the Environment with the given Gymnasium environment.

        Args:
            env_name (str): The name of the Gymnasium environment.
        """
        self.env = env
        self.obs_shape = env.observation_space.shape or env.observation_space.n
        self.action_shape = env.action_space.shape or env.action_space.n
        self.obs_dim = int(np.prod(self.obs_shape))

        # Determine action space type and dimension
        if isinstance(self.env.action_space,
                      gym.spaces.box.Box):  # Continuous action space
            self.action_dim = self.env.action_space.shape[0]
            self.is_discrete = False
        else:  # Discrete action space
            self.action_dim = self.env.action_space.n
            self.is_discrete = True
