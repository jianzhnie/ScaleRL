import gymnasium as gym
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
