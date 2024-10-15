from typing import Optional

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.spaces.box import Box


def create_atari_env(env_id: str) -> gym.Env:
    """Creates and returns an Atari environment wrapped with rescaling and
    normalization.

    Args:
        env_id (str): The ID of the Atari environment to create.

    Returns:
        Env: The Atari environment wrapped in rescaling and normalization wrappers.
    """
    env = gym.make(env_id)
    env = AtariRescale42x42(
        env)  # Rescale the environment's observations to 42x42
    env = NormalizedEnv(env)  # Normalize the environment's observations
    return env


def _process_frame42(frame: np.ndarray) -> np.ndarray:
    """Processes an input Atari frame by cropping, resizing, and normalizing it
    to 42x42 grayscale.

    Args:
        frame (np.ndarray): The raw frame (image) from the Atari environment.

    Returns:
        np.ndarray: The processed 42x42 grayscale frame with pixel values normalized to [0, 1].
    """
    # Crop the image to focus on the game area (remove the scoreboard)
    frame = frame[34:34 + 160, :160]

    # Resize the frame to 80x80 first, then downsample to 42x42
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))

    # Convert to grayscale by taking the mean across the color channels
    frame = frame.mean(2, keepdims=True)

    # Normalize the frame values to the range [0, 1]
    frame = frame.astype(np.float32)
    frame *= 1.0 / 255.0

    # Change the channel order from HxWxC to CxHxW (required for PyTorch)
    frame = np.moveaxis(frame, -1, 0)

    return frame


class AtariRescale42x42(gym.ObservationWrapper):
    """A Gym ObservationWrapper that resizes the Atari game frames to 42x42
    grayscale."""

    def __init__(self, env: Optional[gym.Env] = None) -> None:
        """Initialize the AtariRescale42x42 wrapper.

        Args:
            env (Optional[Env]): The environment to wrap.
        """
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(
            0.0, 1.0, [1, 42, 42],
            dtype=np.float32)  # Update observation space

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Process the observation using the _process_frame42 function.

        Args:
            observation (np.ndarray): The raw observation from the environment.

        Returns:
            np.ndarray: The processed 42x42 grayscale observation.
        """
        return _process_frame42(observation)


class NormalizedEnv(gym.ObservationWrapper):
    """A Gym ObservationWrapper that normalizes the observations based on
    running mean and std."""

    def __init__(self, env: Optional[gym.Env] = None) -> None:
        """Initialize the NormalizedEnv wrapper.

        Args:
            env (Optional[Env]): The environment to wrap.
        """
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0.0  # Running mean of the observations
        self.state_std = 0.0  # Running standard deviation of the observations
        self.alpha = 0.9999  # Decay rate for running mean and std
        self.num_steps = 0  # Number of steps to correct bias

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Normalize the observation by updating the running mean and std.

        Args:
            observation (np.ndarray): The raw observation from the environment.

        Returns:
            np.ndarray: The normalized observation.
        """
        # Update step counter
        self.num_steps += 1

        # Update running mean and standard deviation with bias correction
        self.state_mean = (self.alpha * self.state_mean +
                           (1 - self.alpha) * observation.mean())
        self.state_std = (self.alpha * self.state_std +
                          (1 - self.alpha) * observation.std())

        # Bias correction for running mean and standard deviation
        unbiased_mean = self.state_mean / (1 - self.alpha**self.num_steps)
        unbiased_std = self.state_std / (1 - self.alpha**self.num_steps)

        # Normalize the observation
        return (observation - unbiased_mean) / (unbiased_std + 1e-8)
