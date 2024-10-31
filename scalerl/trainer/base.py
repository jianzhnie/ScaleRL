"""Base trainer class for reinforcement learning algorithms.

Provides common functionality for training and testing RL agents.
"""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Dict, Optional, Union

import gymnasium as gym
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter

from scalerl.algorithms.base import BaseAgent
from scalerl.algorithms.rl_args import RLArguments
from scalerl.utils import (TensorboardLogger, WandbLogger, get_outdir,
                           get_text_logger)

# Type aliases for better readability
LogInfo = Dict[str, float]
LoggerType = Union[TensorboardLogger, WandbLogger]


class BaseTrainer(ABC):
    """Abstract base class for implementing reinforcement learning trainers.

    This class provides the basic structure and utilities for training RL agents,
    including logging, visualization, and model saving capabilities.

    Attributes:
        args (RLArguments): Configuration arguments for the training process
        train_env (gym.Env): Environment used for training
        test_env (gym.Env): Environment used for testing/evaluation
        agent (BaseAgent): The RL agent being trained
        accelerator (Optional[Accelerator]): Accelerator for distributed training
        text_logger: Logger for text outputs
        vis_logger (LoggerType): Logger for visualization (TensorBoard or W&B)
        writer (SummaryWriter): TensorBoard writer instance
        video_save_dir (str): Directory for saving recorded videos
        model_save_dir (str): Directory for saving model checkpoints
    """

    def __init__(
        self,
        args: RLArguments,
        train_env: gym.Env,
        test_env: gym.Env,
        agent: BaseAgent,
        accelerator: Optional[Accelerator] = None,
    ) -> None:
        """Initialize the trainer with environments and configuration.

        Args:
            args: Training configuration arguments
            train_env: Environment for training
            test_env: Environment for evaluation
            agent: The RL agent to train
            accelerator: Optional accelerator for distributed training
        """
        self.args = args
        self.train_env = train_env
        self.test_env = test_env
        self.agent = agent
        self.accelerator = accelerator

        # Set up logging directories and names
        self._setup_logging_structure()

        # Initialize loggers based on configuration
        self._initialize_loggers()

        # Set up additional directories
        self.video_save_dir = get_outdir(self.work_dir, 'video_dir')
        self.model_save_dir = get_outdir(self.work_dir, 'model_dir')

    def _setup_logging_structure(self) -> None:
        """Set up the directory structure and naming for logs."""
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        self.log_name = os.path.join(self.args.project, self.args.env_id,
                                     self.args.algo_name,
                                     timestamp).replace(os.path.sep, '_')

        self.work_dir = os.path.join(self.args.work_dir, self.args.project,
                                     self.args.env_id, self.args.algo_name)

        # Create logging directories
        self.tb_log_dir = get_outdir(self.work_dir, 'tb_log')
        self.text_log_dir = get_outdir(self.work_dir, 'text_log')
        self.text_log_file = os.path.join(self.text_log_dir,
                                          f'{self.log_name}.log')

    def _initialize_loggers(self) -> None:
        """Initialize text and visualization loggers based on configuration."""
        # Initialize text logger
        is_main_process = self.accelerator is None or self.accelerator.is_main_process
        if is_main_process:
            self.text_logger = get_text_logger(log_file=self.text_log_file,
                                               log_level='INFO')

        # Initialize visualization logger
        if self.args.logger == 'wandb':
            self._setup_wandb_logger()
        else:
            self._setup_tensorboard_logger()

    def _setup_wandb_logger(self) -> None:
        """Set up Weights & Biases logger."""
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        if self.accelerator is None or self.accelerator.is_main_process:
            self.writer = SummaryWriter(self.tb_log_dir)
            self.vis_logger = WandbLogger(
                dir=self.work_dir,
                train_interval=self.args.train_log_interval,
                test_interval=self.args.test_log_interval,
                update_interval=self.args.train_log_interval,
                save_interval=self.args.save_interval,
                project=self.args.project,
                name=self.log_name,
                config=asdict(self.args),
            )
            self.vis_logger.load(self.writer)

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

    def _setup_tensorboard_logger(self) -> None:
        """Set up TensorBoard logger."""
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        if self.accelerator is None or self.accelerator.is_main_process:
            self.writer = SummaryWriter(self.tb_log_dir)
            self.writer.add_text('args', str(self.args))
            self.vis_logger = TensorboardLogger(self.writer)

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

    @abstractmethod
    def run_train_episode(self) -> LogInfo:
        """Run a single training episode.

        Returns:
            Dictionary containing training metrics for the episode
        """
        raise NotImplementedError

    @abstractmethod
    def run_test_episode(self) -> LogInfo:
        """Run a single test episode.

        Returns:
            Dictionary containing testing metrics for the episode
        """
        raise NotImplementedError

    @abstractmethod
    def run(self) -> None:
        """Run the complete training process."""
        raise NotImplementedError

    def log_train_infos(self, infos: LogInfo, steps: int) -> None:
        """Log training information to the visualization logger.

        Args:
            infos: Dictionary of training metrics to log
            steps: Current training step number
        """
        self.vis_logger.log_train_data(infos, steps)

    def log_test_infos(self, infos: LogInfo, steps: int) -> None:
        """Log testing information to the visualization logger.

        Args:
            infos: Dictionary of testing metrics to log
            steps: Current training step number
        """
        self.vis_logger.log_test_data(infos, steps)
