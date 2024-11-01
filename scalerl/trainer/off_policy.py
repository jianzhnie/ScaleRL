import time
from typing import Any, Dict, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from scalerl.algorithms.base import BaseAgent
from scalerl.algorithms.rl_args import DQNArguments, RLArguments
from scalerl.data.replay_buffer import (MultiStepReplayBuffer,
                                        PrioritizedReplayBuffer, ReplayBuffer)
from scalerl.data.replay_data import ReplayDataset
from scalerl.data.sampler import Sampler
from scalerl.envs.env_utils import EpisodeMetrics
from scalerl.trainer.base import BaseTrainer
from scalerl.utils import ProgressBar, calculate_mean


class OffPolicyTrainer(BaseTrainer):
    """Trainer class for off-policy reinforcement learning algorithms.

    This trainer handles:
    - Experience collection and storage
    - Replay buffer management (including prioritized and n-step variants)
    - Training loop execution with distributed support
    - Evaluation and logging

    Attributes:
        replay_buffer: Stores past experiences for sampling and learning.
        n_step_buffer: Buffer for n-step learning, if enabled.
        data_sampler: Handles sampling of data for training.
        n_step_sampler: Sampler for n-step data, if n-step learning is enabled.
    """

    def __init__(
        self,
        args: RLArguments,
        train_env: gym.Env,
        test_env: gym.Env,
        agent: BaseAgent,
        accelerator: Optional[Accelerator] = None,
        device: Optional[Union[str, torch.device]] = 'auto',
    ) -> None:
        """Initializes the OffPolicyTrainer.

        Args:
            args (RLArguments): Arguments for training configuration.
            train_env (gym.Env): Training environment.
            test_env (gym.Env): Evaluation environment.
            agent (BaseAgent): RL agent implementing get_action and learn methods.
            accelerator (Optional[Accelerator]): Accelerator for distributed training.
            device (Optional[Union[str, torch.device]]): Device to use for computations.
        """
        super().__init__(args, train_env, test_env, agent, accelerator)

        # Environment configurations
        self.num_envs = getattr(train_env, 'num_envs', 1)
        self.num_test_envs = getattr(test_env, 'num_envs', 1)
        self.is_vectorised = hasattr(train_env, 'num_envs')
        self.device = device

        # Training state
        self.episode_cnt = 0
        self.global_step = 0
        self.start_time = time.time()
        self.args: DQNArguments = args

        # Initialize metrics trackers
        self.train_metrics = EpisodeMetrics(self.num_envs)
        self.eval_metrics = EpisodeMetrics(self.num_test_envs)

        # Setup replay buffers and samplers
        self._setup_replay_buffers()
        # Initialize sampling setup based on the use of accelerator
        self.data_sampler, self.n_step_sampler = self._setup_samplers()

    def _setup_replay_buffers(self) -> None:
        """Initialize replay buffers and samplers."""
        field_names = ['obs', 'action', 'reward', 'next_obs', 'done']

        # Initialize replay buffers
        if self.args.per:
            self.replay_buffer = PrioritizedReplayBuffer(
                memory_size=self.args.buffer_size,
                field_names=field_names,
                num_envs=self.num_envs,
                alpha=self.args.alpha,
                gamma=self.args.gamma,
                device=self.device,
            )
        else:
            self.replay_buffer = ReplayBuffer(
                memory_size=self.args.buffer_size,
                field_names=field_names,
                device=self.device,
            )

        # Setup n-step buffer if needed
        self.n_step_buffer = (MultiStepReplayBuffer(
            memory_size=self.args.buffer_size,
            field_names=field_names,
            num_envs=self.num_envs,
            gamma=self.args.gamma,
            device=self.device,
        ) if self.args.n_steps else None)

    def _setup_samplers(self) -> tuple[Sampler, Optional[Sampler]]:
        """Initializes samplers for data collection and training.

        Args:
            accelerator (Optional[Accelerator]): Accelerator for distributed training.

        Returns:
            tuple[Sampler, Optional[Sampler]]: Main sampler and optional n-step sampler.
        """
        if self.accelerator:
            # Distributed sampler with accelerator support
            replay_dataset = ReplayDataset(self.replay_buffer,
                                           batch_size=self.args.batch_size)
            replay_dataloader = DataLoader(replay_dataset, batch_size=None)
            replay_dataloader = self.accelerator.prepare(replay_dataloader)
            return Sampler(distributed=True,
                           dataset=replay_dataset,
                           dataloader=replay_dataloader), None

        # Non-distributed sampler setup
        main_sampler = Sampler(distributed=False,
                               per=self.args.per,
                               memory=self.replay_buffer)
        n_step_sampler = (Sampler(
            distributed=False, n_step=True, memory=self.n_step_buffer)
                          if self.n_step_buffer else None)
        return main_sampler, n_step_sampler

    def store_experience(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
    ) -> None:
        """Process and store experience in replay buffer."""
        if self.n_step_buffer:
            transition = self.n_step_buffer.save_to_memory_vect_envs(
                obs, action, reward, next_obs, done)
            if transition:
                self.replay_buffer.save_to_memory_vect_envs(*transition)
        else:
            self.replay_buffer.save_to_memory(
                obs,
                action,
                reward,
                next_obs,
                done,
                is_vectorised=self.is_vectorised,
            )

    def train_step(self) -> Optional[Dict[str, float]]:
        """Perform a single training step if conditions are met."""
        if (self.replay_buffer.size() <= self.args.warmup_learn_steps
                or self.global_step % self.args.train_frequency != 0):
            return None

        learn_results = []
        for _ in range(self.args.learn_steps):
            experiences = self.data_sampler.sample(self.args.batch_size,
                                                   return_idx=bool(
                                                       self.n_step_buffer))

            if self.n_step_buffer:
                n_step_experiences = self.n_step_sampler.sample(experiences[5])
                experiences += n_step_experiences
                result = self.agent.learn(experiences,
                                          n_step=self.args.n_steps)
            else:
                result = self.agent.learn(experiences)

            learn_results.append(result)

        return calculate_mean(learn_results) if learn_results else None

    def run_train_episode(self) -> Dict[str, float]:
        """Executes a single training episode.

        Returns:
            Dict[str, float]: Metrics for the episode.
        """
        episode_results = []
        obs, info = self.train_env.reset()
        self.train_metrics.reset()
        for _ in range(self.args.rollout_length):
            # Agent action
            action = self.agent.get_action(obs)
            action = action[0] if not self.is_vectorised else action

            # Take action in environment
            next_obs, reward, terminated, truncated, info = self.train_env.step(
                action)

            done = np.logical_or(terminated, truncated)
            # Update metrics
            self.train_metrics.update(reward, terminated, truncated)

            # Save experiences to buffer
            self.store_experience(obs, action, reward, next_obs, done)

            obs = next_obs
            # Training step
            if result := self.train_step():
                episode_results.append(result)

        metrics = self.train_metrics.get_episode_info()
        if episode_results:
            metrics.update(calculate_mean(episode_results))

        return metrics

    def run_evaluate_episodes(self,
                              n_eval_episodes: int = 5) -> Dict[str, float]:
        """Evaluates the agent for a set number of episodes.

        Args:
            max_steps (Optional[int]): Maximum steps per episode.
            n_eval_episodes (int): Number of episodes to evaluate.

        Returns:
            Dict[str, float]: Evaluation results.
        """
        eval_results = []
        for _ in range(n_eval_episodes):
            obs, info = self.test_env.reset()
            self.eval_metrics.reset()
            finished = np.zeros(self.num_test_envs, dtype=bool)
            while not np.all(finished):
                action = self.agent.predict(obs)
                obs, reward, terminated, truncated, info = self.test_env.step(
                    action)

                self.eval_metrics.update(reward, terminated, truncated)
                done = np.logical_or(terminated, truncated)
                for i in range(self.num_test_envs):
                    if done[i] and not finished[i]:
                        finished[i] = True
            metrics = self.eval_metrics.get_episode_info()
            eval_results.append(metrics)
        return calculate_mean(eval_results) if eval_results else None

    def run(self) -> None:
        """Starts the training process."""
        print(f'\nDistributed training on {self.accelerator.device}...')
        if self._is_main_process():
            self.text_logger.info('Start Training')

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                progress_bar = ProgressBar(self.args.max_timesteps)
        else:
            progress_bar = ProgressBar(self.args.max_timesteps)

        while self.global_step < self.args.max_timesteps:
            # Train an episode
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

            train_info = self.run_train_episode()
            env_steps = self.args.rollout_length * self.num_envs
            self.global_step += env_steps
            if self._is_main_process():
                progress_bar.update(env_steps)
            self.episode_cnt += train_info['episode_cnt']

            # Prepare logging information
            train_info.update({
                'num_episode':
                self.episode_cnt,
                'rpm_size':
                self.replay_buffer.size(),
                'eps_greedy':
                getattr(self.agent, 'eps_greedy', 0.0),
                'learning_rate':
                getattr(self.agent, 'learning_rate', 0.0),
                'learner_update_step':
                self.agent.learner_update_step,
                'target_model_update_step':
                self.agent.target_model_update_step,
                'fps':
                int(self.global_step / (time.time() - self.start_time)),
            })

            # Log training information at specified intervals
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()
                if self._is_main_process():
                    if self.global_step % self.args.train_log_interval == 0:
                        self.log_training_info(train_info)
                self.accelerator.wait_for_everyone()

            # Log evaluation information at specified intervals
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()
            if self.global_step % self.args.test_log_interval == 0:
                self.log_evaluation_info(train_info)

        # Save the model if specified
        if self.args.save_model:
            self.agent.save_checkpoint(self.model_save_dir)

    def log_training_info(self, train_info: Dict[str, Any]) -> None:
        """Logs training information."""

        log_message = (f'[Train] Step: {self.global_step}, '
                       f'Episodes: {train_info["num_episode"]}, '
                       f'FPS: {train_info["fps"]}, '
                       f'Episode Reward: {train_info["episode_return"]:.2f}, '
                       f'Episode Length :{train_info["episode_length"]} ')
        self.text_logger.info(log_message)
        self.log_train_infos(train_info, self.global_step)

    def log_evaluation_info(self, train_info) -> None:
        """Logs evaluation information."""
        test_info = self.run_evaluate_episodes(
            n_eval_episodes=self.args.eval_episodes)
        test_info['num_episode'] = self.episode_cnt
        log_message = (f'[Eval] Step: {self.global_step}, '
                       f'Episodes: {train_info["num_episode"]}, '
                       f'Episode Reward: {test_info["episode_return"]:.2f}, '
                       f'Episode Length :{test_info["episode_length"]} ')

        if self._is_main_process():
            self.text_logger.info(log_message)
            self.log_test_infos(test_info, self.global_step)
