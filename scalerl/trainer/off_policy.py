import time
from typing import Any, Dict, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from scalerl.algorithms.base import BaseAgent
from scalerl.algorithms.rl_args import RLArguments
from scalerl.data.replay_buffer import (MultiStepReplayBuffer,
                                        PrioritizedReplayBuffer, ReplayBuffer)
from scalerl.data.replay_data import ReplayDataset
from scalerl.data.sampler import Sampler
from scalerl.trainer.base import BaseTrainer
from scalerl.utils import ProgressBar, calculate_mean


class OffPolicyTrainer(BaseTrainer):
    """Trainer class for off-policy reinforcement learning algorithms.

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
        super().__init__(args, train_env, test_env, agent)

        # Environment configurations
        self.num_envs = getattr(train_env, 'num_envs', 1)
        self.is_vectorised = hasattr(train_env, 'num_envs')

        # Training state
        self.episode_cnt = 0
        self.global_step = 0
        self.start_time = time.time()
        self.device = device

        # Replay buffer setup
        field_names = ['obs', 'action', 'reward', 'next_obs', 'done']
        self._initialize_replay_buffers(field_names)

        # Initialize sampling setup based on the use of accelerator
        self.data_sampler, self.n_step_sampler = self._initialize_samplers(
            accelerator)

    def _initialize_replay_buffers(self, field_names: list) -> None:
        """Initializes the appropriate replay buffer(s) based on training
        arguments."""
        self.replay_buffer = None
        self.n_step_buffer = None

        if self.args.per:
            # Prioritized Experience Replay
            self.replay_buffer = PrioritizedReplayBuffer(
                memory_size=self.args.buffer_size,
                field_names=field_names,
                num_envs=self.num_envs,
                alpha=self.args.alpha,
                gamma=self.args.gamma,
                device=self.device,
            )
            if self.args.n_steps:
                # Multi-step buffer for n-step learning
                self.n_step_buffer = MultiStepReplayBuffer(
                    memory_size=self.args.buffer_size,
                    field_names=field_names,
                    num_envs=self.num_envs,
                    gamma=self.args.gamma,
                    device=self.device,
                )
        elif self.args.n_steps:
            # Regular Replay with n-step buffer
            self.replay_buffer = ReplayBuffer(
                memory_size=self.args.buffer_size,
                field_names=field_names,
                device=self.device,
            )
            self.n_step_buffer = MultiStepReplayBuffer(
                memory_size=self.args.buffer_size,
                field_names=field_names,
                num_envs=self.num_envs,
                gamma=self.args.gamma,
                device=self.device,
            )
        else:
            # Regular Replay Buffer without n-steps
            self.replay_buffer = ReplayBuffer(
                memory_size=self.args.buffer_size,
                field_names=field_names,
                device=self.device,
            )

    def _initialize_samplers(
        self, accelerator: Optional[Accelerator]
    ) -> tuple[Sampler, Optional[Sampler]]:
        """Initializes samplers for data collection and training.

        Args:
            accelerator (Optional[Accelerator]): Accelerator for distributed training.

        Returns:
            tuple[Sampler, Optional[Sampler]]: Main sampler and optional n-step sampler.
        """
        if accelerator:
            # Distributed sampler with accelerator support
            replay_dataset = ReplayDataset(self.replay_buffer,
                                           batch_size=self.args.batch_size)
            replay_dataloader = DataLoader(replay_dataset, batch_size=None)
            replay_dataloader = accelerator.prepare(replay_dataloader)
            return Sampler(distributed=True,
                           dataset=replay_dataset,
                           dataloader=replay_dataloader), None
        else:
            # Non-distributed sampler setup
            main_sampler = Sampler(distributed=False,
                                   per=self.args.per,
                                   memory=self.replay_buffer)
            n_step_sampler = (Sampler(
                distributed=False, n_step=True, memory=self.n_step_buffer)
                              if self.n_step_buffer else None)
            return main_sampler, n_step_sampler

    def run_train_episode(self) -> Dict[str, float]:
        """Executes a single training episode.

        Returns:
            Dict[str, float]: Metrics for the episode.
        """
        obs, info = self.train_env.reset()
        scores = np.zeros(self.num_envs)
        completed_episode_scores = []
        episode_result_info = []

        for _ in range(self.args.rollout_length):
            # Agent action
            action = self.agent.get_action(obs)
            action = action[0] if not self.is_vectorised else action

            try:
                # Take action in environment
                next_obs, reward, terminated, truncated, info = self.train_env.step(
                    action)
            except Exception as e:
                self.text_logger.error(f'Error during environment step: {e}')
                break  # Exit on error

            scores += np.array(reward)
            done = np.logical_or(terminated, truncated)

            # Score tracking for completed episodes
            for idx, (term, trunc) in enumerate(zip(terminated, truncated)):
                if term or trunc:
                    completed_episode_scores.append(scores[idx])
                    scores[idx] = 0
            self.global_step += self.num_envs

            # Save experiences to buffer
            if self.n_step_buffer:
                one_step_transition = self.n_step_buffer.save_to_memory_vect_envs(
                    obs, action, reward, next_obs, done)
                if one_step_transition:
                    self.replay_buffer.save_to_memory_vect_envs(
                        *one_step_transition)
            else:
                self.replay_buffer.save_to_memory(
                    obs,
                    action,
                    reward,
                    next_obs,
                    done,
                    is_vectorised=self.is_vectorised,
                )

            # Training step
            if (self.replay_buffer.size() > self.args.warmup_learn_steps
                    and self.global_step % self.args.train_frequency == 0):
                for _ in range(self.num_envs // self.args.learn_steps):
                    experiences = self.data_sampler.sample(
                        self.args.batch_size,
                        return_idx=bool(self.n_step_buffer))
                    if self.n_step_buffer:
                        n_step_experiences = self.n_step_sampler.sample(
                            experiences[5])
                        experiences += n_step_experiences
                        learn_result = self.agent.learn(
                            experiences, n_step=self.args.n_step)
                    else:
                        learn_result = self.agent.learn(experiences)
                    episode_result_info.append(learn_result)

            obs = next_obs

        mean_scores = (np.mean(completed_episode_scores)
                       if completed_episode_scores else 0.0)
        episode_info = calculate_mean(episode_result_info)
        return {
            'episode_reward': mean_scores,
            'episode_step': self.args.rollout_length,
            **episode_info,
        }

    def run_evaluate_episodes(self,
                              n_eval_episodes: int = 5) -> Dict[str, float]:
        """Evaluates the agent for a set number of episodes.

        Args:
            max_steps (Optional[int]): Maximum steps per episode.
            n_eval_episodes (int): Number of episodes to evaluate.

        Returns:
            Dict[str, float]: Evaluation results.
        """
        eval_rewards = []
        num_envs = getattr(self.test_env, 'num_envs', 1)

        for _ in range(n_eval_episodes):
            obs, info = self.test_env.reset()
            scores = np.zeros(num_envs)
            completed_episode_scores = np.zeros(num_envs)
            finished = np.zeros(num_envs)

            while not all(finished):
                action = self.agent.predict(obs)
                next_obs, reward, terminated, truncated, info = self.test_env.step(
                    action)
                scores += np.array(reward)

                # End episode when done or max steps reached
                for idx, (term, trunc) in enumerate(zip(terminated,
                                                        truncated)):
                    if (term or trunc) and not finished[idx]:
                        completed_episode_scores[idx] = scores[idx]
                        finished[idx] = 1
                if np.all(finished):
                    break

                obs = next_obs

            eval_rewards.append(np.mean(completed_episode_scores))

        return {
            'reward_mean': np.mean(eval_rewards),
            'reward_std': np.std(eval_rewards),
        }

    def run(self) -> None:
        """Starts the training process."""
        self.text_logger.info('Start Training')
        progress_bar = ProgressBar(self.args.max_timesteps)

        while self.global_step < self.args.max_timesteps:
            # Train an episode
            train_info = self.run_train_episode()
            episode_step = train_info.get('episode_step', 0)
            progress_bar.update(episode_step * self.num_envs)
            self.episode_cnt += self.num_envs

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
            if self.episode_cnt % self.args.train_log_interval == 0:
                self._log_training_info(train_info)

            # Log evaluation information at specified intervals
            if self.episode_cnt % self.args.test_log_interval == 0:
                self._log_evaluation_info()

        # Save the model if specified
        if self.args.save_model:
            self.agent.save_checkpoint(self.model_save_dir)

    def _log_training_info(self, train_info: Dict[str, Any]) -> None:
        """Logs training information."""
        log_message = (
            '[Train], global_step: {}, episodes: {}, train_fps: {}, '
            'episode_reward: {:.2f}, episode_steps: {:.2f}'.format(
                self.global_step,
                self.episode_cnt,
                train_info['fps'],
                train_info['episode_reward'],
                train_info['episode_step'],
            ))
        self.text_logger.info(log_message)
        self.log_train_infos(train_info, self.global_step)

    def _log_evaluation_info(self) -> None:
        """Logs evaluation information."""
        test_info = self.run_evaluate_episodes(
            n_eval_episodes=self.args.eval_episodes)
        test_info['num_episode'] = self.episode_cnt
        log_message = (
            '[Eval], global_step: {}, episode: {}, eval_rewards: {:.2f}'.
            format(self.global_step, self.episode_cnt,
                   test_info['reward_mean']))
        self.text_logger.info(log_message)
        self.log_test_infos(test_info, self.global_step)
