import time
from typing import Optional, Union

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
from scalerl.utils import ProgressBar, calculate_mean, get_device


class OffPolicyTrainer(BaseTrainer):

    def __init__(
        self,
        args: RLArguments,
        train_env: gym.Env,
        test_env: gym.Env,
        agent: BaseAgent,
        accelerator: Optional[Accelerator] = None,
        device: Optional[Union[str, torch.device]] = 'auto',
    ) -> None:
        super().__init__(args, train_env, test_env, agent)

        # Detect if environment is vectorised
        if hasattr(train_env, 'num_envs'):
            self.num_envs = train_env.num_envs
            self.is_vectorised = True
        else:
            self.num_envs = 1
            self.is_vectorised = False

        # Training
        self.episode_cnt = 0
        self.global_step = 0
        self.start_time = time.time()
        self.eps_greedy = 0.0
        self.device = get_device(device)

        field_names = ['obs', 'action', 'reward', 'next_obs', 'done']

        if self.args.per:
            self.replay_buffer = PrioritizedReplayBuffer(
                memory_size=args.buffer_size,
                field_names=field_names,
                num_envs=self.num_envs,
                alpha=args.alpha,
                gamma=args.gamma,
                device=self.device,
            )
            if self.args.n_steps:
                self.n_step_buffer = MultiStepReplayBuffer(
                    memory_size=args.buffer_size,
                    field_names=field_names,
                    num_envs=self.num_envs,
                    gamma=args.gamma,
                    device=self.device,
                )
        elif self.args.n_steps:
            self.replay_buffer = ReplayBuffer(
                memory_size=self.args.buffer_size,
                field_names=field_names,
                device=self.device,
            )
            self.n_step_buffer = MultiStepReplayBuffer(
                memory_size=args.buffer_size,
                field_names=field_names,
                num_envs=self.num_envs,
                gamma=args.gamma,
                device=self.device,
            )
        else:
            self.replay_buffer = ReplayBuffer(
                memory_size=self.args.buffer_size,
                field_names=field_names,
                device=self.device,
            )

        if accelerator is not None:
            # Create dataloader from replay buffer
            replay_dataset = ReplayDataset(self.replay_buffer,
                                           batch_size=args.batch_size)
            replay_dataloader = DataLoader(replay_dataset, batch_size=None)
            replay_dataloader = accelerator.prepare(replay_dataloader)
            self.data_sampler = Sampler(
                distributed=True,
                dataset=replay_dataset,
                dataloader=replay_dataloader,
            )
        else:
            self.sampler = Sampler(distributed=False,
                                   per=self.args.per,
                                   memory=self.replay_buffer)
            if self.n_step_buffer is not None:
                self.n_step_sampler = Sampler(distributed=False,
                                              n_step=True,
                                              memory=self.n_step_buffer)

    # train an episode
    def run_train_episode(self) -> dict[str, float]:
        episode_result_info = []
        # Reset environment at start of episode
        obs, info = self.train_env.reset()
        scores = np.zeros(self.num_envs)
        completed_episode_scores = []
        for step_idx in range(self.args.rollout_length):
            action = self.agent.get_action(obs)
            if not self.is_vectorised:
                action = action[0]
            # Act in Environment
            next_obs, reward, terminated, truncated, info = self.train_env.step(
                action)
            scores += np.array(reward)
            if not self.is_vectorised:
                terminated = [terminated]
                truncated = [truncated]

            done = np.logical_and(terminated, terminated)
            # Collect scores for completed episodes
            for idx, (term, trunc) in enumerate(zip(terminated, truncated)):
                if term or trunc:
                    completed_episode_scores.append(scores[idx])
                    scores[idx] = 0
            self.global_step += self.num_envs

            if self.n_step_buffer is not None:
                one_step_transition = self.n_step_buffer.save_to_memory_vect_envs(
                    obs,
                    action,
                    reward,
                    next_obs,
                    done,
                )
                if one_step_transition:
                    self.replay_buffer.save_to_memory_vect_envs(
                        *one_step_transition)
            else:
                # Save experience to replay buffer
                self.replay_buffer.save_to_memory(
                    obs,
                    action,
                    reward,
                    next_obs,
                    terminated,
                    is_vectorised=self.is_vectorised,
                )
            # Learn according to learning frequency
            if self.replay_buffer.size() > self.args.warmup_learn_steps:
                if self.global_step % self.args.train_frequency == 0:
                    for _ in range(self.num_envs // self.args.learn_steps):
                        # Sample dataloader
                        experiences = self.data_sampler.sample(
                            self.args.batch_size,
                            return_idx=True
                            if self.n_step_buffer is not None else False,
                        )
                        if self.n_step_buffer is not None:
                            n_step_experiences = self.n_step_sampler.sample(
                                experiences[5])
                            experiences += n_step_experiences
                            learn_result = self.agent.learn(
                                experiences, n_step=self.args.n_step)
                        else:
                            # Learn according to agent's RL algorithm
                            learn_result = self.agent.learn(experiences)
                        episode_result_info.append(learn_result)

            obs = next_obs

        mean_scores = (np.mean(completed_episode_scores)
                       if len(completed_episode_scores) > 0 else
                       '0 completed episodes')

        episode_info = calculate_mean(episode_result_info)
        train_info = {
            'episode_reward': mean_scores,
            'episode_steps': self.args.rollout_length,
        }
        train_info.update(episode_info)
        return train_info

    def run_evaluate_episodes(self,
                              max_steps: Optional[int] = None,
                              n_eval_episodes: int = 5) -> dict[str, float]:
        eval_rewards = []
        num_envs = self.test_env.num_envs if hasattr(self.test_env,
                                                     'num_envs') else 1
        for _ in range(n_eval_episodes):
            obs, info = self.test_env.reset()
            scores = np.zeros(num_envs)
            completed_episode_scores = np.zeros(num_envs)
            finished = np.zeros(num_envs)
            step = 0
            while not np.all(finished):
                action_mask = info.get('action_mask', None)
                action = self.agent.predict(obs, action_mask)
                next_obs, reward, terminated, truncated, info = self.test_env.step(
                    action)
                step += 1
                obs = next_obs
                scores += np.array(reward)
                for idx, (term, trunc) in enumerate(zip(terminated,
                                                        truncated)):
                    if (term or trunc or
                        (max_steps is not None
                         and step == max_steps)) and not finished[idx]:
                        completed_episode_scores[idx] = scores[idx]
                        finished[idx] = 1
            eval_rewards.append(np.mean(completed_episode_scores))

        return {
            'reward_mean': np.mean(eval_rewards),
            'reward_std': np.std(eval_rewards),
        }

    def run(self) -> None:
        """Train the agent."""
        self.text_logger.info('Start Training')
        progress_bar = ProgressBar(self.args.max_timesteps)
        while self.global_step < self.args.max_timesteps:
            # Training logic
            train_info = self.run_train_episode()
            episode_step = train_info['episode_step']
            progress_bar.update(episode_step * self.num_envs)

            train_info['rpm_size'] = self.replay_buffer.size()
            train_info['eps_greedy'] = (self.agent.eps_greedy if hasattr(
                self.agent, 'eps_greedy') else 0.0)
            train_info['learning_rate'] = (self.agent.learning_rate if hasattr(
                self.agent, 'learning_rate') else 0.0)
            train_info['learner_update_step'] = self.agent.learner_update_step
            train_info[
                'target_model_update_step'] = self.agent.target_model_update_step

            # Log training information
            train_fps = int(self.global_step / (time.time() - self.start_time))
            train_info['fps'] = train_fps

            # Log training information
            if self.episode_cnt % self.args.train_log_interval == 0:
                log_message = (
                    '[Train], global_step: {}, episodes: {}, train_fps: {}, '
                    'episode_reward: {:.2f}, episode_step: {:.2f}').format(
                        self.global_step,
                        self.episode_cnt,
                        train_fps,
                        train_info['episode_reward'],
                        train_info['episode_step'],
                    )
                self.text_logger.info(log_message)
                self.log_train_infos(train_info, self.global_step)

            # Log testing information
            if self.episode_cnt % self.args.test_log_interval == 0:
                test_info = self.run_evaluate_episodes(
                    n_eval_episodes=self.args.eval_episodes)
                test_info['num_episode'] = self.episode_cnt
                log_message = (
                    '[Eval], global_step: {}, episode: {}, eval_rewards: {:.2f}'
                    .format(self.global_step, self.episode_cnt,
                            test_info['reward_mean']))
                self.text_logger.info(log_message)
                self.log_test_infos(test_info, self.global_step)

        # Save model
        if self.args.save_model:
            self.agent.save_checkpoint(self.model_save_dir)
