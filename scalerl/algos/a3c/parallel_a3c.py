from __future__ import print_function

import os
import queue
import sys
from typing import List

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())
from scalerl.algos.a3c.share_optim import SharedAdam
from scalerl.algos.rl_args import A3CArguments
from scalerl.envs.gym_env import make_gym_env


class ActorCriticNet(nn.Module):

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int) -> None:
        """初始化Actor-Critic网络。

        Args:
            obs_dim (int): 观察空间的维度。
            hidden_dim (int): 隐藏层维度。
            action_dim (int): 动作空间的维度。
        """
        super(ActorCriticNet, self).__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_linear = nn.Linear(hidden_dim, action_dim)
        self.critic_linear = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回动作的logits和状态值。

        Args:
            obs (torch.Tensor): 观察值。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 动作的logits和状态值。
        """
        feature = self.feature_net(obs)
        value = self.critic_linear(feature)
        logits = self.actor_linear(feature)
        return logits, value


class A3CTrainer:

    def __init__(self, args: A3CArguments) -> None:
        """Initialize the A3CAgent with shared model, environment and
        optimizer."""
        self.args = args
        self.gamma = args.gamma

        # Initialize environment
        self.env = make_gym_env(self.args.env_name)
        self.train_env = make_gym_env(env_id=self.args.env_name)
        self.test_env = make_gym_env(env_id=self.args.env_name)

        # Observation and action dimensions
        obs_shape = self.env.observation_space.shape or self.env.observation_space.n
        action_shape = self.env.action_space.shape or self.env.action_space.n
        self.obs_dim = int(np.prod(obs_shape))
        self.action_dim = int(np.prod(action_shape))

        self.local_model = ActorCriticNet(
            obs_dim=self.obs_dim,
            hidden_dim=self.args.hidden_dim,
            action_dim=self.action_dim,
        )

        # Initialize shared model
        self.shared_model = ActorCriticNet(
            obs_dim=self.obs_dim,
            hidden_dim=self.args.hidden_dim,
            action_dim=self.action_dim,
        )
        self.shared_model.share_memory()
        # Ensure model is shared across processes

        # Initialize optimizer
        if not self.args.no_shared:
            self.optimizer = SharedAdam(self.shared_model.parameters(),
                                        lr=self.args.lr)
            self.optimizer.share_memory()
        else:
            self.optimizer = None

    def get_action(self, obs: torch.Tensor) -> int:
        """选择动作并返回整型动作。

        Args:
            obs (torch.Tensor): 观察值。

        Returns:
            int: 选择的动作。
        """
        if obs.ndim == 1:
            # Expand to have batch_size = 1
            obs = np.expand_dims(obs, axis=0)

        obs = torch.tensor(obs, dtype=torch.float)
        with torch.no_grad():
            logits, _ = self.local_model(obs)
        probs = F.softmax(logits, dim=-1)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()  # 返回整数动作

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
        obs = torch.tensor(obs, dtype=torch.float)
        with torch.no_grad():
            logits, _ = self.local_model(obs)
        prob = F.softmax(logits, dim=-1)
        action = prob.max(1, keepdim=True)[1]
        return action.item()

    def _sync_with_shared_model(self, local_model: nn.Module,
                                shared_model: nn.Module):
        """Load the shared model parameters into the local model."""
        local_model.load_state_dict(shared_model.state_dict())

    def ensure_shared_grads(self, local_model: nn.Module,
                            shared_model: nn.Module) -> None:
        """Ensure that gradients from the local model are copied to the shared
        model."""
        for param, shared_param in zip(local_model.parameters(),
                                       shared_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad.clone()

    def compute_loss(self, transition_dict: dict) -> torch.Tensor:
        """计算Actor和Critic的损失。

        Args:
            transition_dict (dict): 包含观察、动作、奖励、下一观察和完成标志的字典。

        Returns:
            torch.Tensor: 总损失。
        """
        # 先将 numpy 列表转换为 numpy 数组，再转换为 tensor
        obs = torch.tensor(np.array(transition_dict['obs']),
                           dtype=torch.float32)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.long).view(-1, 1)
        rewards = torch.tensor(np.array(transition_dict['rewards']),
                               dtype=torch.float32).view(-1, 1)
        next_obs = torch.tensor(np.array(transition_dict['next_obs']),
                                dtype=torch.float32)
        dones = torch.tensor(np.array(transition_dict['dones']),
                             dtype=torch.float32).view(-1, 1)

        curr_policy, curr_value = self.local_model(obs)
        _, next_value = self.local_model(next_obs)
        curr_probs = F.softmax(curr_policy, dim=-1)

        td_target = rewards + self.gamma * next_value * (1 - dones)
        td_error = td_target - curr_value

        log_probs = torch.log(curr_probs).gather(1, actions)
        actor_loss = torch.mean(-log_probs * td_error.detach())

        critic_loss = F.mse_loss(td_target.detach(), curr_value)
        total_loss = actor_loss + critic_loss
        return total_loss

    def train(
        self,
        worker_id: int,
        optimizer: torch.optim.Optimizer = None,
        global_ep_counter: mp.Value = None,
        global_reward_queue: mp.Queue = None,
    ) -> None:
        seed = self.args.seed + worker_id
        torch.manual_seed(seed)
        self.local_model.train()
        if optimizer is None:
            optimizer = torch.optim.Adam(self.shared_model.parameters(),
                                         lr=self.args.lr)
        self._sync_with_shared_model(self.local_model, self.shared_model)
        while global_ep_counter.value < self.args.max_episode_size:
            transition_dict = {
                'obs': [],
                'actions': [],
                'next_obs': [],
                'rewards': [],
                'dones': [],
            }
            episode_reward = 0
            obs, _ = self.train_env.reset(seed=seed)
            done = False
            while not done:
                action = self.get_action(obs)
                next_obs, reward, terminal, truncated, _ = self.train_env.step(
                    action)
                done = terminal or truncated
                transition_dict['obs'].append(obs)
                transition_dict['actions'].append(action)
                transition_dict['next_obs'].append(next_obs)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                obs = next_obs
                episode_reward += reward

            total_loss = self.compute_loss(transition_dict)

            print('Total loss', total_loss.item(), 'episode reward',
                  episode_reward)
            # 反向传播和优化
            self.optimizer.zero_grad()
            total_loss.backward()

            self.ensure_shared_grads(self.local_model, self.shared_model)
            self.optimizer.step()

            with global_ep_counter.get_lock():
                global_ep_counter.value += 1
                global_reward_queue.put(episode_reward)

    def test(self, worker_id: int) -> None:
        """Test worker function to evaluate the performance of the agent."""
        self.local_model.eval()
        seed = self.args.seed + worker_id
        torch.manual_seed(seed=seed)
        self._sync_with_shared_model(self.local_model, self.shared_model)
        obs, info = self.test_env.reset(seed=seed)
        done = False
        episode_reward = 0.0
        episode_length = 0

        while not done:
            action = self.predict(obs)
            next_obs, reward, terminal, truncated, _ = self.test_env.step(
                action)
            done = terminal or truncated
            obs = next_obs
            if info and 'episode' in info:
                info_item = {k: v.item() for k, v in info['episode'].items()}
                episode_reward = info_item['r']
                episode_length = info_item['l']

        print(
            f'Episode reward {episode_reward}, Episode length {episode_length}'
        )
        return episode_reward, episode_length

    def run(self) -> None:
        """Run the agent, spawning processes for training and testing."""
        processes: List[mp.Process] = []

        # Create shared counter for processes
        global_ep_counter = mp.Value('i', 0)
        global_reward_queue = queue.Queue()

        # Start the testing process
        # test_process = mp.Process(
        #     target=self.test,
        #     args=(self.args.num_processes, global_ep_counter),
        # )
        # test_process.start()
        # processes.append(test_process)

        # Start training processes
        for rank in range(self.args.num_processes):
            p = mp.Process(
                target=self.train,
                args=(
                    rank,
                    self.optimizer,
                    global_ep_counter,
                    global_reward_queue,
                ),
            )

            p.start()
            processes.append(p)

        for p in processes:
            p.join()


if __name__ == '__main__':
    args = A3CArguments()
    a3c = A3CTrainer(args)
    a3c.run()
