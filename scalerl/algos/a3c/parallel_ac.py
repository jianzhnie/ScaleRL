import queue

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo


def make_gym_env(
    env_id: str,
    seed: int = 42,
    capture_video: bool = False,
    save_video_dir: str = 'work_dir',
    save_video_name: str = 'test',
) -> RecordEpisodeStatistics:
    """创建并封装Gym环境，包含必要的封装器。

    Args:
        env_id (str): 环境ID。
        seed (int): 随机种子。
        capture_video (bool): 是否录制视频。
        save_video_dir (str): 保存视频的目录。
        save_video_name (str): 保存视频的文件名。

    Returns:
        RecordEpisodeStatistics: 封装后的环境。
    """
    if capture_video:
        env = gym.make(env_id, render_mode='rgb_array')
        env = RecordVideo(env, f'{save_video_dir}/{save_video_name}')
    else:
        env = gym.make(env_id)
    env = RecordEpisodeStatistics(env)
    env.reset(seed=seed)  # 正确的设置随机种子方式
    return env


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

    def get_action_value(
            self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


class Worker(mp.Process):

    def __init__(
        self,
        share_model: ActorCriticNet,
        optimizer: optim.Optimizer,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
        env_name: str,
        global_ep_counter: mp.Value,
        global_reward_queue: queue.Queue,
        max_episode_size: int = 10000,
        gamma: float = 0.99,
    ) -> None:
        """初始化Worker进程。

        Args:
            share_model (ActorCriticNet): 共享的Actor-Critic网络模型。
            optimizer (optim.Optimizer): 优化器。
            obs_dim (int): 观察空间的维度。
            hidden_dim (int): 隐藏层维度。
            action_dim (int): 动作空间的维度。
            env_name (str): 环境名称。
            global_ep_counter (mp.Value): 全局回合计数器。
            global_reward_queue (queue.Queue): 全局奖励队列。
            max_episode_size (int): 最大回合数。
            gamma (float): 折扣因子。
        """
        super(Worker, self).__init__()
        self.local_model = ActorCriticNet(obs_dim, hidden_dim, action_dim)
        self.share_model = share_model
        self.optimizer = optimizer
        self.env = make_gym_env(env_name)
        self.global_ep_counter = global_ep_counter
        self.global_reward_queue = global_reward_queue
        self.max_episode_size = max_episode_size
        self.gamma = gamma

    def get_action(self, obs: np.ndarray) -> int:
        """选择动作并返回整型动作。

        Args:
            obs (np.ndarray): 观察值。

        Returns:
            int: 选择的动作。
        """
        if obs.ndim == 1:
            obs = np.expand_dims(obs, axis=0)
        obs = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            logits, _ = self.local_model.get_action_value(obs)
        probs = F.softmax(logits, dim=-1)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()  # 返回整数动作

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

        curr_policy, curr_value = self.local_model.get_action_value(obs)
        _, next_value = self.local_model.get_action_value(next_obs)
        curr_probs = F.softmax(curr_policy, dim=-1)

        td_target = rewards + self.gamma * next_value * (1 - dones)
        td_error = td_target - curr_value

        log_probs = torch.log(curr_probs).gather(1, actions)
        actor_loss = torch.mean(-log_probs * td_error.detach())

        critic_loss = F.mse_loss(td_target.detach(), curr_value)
        total_loss = actor_loss + critic_loss
        return total_loss

    def run(self) -> None:
        """Worker进程主循环。"""
        while self.global_ep_counter.value < self.max_episode_size:
            transition_dict = {
                'obs': [],
                'actions': [],
                'next_obs': [],
                'rewards': [],
                'dones': [],
            }
            episode_reward = 0
            obs, _ = self.env.reset()
            done = False
            while not done:
                action = self.get_action(obs)
                next_obs, reward, terminal, truncated, _ = self.env.step(
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

            # 全局模型参数更新
            for local_param, global_param in zip(
                    self.local_model.parameters(),
                    self.share_model.parameters(),
            ):
                global_param.grad = local_param.grad

            self.optimizer.step()

            # 同步本地网络与全局网络
            self.local_model.load_state_dict(self.share_model.state_dict())
            with self.global_ep_counter.get_lock():
                self.global_ep_counter.value += 1
                self.global_reward_queue.put(episode_reward)


if __name__ == '__main__':
    env_name = 'CartPole-v1'
    env = make_gym_env(env_name)

    hidden_dim = 8
    obs_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    obs_dim = int(np.prod(obs_shape))
    action_dim = int(np.prod(action_shape))

    share_model = ActorCriticNet(obs_dim, hidden_dim, action_dim)
    share_model.share_memory()
    optimizer = optim.Adam(share_model.parameters(), lr=0.001)

    global_ep_counter = mp.Value('i', 0)
    global_reward_queue = queue.Queue()

    num_workers = 4
    workers = [
        Worker(
            share_model,
            optimizer,
            obs_dim,
            hidden_dim,
            action_dim,
            env_name,
            global_ep_counter,
            global_reward_queue,
            max_episode_size=10000,
            gamma=0.99,
        ) for _ in range(num_workers)
    ]

    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()
