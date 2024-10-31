import gymnasium as gym
import numpy as np


def test():
    env = gym.make('CartPole-v1')
    observation, info = env.reset()

    episode_over = False
    episode_step = 0
    while not episode_over:
        action = (env.action_space.sample()
                  )  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        episode_step += 1
        episode_over = terminated or truncated
        print(episode_step, reward, episode_over)
    env.close()
    print(episode_step)


def test2():
    env = gym.make('CartPole-v1')
    observation, info = env.reset()
    episode_step = 0
    rollout_length = 200
    data = []
    for _ in range(rollout_length):
        action = (env.action_space.sample()
                  )  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        episode_step += 1
        episode_over = terminated or truncated
        print(episode_step, reward, episode_over)
        data.append(episode_over)

    env.close()
    print(episode_step)


def test_vec_env():
    num_envs = 4
    envs = gym.make_vec('CartPole-v1',
                        num_envs=num_envs,
                        vectorization_mode='async')
    observations, infos = envs.reset(seed=42)
    all_done = np.zeros(num_envs)
    episode_step = 0
    while not all_done.all():
        observations, rewards, terminations, truncations, infos = envs.step(
            envs.action_space.sample())
        all_done = np.logical_or(all_done, terminations)
        episode_step += 1
        print(episode_step, rewards, terminations, truncations)
    envs.close()
    print(episode_step)


def test_vec_env2():
    num_envs = 4
    envs = gym.vector.AsyncVectorEnv(
        [lambda: gym.make('CartPole-v1') for _ in range(num_envs)])

    # 初始化每个环境的统计数据
    observations, infos = envs.reset()
    episode_returns = np.zeros(num_envs)  # 记录每个环境的累积奖励（return）
    episode_lengths = np.zeros(num_envs)  # 记录每个环境的步数
    completed_returns = []  # 存储每局完成的return
    completed_lengths = []  # 存储每局完成的步数
    rollout_length = 100

    for step_idx in range(rollout_length):
        actions = envs.action_space.sample()  # 为每个环境生成随机动作
        observations, rewards, terminations, truncations, infos = envs.step(
            actions)

        # 累积奖励和步数
        episode_returns += rewards
        episode_lengths += 1

        # 检查每个环境是否终止
        for i in range(num_envs):
            if terminations[i] or truncations[i]:  # 判断一局是否结束
                completed_returns.append(episode_returns[i])  # 记录完成局的 return
                completed_lengths.append(episode_lengths[i])  # 记录完成局的步数

                # 重置统计变量
                episode_returns[i] = 0
                episode_lengths[i] = 0
        print(rewards, terminations, truncations, episode_lengths)

    # 输出完成的 episode 统计数据
    print('Completed Episode Returns:', completed_returns)
    print('Completed Episode Lengths:', completed_lengths)
    print('Total Episode Lengths:', np.sum(completed_lengths))


if __name__ == '__main__':
    test()
    test2()
    test_vec_env()
    print(
        'Running vectorized environment test with return and length tracking...'
    )
    test_vec_env2()
