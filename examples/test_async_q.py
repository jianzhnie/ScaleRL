import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

sys.path.append(os.getcwd())
from scalerl.algos.async_dqn.parallel_dqn import AsyncDQN

if __name__ == '__main__':
    impala_dqn = AsyncDQN(env_name='CartPole-v0',
                          num_actors=4,
                          max_episode_size=1000)
    impala_dqn.run()
