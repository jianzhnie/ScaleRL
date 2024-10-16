import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

sys.path.append(os.getcwd())
from scalerl.algos.async_qlearning.async_dqn import AsyncDQN

if __name__ == '__main__':
    impala_dqn = AsyncDQN(state_dim=4, action_dim=2)
    impala_dqn.run()
