import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

sys.path.append(os.getcwd())
from scalerl.algorithms.dqn.parallel_dqn import ParallelDQN


if __name__ == "__main__":
    parallel_dqn = ParallelDQN(
        env_name="CartPole-v0", num_actors=4, max_episode_size=1000
    )
    parallel_dqn.run()
