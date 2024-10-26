import os
import sys

sys.path.append(os.getcwd())
from scalerl.algorithms.a3c.parallel_a3c import ParallelA3C

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    a3c = ParallelA3C(env_name="CartPole-v0")
    a3c.run()
