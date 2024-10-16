import os
import sys

sys.path.append(os.getcwd())
from scalerl.algos.impala.impala_atari import ImpalaTrainer
from scalerl.algos.rl_args import parse_args

if __name__ == '__main__':
    flags = parse_args()
    print(flags)
    agent = ImpalaTrainer(flags)
    agent.train()
