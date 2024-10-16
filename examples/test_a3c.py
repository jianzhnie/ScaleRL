import os
import sys

sys.path.append(os.getcwd())
from scalerl.algos.a3c.parallel_a3c import A3CArguments, A3CTrainer

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    args = A3CArguments()
    print(args)
    agent = A3CTrainer(args)
    agent.run()
