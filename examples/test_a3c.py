import os
import sys

sys.path.append(os.getcwd())
from scalerl.algos.a3c.a3c_agent import A3CAgent, A3CArguments

if __name__ == '__main__':
    args = A3CArguments()
    agent = A3CAgent(args)
    agent.run()
