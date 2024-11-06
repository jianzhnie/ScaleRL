import os
import sys

import gymnasium as gym
import torch
import tyro

sys.path.append(os.getcwd())
from accelerate import Accelerator

from scalerl.algorithms.dqn.dqn_agent import DQNAgent
from scalerl.algorithms.rl_args import DQNArguments
from scalerl.envs.env_utils import make_vect_envs
from scalerl.trainer.off_policy import OffPolicyTrainer
from scalerl.utils import get_device

if __name__ == '__main__':
    accelerator = Accelerator()
    args: DQNArguments = tyro.cli(DQNArguments)
    train_env: gym.Env = make_vect_envs(args.env_id, num_envs=args.num_envs)
    test_env: gym.Env = make_vect_envs(args.env_id, num_envs=args.num_envs)

    state_shape = (train_env.single_observation_space.shape
                   or train_env.single_observation_space.n)
    action_shape = (train_env.single_action_space.shape
                    or train_env.single_action_space.n)

    args.action_bound = (train_env.action_space.high[0] if isinstance(
        train_env.action_space, gym.spaces.Box) else None)

    if accelerator is None:
        device = get_device(args.device)
    else:
        device = accelerator.device

    args.num_processes = torch.cuda.device_count()

    if accelerator is None or accelerator.is_main_process:
        print('---------------------------------------')
        print('Environment:', args.env_id)
        print('Algorithm:', args.algo_name)
        print('State Shape:', state_shape)
        print('Action Shape:', action_shape)
        print('Action Bound:', args.action_bound)
        print('Num Process:', args.num_processes)
        print('Device:', device)
        print('---------------------------------------')
        print(args)

    # agent
    agent = DQNAgent(
        args=args,
        state_shape=state_shape,
        action_shape=action_shape,
        accelerator=accelerator,
        device=device,
    )
    runner = OffPolicyTrainer(
        args,
        train_env=train_env,
        test_env=test_env,
        agent=agent,
        accelerator=accelerator,
        device=device,
    )
    runner.run()
