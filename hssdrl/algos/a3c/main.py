from __future__ import print_function
import numpy as np
import argparse
import os
from typing import List, Optional, Tuple
import torch
import torch.multiprocessing as mp
from hssdrl.algos.a3c.worker import train, test, ActorCritic
from hssdrl.algos.a3c.share_optim import SharedAdam
from hssdrl.envs.gym_env import make_gym_env


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the A3C training."""
    parser = argparse.ArgumentParser(description="A3C")

    # Hyperparameters and configurations
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="Learning rate (default: 0.0001)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for rewards (default: 0.99)",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=1.00,
        help="Lambda parameter for GAE (default: 1.00)",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="Entropy term coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--value-loss-coef",
        type=float,
        default=0.5,
        help="Value loss coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=50,
        help="Maximum gradient norm (default: 50)",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=4,
        help="Number of training processes to use (default: 4)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=20,
        help="Number of forward steps in A3C (default: 20)",
    )
    parser.add_argument(
        "--max-episode-length",
        type=int,
        default=1000000,
        help="Maximum length of an episode (default: 1000000)",
    )
    parser.add_argument(
        "--env-name",
        default="PongDeterministic-v4",
        help="Environment to train on (default: PongDeterministic-v4)",
    )
    parser.add_argument(
        "--no-shared",
        action="store_true",
        default=False,
        help="Use an optimizer without shared momentum.",
    )

    return parser.parse_args()


def main():
    """Main entry point for the A3C training."""
    os.environ["OMP_NUM_THREADS"] = "1"  # Limit CPU threads
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU usage

    args = parse_arguments()
    torch.manual_seed(args.seed)  # Set random seed for reproducibility

    # Initialize environment and shared model
    env = make_gym_env(args.env_name)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    # Flatten state and action shapes
    obs_dim = int(np.prod(state_shape))
    action_dim = int(np.prod(action_shape))

    # Create a local model
    shared_model = ActorCritic(
        obs_dim=obs_dim, hidden_dim=args.hidden_dim, action_dim=action_dim
    )
    # Initialize optimizer, shared across processes
    if args.no_shared:
        optimizer: Optional[torch.optim.Optimizer] = None
    else:
        optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    # Shared counter and lock for multiprocessing
    counter = mp.Value("i", 0)  # Global step counter
    lock = mp.Lock()  # Multiprocessing lock for shared resources

    # Create and start test process
    processes: List[mp.Process] = []
    test_process = mp.Process(
        target=test, args=(args.num_processes, args, shared_model, counter)
    )
    test_process.start()
    processes.append(test_process)

    # Create and start training processes
    for rank in range(0, args.num_processes):
        p = mp.Process(
            target=train, args=(rank, args, shared_model, counter, lock, optimizer)
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
