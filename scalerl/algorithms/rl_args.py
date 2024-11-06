from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class RLArguments:
    """Common settings for Reinforcement Learning algorithms."""

    # Common settings
    project: str = field(
        default='rltoolkit',
        metadata={'help': "Name of the project. Defaults to 'rltoolkit'"},
    )
    algo_name: str = field(
        default='dqn',
        metadata={'help': "Name of the algorithm. Defaults to 'dqn'"},
    )
    use_cuda: bool = field(
        default=True,
        metadata={'help': 'Whether to use CUDA. Defaults to True'},
    )
    device: str = field(
        default='cuda' if torch.cuda.is_available() else 'cpu',
        metadata={
            'help':
            'Device to use for computation. Defaults to CUDA if available, else CPU'
        },
    )

    torch_deterministic: bool = field(
        default=False,
        metadata={
            'help':
            'Whether to use deterministic operations in CUDA. Defaults to True'
        },
    )
    seed: int = field(
        default=42,
        metadata={
            'help': 'Seed for environment randomization. Defaults to 42'
        },
    )
    # Environment specific settings
    env_id: str = field(
        default='CartPole-v0',
        metadata={'help': 'Environment ID (default: CartPole-v1)'},
    )
    num_envs: int = field(
        default=4,
        metadata={
            'help':
            'Number of parallel environments to run for collecting experiences. Defaults to 10'
        },
    )
    capture_video: Optional[bool] = field(
        default=None,
        metadata={
            'help':
            'Flag indicating whether to capture videos of the environment during training.'
        },
    )
    # ReplayBuffer settings
    buffer_size: int = field(
        default=10000,
        metadata={
            'help': 'Maximum size of the replay buffer. Defaults to 10000'
        },
    )
    batch_size: int = field(
        default=32,
        metadata={
            'help':
            'Size of the mini-batches sampled from the replay buffer during training. Defaults to 32'
        },
    )
    # Training parameters
    max_timesteps: int = field(
        default=10000,
        metadata={
            'help': 'Maximum number of training steps. Defaults to 12000'
        },
    )
    rollout_length: int = field(
        default=200, metadata={'help': 'The rollout length (time dimension)'})
    eval_episodes: int = field(
        default=5,
        metadata={'help': 'Number of episodes to evaluate. Defaults to 10'},
    )

    # Hyperparameters
    n_steps: bool = field(
        default=False,
        metadata={
            'help':
            'Use multi-step experience replay buffer, defaults to False'
        },
    )
    gamma: float = field(
        default=0.99,
        metadata={
            'help': 'Discount factor for future rewards. Defaults to 0.99'
        },
    )
    epsilon_greedy: float = field(
        default=0.01, metadata={'help': 'The probability for exploration'})
    max_grad_norm: float = field(default=40.0,
                                 metadata={'help': 'Max norm of gradients'})

    # Optimizer settings
    learning_rate: float = field(default=0.0001,
                                 metadata={'help': 'Learning rate'})
    alpha: float = field(default=0.99,
                         metadata={'help': 'RMSProp smoothing constant'})
    momentum: float = field(default=0.0, metadata={'help': 'RMSProp momentum'})
    epsilon: float = field(default=1e-5, metadata={'help': 'RMSProp epsilon'})

    # Logging and saving
    work_dir: str = field(
        default='work_dirs',
        metadata={
            'help':
            "Directory for storing work-related files. Defaults to 'work_dirs'"
        },
    )
    save_model: Optional[bool] = field(
        default=False,
        metadata={
            'help': 'Flag indicating whether to save the trained model.'
        },
    )
    train_log_interval: int = field(
        default=100,
        metadata={'help': 'Logging interval during training. Defaults to 10'},
    )
    test_log_interval: int = field(
        default=500,
        metadata={
            'help': 'Logging interval during evaluation. Defaults to 20'
        },
    )
    save_interval: int = field(
        default=1000,
        metadata={'help': 'Frequency of saving the model. Defaults to 1000'},
    )
    logger: str = field(
        default='tensorboard',
        metadata={
            'help': "Logger to use for recording logs. Defaults to 'wandb'"
        },
    )

    # MultiProcess settings
    num_actors: int = field(
        default=4,
        metadata={'help': 'The number of actors for each simulation device'})
    num_learners: int = field(default=1,
                              metadata={'help': 'Number learner threads'})


@dataclass
class DQNArguments(RLArguments):
    """DQN-specific settings."""

    per: bool = field(
        default=False,
        metadata={
            'help':
            'Flag indicating whether to use Prioritized Experience Replay. Defaults to False'
        },
    )
    hidden_dim: int = field(
        default=128,
        metadata={
            'help':
            'The hidden dimension size of the neural network. Defaults to 128'
        },
    )
    double_dqn: bool = field(
        default=False,
        metadata={
            'help':
            'Flag indicating whether to use Double DQN. Defaults to False'
        },
    )
    dueling_dqn: bool = field(
        default=False,
        metadata={
            'help':
            'Flag indicating whether to use Dueling DQN. Defaults to False'
        },
    )
    noisy_dqn: bool = field(
        default=False,
        metadata={
            'help':
            'Flag indicating whether to use Noisy DQN. Defaults to False'
        },
    )
    categorical_dqn: bool = field(
        default=False,
        metadata={
            'help':
            'Flag indicating whether to use Categorical DQN. Defaults to False'
        },
    )
    v_min: float = field(
        default=0.0,
        metadata={
            'help': 'Minimum value for the value function. Defaults to 0.0'
        },
    )
    v_max: float = field(
        default=200.0,
        metadata={
            'help': 'Maximum value for the value function. Defaults to 200.0'
        },
    )
    num_atoms: float = field(
        default=51,
        metadata={
            'help': 'Number of atoms for the value function. Defaults to 51'
        },
    )
    noisy_std: float = field(
        default=0.5,
        metadata={
            'help':
            'Standard deviation for the initial weights of the value function. Defaults to 0.1'
        },
    )
    learning_rate: float = field(
        default=1e-3,
        metadata={
            'help': 'Learning rate used by the optimizer. Defaults to 1e-4'
        },
    )
    min_learning_rate: float = field(
        default=1e-5,
        metadata={
            'help':
            'Minimum learning rate used by the optimizer. Defaults to 1e-5'
        },
    )
    lr_scheduler_method: str = field(
        default='linear',
        metadata={
            'help':
            "Method used for learning rate scheduling. Defaults to 'linear'"
        },
    )
    eps_greedy_start: float = field(
        default=1.0,
        metadata={
            'help':
            'Initial value of epsilon for epsilon-greedy exploration. Defaults to 1.0'
        },
    )
    eps_greedy_end: float = field(
        default=0.1,
        metadata={
            'help':
            'Final value of epsilon for epsilon-greedy exploration. Defaults to 0.1'
        },
    )
    eps_greedy_scheduler: str = field(
        default='linear',
        metadata={
            'help':
            "Type of scheduler used for epsilon-greedy exploration. Defaults to 'linear'"
        },
    )
    max_grad_norm: float = field(
        default=None,
        metadata={'help': 'Maximum gradient norm. Defaults to 1.0'},
    )
    use_smooth_l1_loss: bool = field(
        default=False,
        metadata={
            'help':
            'Flag indicating whether to use the smooth L1 loss. Defaults to False'
        },
    )
    warmup_learn_steps: int = field(
        default=1000,
        metadata={
            'help':
            'Number of steps before starting to update the model. Defaults to 1000'
        },
    )
    target_update_frequency: int = field(
        default=100,
        metadata={
            'help': 'Frequency of updating the target network. Defaults to 100'
        },
    )
    soft_update_tau: float = field(
        default=1.0,
        metadata={
            'help':
            'Interpolation parameter for soft target updates. Defaults to 1.0'
        },
    )
    train_frequency: int = field(
        default=10,
        metadata={'help': 'Frequency of training updates. Defaults to 1'},
    )
    learn_steps: int = field(
        default=1,
        metadata={
            'help':
            'Number of times to update the learner network. Defaults to 1'
        },
    )


@dataclass
class A3CArguments:
    """Command-line argument settings for A3C algorithm."""

    env_name: str = field(
        default='CartPole-v0',
        metadata={'help': 'Environment to train on (default: CartPole-v0)'},
    )
    seed: int = field(default=1, metadata={'help': 'Random seed (default: 1)'})

    hidden_dim: int = field(
        default=8, metadata={'help': 'Hidden dimension (default: 256)'})
    max_episode_size: int = field(
        default=10000,
        metadata={'help': 'Max train eposide size (default: 10000)'})
    lr: float = field(default=0.0001,
                      metadata={'help': 'Learning rate (default: 0.0001)'})
    gamma: float = field(
        default=0.99,
        metadata={'help': 'Discount factor for rewards (default: 0.99)'})
    gae_lambda: float = field(
        default=1.00,
        metadata={'help': 'Lambda parameter for GAE (default: 1.00)'})
    entropy_coef: float = field(
        default=0.01,
        metadata={'help': 'Entropy term coefficient (default: 0.01)'})
    value_loss_coef: float = field(
        default=0.5,
        metadata={'help': 'Value loss coefficient (default: 0.5)'})
    max_grad_norm: float = field(
        default=50.0,
        metadata={'help': 'Maximum gradient norm (default: 50.0)'})
    num_processes: int = field(
        default=4,
        metadata={'help': 'Number of training processes to use (default: 4)'})
    num_steps: int = field(
        default=20,
        metadata={'help': 'Number of forward steps in A3C (default: 20)'})
    max_episode_length: int = field(
        default=1000000,
        metadata={'help': 'Maximum length of an episode (default: 1000000)'},
    )
    no_shared: bool = field(
        default=False,
        metadata={'help': 'Use an optimizer without shared momentum.'})
