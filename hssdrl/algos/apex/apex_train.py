import multiprocessing as mp

import gymnasium as gym
import torch

from hssdrl.algos.apex.memory import PrioritizedReplayBuffer
from hssdrl.algos.apex.network import QNet
from hssdrl.algos.apex.worker import Actor, Learner


class ApexTrainer(object):

    def __init__(
        self,
        env_name: str,
        num_actors: int = 4,
        max_timesteps: int = 1000,
        buffer_size: int = 10000,
        batch_size: int = 32,
        eval_interval: int = 1000,
        train_log_interval: int = 1000,
        test_log_interval: int = 1000,
        target_update_frequency: int = 2000,
        gamma: float = 0.99,
        epsilon: float = 0.9,
        alpha: float = 0.6,
        beta: float = 0.4,
        learning_rate: float = 0.0001,
        device: str = 'cpu',
    ) -> None:
        self.env_name = env_name
        self.num_actors = num_actors
        self.max_timesteps = max_timesteps
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.eval_interval = eval_interval
        self.train_log_interval = train_log_interval
        self.test_log_interval = test_log_interval
        self.target_update_frequency = target_update_frequency

        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device

        self.env = gym.make(id=env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.model = QNet(self.state_dim, self.action_dim)
        self.target_model = QNet(self.state_dim, self.action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size=buffer_size,
                                                     alpha=alpha,
                                                     beta=beta)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate)
        self.global_step = mp.Value('i', 0)
        self.global_epsode = mp.Value('i', 0)
        self.global_episode_reward = mp.Value('d', 0)
        self.global_buffer_pipe = mp.Queue()

    def train(self) -> None:
        actors = []
        for actor_id in range(len(self.num_actors)):
            actor_name = 'actor' + str(actor_id)
            actor = Actor(
                actor_name,
                self.env_name,
                self.replay_buffer,
                eps=self.epsilon,
                gamma=self.gamma,
            )
            actor.start()
            actors.append(actor)

        learner = Learner(
            self.model,
            self.target_model,
            self.replay_buffer,
            self.global_buffer_pipe,
            self.batch_size,
        )
        learner.start()

        for _ in range(1000):
            learner.train()
            learner.update_target_model()

        for p in actors:
            p.join()


if __name__ == '__main__':
    trainer = ApexTrainer(env_name='CartPole-v1')
    trainer.train()
