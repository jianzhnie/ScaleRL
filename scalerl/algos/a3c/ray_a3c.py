import gymnasium as gym
import numpy as np
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ActorCritic(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, output_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return policy, value


@ray.remote(num_gpus=1)
class A3CWorker:

    def __init__(self,
                 worker_id,
                 global_network,
                 optimizer,
                 env_name,
                 gamma=0.99):
        self.worker_id = worker_id
        self.global_network = global_network
        self.optimizer = optimizer
        self.env = gym.make(env_name)
        self.gamma = gamma
        self.local_network = ActorCritic(self.env.observation_space.shape[0],
                                         self.env.action_space.n)

    def synchronize_with_global(self):
        self.local_network.load_state_dict(self.global_network.state_dict())

    def compute_loss(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).view(-1, 1)
        rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).view(-1, 1)

        # Compute values and policies
        with torch.no_grad():
            _, next_values = self.global_network(next_states)
        policy, values = self.local_network(states)
        dist = torch.distributions.Categorical(policy)
        log_probs = dist.log_prob(actions)
        # Compute advantages
        advantages = rewards + self.gamma * next_values * (1 - dones) - values

        # Compute losses
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        entropy_loss = -(log_probs.exp() * log_probs).mean()

        total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
        return total_loss

    def train(self, num_steps):
        self.synchronize_with_global()
        states, actions, rewards, next_states, dones = [], [], [], [], []
        state, info = self.env.reset()
        done = False

        for _ in range(num_steps):
            state_tensor = torch.tensor(state,
                                        dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                policy, _ = self.local_network(state_tensor)
            action = np.random.choice(self.env.action_space.n,
                                      p=policy.numpy()[0])
            next_state, reward, done, _, _ = self.env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            state = next_state
            if done:
                state, info = self.env.reset()

        loss = self.compute_loss(states, actions, rewards, next_states, dones)
        self.optimizer.zero_grad()
        loss.backward()
        for local_param, global_param in zip(self.local_network.parameters(),
                                             self.global_network.parameters()):
            global_param._grad = local_param.grad
        self.optimizer.step()

        return loss.item()


if __name__ == '__main__':
    ray.init(num_gpus=1)

    env_name = 'CartPole-v1'
    input_dim = 4
    output_dim = 2

    global_network = ActorCritic(input_dim, output_dim)
    optimizer = optim.Adam(global_network.parameters(), lr=0.001)

    workers = [
        A3CWorker.remote(i, global_network, optimizer, env_name)
        for i in range(10)
    ]

    for episode in range(1000):
        results = ray.get([worker.train.remote(20) for worker in workers])
        avg_loss = sum(results) / len(results)
        print(f'Episode {episode}, Average Loss: {avg_loss}')

    ray.shutdown()
