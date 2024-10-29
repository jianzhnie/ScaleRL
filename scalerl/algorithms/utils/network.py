import torch
import torch.nn as nn


class QNet(nn.Module):
    """A simple feedforward neural network for Q-learning.

    This network takes in observations and outputs Q-values for each action.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        """Initializes the QNet.

        :param obs_dim: Dimension of the observation space.
        :type obs_dim: int
        :param action_dim: Dimension of the action space.
        :type action_dim: int
        :param hidden_dim: Dimension of the hidden layers, defaults to 128.
        :type hidden_dim: int, optional
        """
        super(QNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the QNet.

        :param obs: Observation tensor.
        :type obs: torch.Tensor
        :return: Q-values for each action.
        :rtype: torch.Tensor
        """
        return self.network(obs)


class ActorNet(nn.Module):

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(ActorNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.net(obs)
        return logits


class CriticNet(nn.Module):

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim):
        super(CriticNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ActorCriticNet(nn.Module):

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(ActorCriticNet, self).__init__()
        self.actor = ActorNet(obs_dim, hidden_dim, action_dim)
        self.critic = CriticNet(obs_dim, hidden_dim, action_dim)

    def network_init(self) -> None:
        for layer in self.modules:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.actor(obs)
        value = self.critic(obs)
        return logits, value
