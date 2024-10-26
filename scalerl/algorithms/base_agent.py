from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentBase(nn.Module, metaclass=ABCMeta):
    """Abstract base class for reinforcement learning agents.

    This class defines the common interface and methods that any agent should
    implement.
    """

    @abstractmethod
    def __init__(self):
        """Initialize the AgentBase.

        This method should be overridden by subclasses to initialize the
        agent's specific architecture.
        """
        super(AgentBase, self).__init__()

    @abstractmethod
    def put_data(self, transition: dict) -> None:
        """Store a transition (state, action, reward, next_state, done) in the
        agent's memory.

        Args:
            transition (dict): A dictionary containing the transition data.
        """
        pass

    @abstractmethod
    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        """Get the action to take given the current state.

        Args:
            x (torch.Tensor): The current state.

        Returns:
            torch.Tensor: The action to take.
        """
        pass

    @abstractmethod
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get the value of the current state.

        Args:
            x (torch.Tensor): The current state.

        Returns:
            torch.Tensor: The value of the current state.
        """
        pass

    @abstractmethod
    def get_weights(self) -> dict:
        """Get the current weights of the agent's neural network.

        Returns:
            dict: A dictionary containing the weights.
        """
        pass

    @abstractmethod
    def set_weights(self, weights: dict) -> None:
        """Set the weights of the agent's neural network.

        Args:
            weights (dict): A dictionary containing the weights to set.
        """
        pass

    @abstractmethod
    def get_gradients(self) -> dict:
        """Get the current gradients of the agent's neural network.

        Returns:
            dict: A dictionary containing the gradients.
        """
        pass

    @abstractmethod
    def set_gradients(self, gradients: dict) -> None:
        """Set the gradients of the agent's neural network.

        Args:
            gradients (dict): A dictionary containing the gradients to set.
        """
        pass


class Agent(AgentBase):
    """Concrete implementation of the AgentBase class.

    This class provides the specific implementation for a reinforcement
    learning agent.
    """

    def __init__(self, state_dim: int, action_dim: int, args: Dict):
        """Initialize the Agent.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            args (Dict): Additional arguments required for initialization.
        """
        super(Agent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.args = args
        self.data = self.args.get('data',
                                  None)  # Assuming 'data' is a key in args

    def name(self) -> str:
        """Get the name of the agent class in lowercase.

        Returns:
            str: The name of the agent class in lowercase.
        """
        return self.__class__.__name__.lower()

    def put_data(self, transition: Dict) -> None:
        """Store a transition (state, action, reward, next_state, done) in the
        agent's memory.

        Args:
            transition (Dict): A dictionary containing the transition data.
        """
        if self.data is not None:
            self.data.put_data(transition)

    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        """Get the action to take given the current state.

        Args:
            x (torch.Tensor): The current state.

        Returns:
            torch.Tensor: The action to take.
        """
        raise NotImplementedError('Subclasses should implement this method.')

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get the value of the current state.

        Args:
            x (torch.Tensor): The current state.

        Returns:
            torch.Tensor: The value of the current state.
        """
        raise NotImplementedError('Subclasses should implement this method.')

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Get the current weights of the agent's neural network.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the weights.
        """
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Set the weights of the agent's neural network.

        Args:
            weights (Dict[str, torch.Tensor]): A dictionary containing the weights to set.
        """
        self.load_state_dict(weights)

    def get_gradients(self) -> List[Optional[torch.Tensor]]:
        """Get the current gradients of the agent's neural network.

        Returns:
            List[Optional[torch.Tensor]]: A list containing the gradients.
        """
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients: List[Optional[torch.Tensor]]) -> None:
        """Set the gradients of the agent's neural network.

        Args:
            gradients (List[Optional[torch.Tensor]]): A list containing the gradients to set.
        """
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = g.to(p.device)

    def add_gradients(self, gradients: List[Optional[torch.Tensor]]) -> None:
        """Add the given gradients to the current gradients of the agent's
        neural network.

        Args:
            gradients (List[Optional[torch.Tensor]]): A list containing the gradients to add.
        """
        for g, p in zip(gradients, self.parameters()):
            if g is None:
                continue
            if p.grad is None:
                p.grad = torch.zeros(g.shape).to(p.device)
            p.grad += g.to(p.device)


class ActorCritic(Agent):
    """Concrete implementation of the ActorCritic agent.

    This class extends the Agent class and provides specific implementations
    for an Actor-Critic architecture.
    """

    def __init__(self, state_dim: int, action_dim: int, args: Dict[str, Any]):
        """Initialize the ActorCritic agent.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            args (Dict[str, Any]): Additional arguments required for initialization.
        """
        super(ActorCritic, self).__init__(state_dim, action_dim, args)
        self.args = args

    def get_action(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Get the action to take given the current state.

        Args:
            x (torch.Tensor): The current state.

        Returns:
            Tuple[torch.Tensor, ...]: The action to take.
                If discrete, returns the probability distribution over actions.
                If continuous, returns the mean and standard deviation of the action distribution.
        """
        if self.args['discrete']:
            mu, _ = self.actor(x)
            prob = F.softmax(mu, dim=-1)
            return prob
        else:
            mu, std = self.actor(x)
            return mu, std

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get the value of the current state.

        Args:
            x (torch.Tensor): The current state.

        Returns:
            torch.Tensor: The value of the current state.
        """
        return self.critic(x)
