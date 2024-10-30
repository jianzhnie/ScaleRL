from abc import ABCMeta, abstractmethod
from typing import Any


class BaseAgent(metaclass=ABCMeta):
    """Abstract base class for reinforcement learning agents.

    This class defines the common interface and methods that any agent should
    implement.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Initialize the AgentBase.

        This method should be overridden by subclasses to initialize the
        agent's specific architecture.
        """
        super(BaseAgent, self).__init__()

    @abstractmethod
    def get_action(self, *args: Any, **kwargs: Any) -> Any:
        """Return an action with noise when given the observation of the
        environment.

        This function is typically used during training to perform exploration and will usually do the following things:
           1. Accept numpy data as input;
           2. Feed numpy data or onvert numpy data to tensor (optional);
           3. Add sampling operation in numpy level.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: Action to be taken in the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Predict the action when given the observation of the environment.

        This function is often used during evaluation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: Estimated Q value or action.
        """
        raise NotImplementedError

    @abstractmethod
    def get_value(self, args: Any, **kwargs: Any) -> Any:
        """Get the value of the current state. This method should be overridden
        by subclasses to implement the value estimation.

        Args:
            args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: Value of the current state.
        """

        raise NotImplementedError

    @abstractmethod
    def get_weights(self) -> dict:
        """Get the current weights of the agent's neural network.

        Returns:
            dict: A dictionary containing the weights.
        """
        raise NotImplementedError('Subclasses should implement this method.')

    @abstractmethod
    def set_weights(self, weights: dict) -> None:
        """Set the weights of the agent's neural network.

        Args:
            weights (dict): A dictionary containing the weights to set.
        """
        raise NotImplementedError('Subclasses should implement this method.')

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Save the model to the specified directory.

        Args:
            path (str): Directory to save the model.
        """
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load the model from the specified directory.

        Args:
            path (str): Directory to load the model from.
        """
        raise NotImplementedError

    def name(self) -> str:
        """Get the name of the agent class in lowercase.

        Returns:
            str: The name of the agent class in lowercase.
        """
        return self.__class__.__name__.lower()
