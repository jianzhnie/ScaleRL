from typing import Any


class ParameterServer:
    """A simple parameter server class for managing and updating weights.

    Attributes:
        weights (Any): The current weights managed by the parameter server.
    """

    def __init__(self, weights: Any):
        """Initializes the ParameterServer with the given weights.

        Args:
            weights (Any): The initial weights to be managed by the parameter server.
        """
        self.weights = weights

    def push(self, weights: Any) -> None:
        """Updates the current weights with the provided weights.

        Args:
            weights (Any): The new weights to be set.
        """
        self.weights = weights

    def pull(self) -> Any:
        """Retrieves the current weights.

        Returns:
            Any: The current weights managed by the parameter server.
        """
        return self.weights
