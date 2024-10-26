from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_columns_initializer(weights: torch.Tensor,
                                   std: float = 1.0) -> torch.Tensor:
    """Initializes the weights of the given tensor using a normalized column-
    based initialization.

    Args:
        weights (torch.Tensor): The weight tensor to initialize.
        std (float): The standard deviation to use for normalization.

    Returns:
        torch.Tensor: The initialized weights tensor.
    """
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m: nn.Module) -> None:
    """Initializes the weights and biases of a module using uniform
    distribution. It applies specific initialization strategies for
    Convolutional and Linear layers.

    Args:
        m (nn.Module): The layer (module) whose weights need to be initialized.
    """
    classname = m.__class__.__name__

    # Initialization for Convolutional layers
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(
            weight_shape[1:4])  # Product of input feature dimensions
        fan_out = (np.prod(weight_shape[2:4]) * weight_shape[0]
                   )  # Product of output feature dimensions
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

    # Initialization for Linear layers
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]  # Input dimensions
        fan_out = weight_shape[0]  # Output dimensions
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorCritic(nn.Module):
    """The ActorCritic model consists of convolutional layers followed by an
    LSTM and two linear layers. It outputs both the critic value (for value-
    based RL) and the actor policy (for action selection).

    Attributes:
        conv1, conv2, conv3, conv4 (nn.Conv2d): Four convolutional layers for processing input images.
        lstm (nn.LSTMCell): LSTM cell for maintaining the hidden state across time steps.
        critic_linear (nn.Linear): Linear layer for estimating the value function (critic).
        actor_linear (nn.Linear): Linear layer for producing the action policy (actor).
    """

    def __init__(self, num_inputs: int, action_dim: torch.Tensor) -> None:
        """Initializes the ActorCritic model architecture.

        Args:
            num_inputs (int): Number of input channels (e.g., the number of image channels).
            action_dim (torch.Tensor): Dimensionality of the action space.
        """
        super(ActorCritic, self).__init__()

        # Define convolutional layers
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        # Define LSTM cell
        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)

        # Define actor and critic layers
        self.critic_linear = nn.Linear(256,
                                       1)  # Single output for value prediction
        self.actor_linear = nn.Linear(256, action_dim)
        # Output for action probabilities

        # Apply the custom weights initialization function
        self.apply(weights_init)

        # Custom initialization for actor and critic layers
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        # Zero out the LSTM biases
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        # Set the model to training mode
        self.train()

    def forward(
        self, inputs: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the ActorCritic model.

        Args:
            inputs (Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
                A tuple containing the input tensor and the LSTM hidden states (hx, cx).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - The critic value estimation (value function).
                - The actor's action logits (before applying softmax).
                - The new hidden states (hx, cx) of the LSTM.
        """
        input_data, (hx, cx) = inputs

        # Pass input through convolutional layers with ELU activation
        x = F.elu(self.conv1(input_data))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        # Flatten the output of the final convolutional layer
        x = x.view(-1, 32 * 3 * 3)

        # Pass through LSTM cell
        hx, cx = self.lstm(x, (hx, cx))
        x = hx  # Use the hidden state for the next layers

        value = self.critic_linear(x)
        policy = self.actor_linear(x)
        # Return the critic value, actor logits, and the new LSTM hidden states
        return value, policy, (hx, cx)
