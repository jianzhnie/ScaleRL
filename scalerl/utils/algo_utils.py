from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Union

import torch
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from torch import nn


def unwrap_optimizer(
    optimizer: Union[AcceleratedOptimizer, torch.optim.Optimizer],
    network: Union[nn.Module, List[nn.Module], Tuple[nn.Module]],
    lr: float,
) -> torch.optim.Optimizer:
    """Unwraps an AcceleratedOptimizer to a standard PyTorch optimizer.

    :param optimizer: The optimizer to unwrap.
    :type optimizer: Union[AcceleratedOptimizer, torch.optim.Optimizer]
    :param network: The network or list of networks whose parameters the optimizer should manage.
    :type network: Union[nn.Module, List[nn.Module], Tuple[nn.Module]]
    :param lr: The learning rate for the optimizer.
    :type lr: float
    :return: The unwrapped optimizer.
    :rtype: torch.optim.Optimizer
    """
    if isinstance(optimizer, AcceleratedOptimizer):
        if isinstance(network, (list, tuple)):
            optim_arg = [{
                'params': net.parameters(),
                'lr': lr
            } for net in network]
            unwrapped_optimizer = type(optimizer.optimizer)(optim_arg)
        else:
            unwrapped_optimizer = type(optimizer.optimizer)(
                network.parameters(), lr=lr)
        unwrapped_optimizer.load_state_dict(optimizer.state_dict())
        return unwrapped_optimizer
    else:
        return optimizer


def chkpt_attribute_to_device(chkpt_dict: Dict[str, Any],
                              device: str) -> Dict[str, Any]:
    """Places checkpoint attributes on the specified device.

    :param chkpt_dict: Checkpoint dictionary.
    :type chkpt_dict: Dict[str, Any]
    :param device: Device for accelerated computing, 'cpu' or 'cuda'.
    :type device: str
    :return: Checkpoint dictionary with attributes moved to the specified device.
    :rtype: Dict[str, Any]
    """
    for key, value in chkpt_dict.items():
        if hasattr(value, 'device') and not isinstance(value, Accelerator):
            chkpt_dict[key] = value.to(device)
    return chkpt_dict


def key_in_nested_dict(nested_dict: Dict[str, Any], target: str) -> bool:
    """Helper function to determine if a key is in a nested dictionary.

    :param nested_dict: Nested dictionary.
    :type nested_dict: Dict[str, Any]
    :param target: Target string to search for.
    :type target: str
    :return: True if the key is found, False otherwise.
    :rtype: bool
    """
    for k, v in nested_dict.items():
        if k == target:
            return True
        if isinstance(v, dict):
            if key_in_nested_dict(v, target):
                return True
    return False


def compile_model(model: nn.Module,
                  mode: Union[str, None] = 'default') -> nn.Module:
    """Compiles a torch model if it is not already compiled.

    :param model: The torch model to compile.
    :type model: nn.Module
    :param mode: The torch compile mode, defaults to "default".
    :type mode: Union[str, None], optional
    :return: The compiled model.
    :rtype: nn.Module
    """
    if (not isinstance(model, torch._dynamo.eval_frame.OptimizedModule)
            and mode is not None):
        return torch.compile(model, mode=mode)
    return model


def remove_compile_prefix(state_dict: Dict[str, Any]) -> OrderedDict[str, Any]:
    """Removes the '_orig_mod' prefix from the state dictionary created by
    torch compile.

    :param state_dict: The model state dictionary.
    :type state_dict: Dict[str, Any]
    :return: The state dictionary with the prefix removed.
    :rtype: OrderedDict[str, Any]
    """
    return OrderedDict([(k.split('.', 1)[1],
                         v) if k.startswith('_orig_mod') else (k, v)
                        for k, v in state_dict.items()])
