from typing import Union

import torch


def get_device(device: Union[torch.device, str] = 'auto') -> torch.device:
    """Retrieve PyTorch device. It checks that the requested device is
    available first. For now, it supports only cpu and cuda. By default, it
    tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    # Cuda by default
    if device == 'auto':
        device = 'cuda'
    # Force conversion to th.device
    device = torch.device(device)

    # Cuda not available
    if device.type == torch.device(
            'cuda').type and not torch.cuda.is_available():
        return torch.device('cpu')

    return device
