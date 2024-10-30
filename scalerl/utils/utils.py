from typing import Dict, List, Union

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


def calculate_mean(dict_list: List[Dict[str, float]]) -> Dict[str, float]:
    """计算包含字典的列表中每个键的均值。

    Args:
        dict_list (List[Dict[str, float]]): 包含字典的列表。

    Returns:
        Dict[str, float]: 每个键的均值字典。
    """
    # 检查是否为空列表
    if not dict_list:
        return {}

    # 初始化累加器字典
    sum_dict = {key: 0 for key in dict_list[0].keys()}

    # 累加每个字典中的值
    for d in dict_list:
        for key, value in d.items():
            sum_dict[key] += value

    # 计算均值
    mean_dict = {
        key: total / len(dict_list)
        for key, total in sum_dict.items()
    }

    return mean_dict
