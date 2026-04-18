from __future__ import annotations

import numpy as np
import torch


def complex_np_to_channels(array: np.ndarray) -> np.ndarray:
    if not np.iscomplexobj(array):
        raise ValueError("Expected complex numpy array.")
    return np.stack([array.real, array.imag], axis=1).astype(np.float32, copy=False)


def channels_to_complex_np(array: np.ndarray) -> np.ndarray:
    if array.ndim < 2 or array.shape[1] != 2:
        raise ValueError("Expected [N, 2, ...] array.")
    return array[:, 0] + 1j * array[:, 1]


def complex_tensor_to_channels(tensor: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(tensor):
        raise ValueError("Expected complex tensor.")
    return torch.stack([tensor.real, tensor.imag], dim=1)


def channels_to_complex_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 2 or tensor.shape[1] != 2:
        raise ValueError("Expected [N, 2, ...] tensor.")
    return torch.complex(tensor[:, 0], tensor[:, 1])


def normalized_complex_mse_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    error_power = (prediction - target).pow(2).sum(dim=1).sum(dim=(-1, -2))
    target_power = target.pow(2).sum(dim=1).sum(dim=(-1, -2)).clamp_min(eps)
    return (error_power / target_power).mean()


def nmse_linear(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    error_power = (prediction - target).pow(2).sum(dim=1).sum(dim=(-1, -2))
    target_power = target.pow(2).sum(dim=1).sum(dim=(-1, -2)).clamp_min(eps)
    return (error_power / target_power).mean()


def nmse_db(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return 10.0 * torch.log10(nmse_linear(prediction, target, eps=eps))

