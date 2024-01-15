import numpy as np
import torch
from torch.nn import functional

def shifted_softplus(x: torch.Tensor) -> torch.Tensor:
    r"""Compute shifted soft-plus activation function.

    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)

    Args:
        x (torch.Tensor): input tensor.

    Returns:
        torch.Tensor: shifted soft-plus of input.

    """
    return functional.softplus(x) - np.log(2.0)


def softplus_inverse(x:torch.Tensor) -> torch.Tensor:
    """
    Inverse of the softplus function. This is useful for initialization of
    parameters that are constrained to be positive (via softplus).
    
    .. math::
       y = x + \ln\left(1 - e^{-x}\right)

    Args:
        x (torch.Tensor): input tensor.

    Returns:
        torch.Tensor: inverse soft-plus of input.
    
    
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x + torch.log(-torch.expm1(-x))