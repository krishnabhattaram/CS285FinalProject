import torch


def lrp_0(x:torch.Tensor) -> torch.Tensor:
    r"""Computes Layer Relevance Propagation-0.

    .. math::
       y = x

    Args:
        x (torch.Tensor): input tensor.

    Returns:
        torch.Tensor: identity of input.

    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    
    return x


def lrp_gamma(x:torch.Tensor, gamma:float = 0.0) -> torch.Tensor:
    r""" Computes Layer relevance Propagation - 0
    
    
    .. math::
       y = x + gamma*max(0, x)
    Args:
        x (torch.Tensor): input tensor.
        gamma (float): value of gamma for LRP-gamma

    Returns:
        torch.Tensor: inverse soft-plus of input.
    
    
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    
    return x + gamma * torch.clamp(x, 0)