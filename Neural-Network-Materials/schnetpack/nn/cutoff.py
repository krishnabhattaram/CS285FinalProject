import numpy as np
import torch
from torch import nn


__all__ = ["CosineCutoff", "MollifierCutoff", "HardCutoff", "GaussianCutoff", "get_cutoff_by_string"]


def get_cutoff_by_string(key):
    # build cutoff module
    if key == "hard":
        cutoff_network = HardCutoff
    elif key == "cosine":
        cutoff_network = CosineCutoff
    elif key == "mollifier":
        cutoff_network = MollifierCutoff
    elif key == "gaussian":
        cutoff_network = GaussianCutoff
    else:
        raise NotImplementedError("cutoff_function {} is unknown".format(key))
    return cutoff_network


class CosineCutoff(nn.Module):
    r"""Class of Behler cosine cutoff.

    .. math::
       f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff (float, optional): cutoff radius.

    """

    def __init__(self, cutoff:float=5.0):
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))
        
        self.relevance = None

    def forward(self, distances:torch.Tensor) -> torch.Tensor:
        """Compute cutoff.

        Args:
            distances (torch.Tensor): values of interatomic distances.

        Returns:
            torch.Tensor: values of cutoff function.

        """
        # Compute values of cutoff function
        cutoffs = 0.5 * (torch.cos(distances * np.pi / self.cutoff) + 1.0)
        # Remove contributions beyond the cutoff radius
        cutoffs *= (distances < self.cutoff).float()
        return cutoffs


    def relprop(self,
                distances:torch.Tensor,
                relevance:torch.Tensor) -> torch.Tensor:
        """Compute layer relevance for its inputs.

        Args:
            distances (dict of torch.Tensor): batch of input values.
            relevance (torch.tensor): relevance of input distances

        Returns:
            torch.Tensor : output relevance of inputs.

        """
        self.relevance = relevance*(distances < self.cutoff).float()
        return self.relevance

class MollifierCutoff(nn.Module):
    r"""Class for mollifier cutoff scaled to have a value of 1 at :math:`r=0`.

    .. math::
       f(r) = \begin{cases}
        \exp\left(1 - \frac{1}{1 - \left(\frac{r}{r_\text{cutoff}}\right)^2}\right)
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff (float, optional): Cutoff radius.
        eps (float, optional): offset added to distances for numerical stability.

    """

    def __init__(self, cutof:float=5.0, eps:float=1.0e-7):
        super(MollifierCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))
        self.register_buffer("eps", torch.FloatTensor([eps]))
        
        self.relevance = None

    def forward(self, distances:torch.Tensor) -> torch.Tensor:
        """Compute cutoff.

        Args:
            distances (torch.Tensor): values of interatomic distances.

        Returns:
            torch.Tensor: values of cutoff function.

        """
        mask = (distances + self.eps < self.cutoff).float()
        exponent = 1.0 - 1.0 / (1.0 - torch.pow(distances * mask / self.cutoff, 2))
        cutoffs = torch.exp(exponent)
        cutoffs = cutoffs * mask
        return cutoffs
        
    def relprop(self,
                distances:torch.Tensor,
                relevance:torch.Tensor) -> torch.Tensor:
        """Compute layer relevance for its inputs.

        Args:
            distances (dict of torch.Tensor): batch of input values.
            relevance (torch.tensor): relevance of input distances

        Returns:
            torch.Tensor : output relevance of inputs.

        """
        self.relevance = relevance*(distances < self.cutoff).float()
        return self.relevance


class HardCutoff(nn.Module):
    r"""Class of hard cutoff.

    .. math::
       f(r) = \begin{cases}
        1 & r \leqslant r_\text{cutoff} \\
        0 & r > r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff (float): cutoff radius.

    """

    def __init__(self, cutoff:float=5.0):
        super(HardCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))
        
        self.relevance = None

    def forward(self, distances:torch.Tensor) -> torch.Tensor:
        """Compute cutoff.

        Args:
            distances (torch.Tensor): values of interatomic distances.

        Returns:
            torch.Tensor: values of cutoff function.

        """
        mask = (distances <= self.cutoff).float()
        return mask
        
    def relprop(self,
                distances:torch.Tensor,
                relevance:torch.Tensor) -> torch.Tensor:
        """Compute layer relevance for its inputs.

        Args:
            distances (dict of torch.Tensor): batch of input values.
            relevance (torch.tensor): relevance of input distances

        Returns:
            torch.Tensor : output relevance of inputs.

        """
        self.relevance = relevance*(distances < self.cutoff).float()
        return self.relevance
        
        
class GaussianCutoff(nn.Module):
    r"""Class of gaussian cutoff.

    .. math::
       f(r) = \begin{cases}
        \exp\left(- \frac{r^2}{(r_\text{cut} - r)(r_\text{cut} + r)} \right) & r < r_\text{cut} \\
        0 & r >= r_\text{cut} \\
        \end{cases}

    Args:
        cutoff (float): cutoff radius.

    """

    def __init__(self, cutoff:float=5.0):
        super(GaussianCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))
        
        self.relevance = None

    def forward(self, distances:torch.Tensor) -> torch.Tensor:
        """Compute cutoff.

        Args:
            distances (torch.Tensor): values of interatomic distances.

        Returns:
            torch.Tensor: values of cutoff function.

        """
        mask = (distances < self.cutoff).float()
        '''Added 1e-3 so that it doesn't explode when distances > cutoff due to precision error '''
        exponent = -(distances ** 2) / ((self.cutoff - distances + 1e-3) * (self.cutoff + distances))
        cutoffs = torch.exp(exponent)
        cutoffs = cutoffs * mask
        
        return cutoffs
        
    
    def relprop(self,
                distances:torch.Tensor,
                relevance:torch.Tensor) -> torch.Tensor:
        """Compute layer relevance for its inputs.

        Args:
            distances (dict of torch.Tensor): batch of input values.
            relevance (torch.tensor): relevance of input distances

        Returns:
            torch.Tensor : output relevance of inputs.

        """
        self.relevance = relevance*(distances < self.cutoff).float()
        return self.relevance
