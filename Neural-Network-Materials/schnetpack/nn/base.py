import torch
import copy
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.functional import linear

from schnetpack.nn.initializers import zeros_initializer
from schnetpack.nn.lrp import lrp_0, lrp_gamma
from typing import Callable, Dict, Optional

__all__ = ["Dense", "GetItem", "ScaleShift", "RotateScaleShift", "Standardize", "Aggregate", "Collect"]


class Dense(nn.Linear):
    r"""Fully connected linear layer with activation function.

    .. math::
       y = activation(xW^T + b)

    Args:
        in_features (int): number of input feature :math:`x`.
        out_features (int): number of output features :math:`y`.
        bias (bool, optional): if False, the layer will not adapt bias :math:`b`.
        activation (callable, optional): if None, no activation function is used.
        weight_init (callable, optional): weight initializer from current weight.
        bias_init (callable, optional): bias initializer from current bias.

    """

    def __init__(
        self,
        in_features:int,
        out_features:int,
        bias:bool=True,
        activation:Optional[Callable]=None,
        weight_init:Callable = xavier_uniform_,
        bias_init:Callable = zeros_initializer,
    ):
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.activation = activation
         
        self.relevance = None
        # initialize linear layer y = xW^T + b
        super(Dense, self).__init__(in_features, out_features, bias)
        
    
    def reset_parameters(self) -> None:
        """Reinitialize model weight and bias values."""
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)
    
    def forward(self, inputs:Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute layer output.

        Args:
            inputs (dict of torch.Tensor): batch of input values.

        Returns:
            torch.Tensor: layer output.

        """
        # compute linear layer y = xW^T + b
        y = super(Dense, self).forward(inputs)
        # add activation function
        if self.activation:
            y = self.activation(y)
        return y
        

    def relprop(self,
                inputs:torch.Tensor,
                relevance:torch.Tensor,
                gamma:float = 0.0,
                epsilon:float = 0.0) -> torch.Tensor:
        """Compute layer relevance for its inputs.

        Args:
            inputs (torch.Tensor): input values.
            relevance (torch.tensor): relevance of outputs
            gamma (float value): gamma value for the LRP function to modify weights
            epsilon (float value): used if we are using LRP-epsilon
        
        Returns:
            torch.Tensor : layer relevance of inputs.

        """

        if not input.is_leaf:    # Store gradient if not a leaf tensor
            input.retain_grad()
            
        if input.grad is not None:  # Let it not accumulate previous gradients
            input.grad = None
        
        y_explain = linear(inputs, lrp_gamma(self.weight, gamma), lrp_gamma(self.bias, gamma)) + epsilon    
        
        s = relevance/(y_explain + 1e-9)
        (y_explain*s.data).sum().backward()
        
        self.relevance = inputs*inputs.grad
        
        return self.relevance
        
    
        
class GetItem(nn.Module):
    """Extraction layer to get an item from SchNetPack dictionary of input tensors.
    
    Args:
        key (str): Property to be extracted from SchNetPack input tensors.
    
    """

    def __init__(self, key:str):
        super(GetItem, self).__init__()
        self.key = key
        self.relevance = None

    def forward(self, inputs:Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute layer output.

        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        Returns:
            torch.Tensor: layer output.

        """
        return inputs[self.key]
        
        
    def relprop(self,
                inputs:Dict[str, torch.Tensor],
                relevance:torch.Tensor) -> torch.Tensor:
        """Compute layer relevance for its inputs.

        Args:
            inputs (dict of torch.Tensor): batch of input values. Just to match the syntax of other relprop functions 
            relevance (torch.tensor): relevance of outputs

        Returns:
            torch.Tensor : layer relevance of inputs

        """
        self.relevance = relevance
        return relevance


class ScaleShift(nn.Module):
    r"""Scale and shift layer for standardization.

    .. math::
       y = x \times \sigma + \mu

    Args:
        mean (torch.Tensor): mean value :math:`\mu`.
        stddev (torch.Tensor): standard deviation value :math:`\sigma`.

    """

    def __init__(self,
                 mean:torch.Tensor,
                 stddev:torch.Tensor):
        super(ScaleShift, self).__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)
        
        self.relevance = None

    def forward(self, input:Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute layer output.

        Args:
            input (torch.Tensor): input data.

        Returns:
            torch.Tensor: layer output.

        """
        y = input * self.stddev.reshape((1,1,-1)) + self.mean.reshape((1,1,-1))
        return y

        
    def relprop(self,
                inputs:Dict[str, torch.Tensor],
                relevance:torch.Tensor) -> torch.Tensor:
        """Compute layer relevance for its inputs.

        Args:
            inputs (dict of torch.Tensor): batch of input values. Just to match the syntax of other relprop functions 
            relevance (torch.tensor): relevance of outputs

        Returns:
            torch.Tensor : layer relevance of inputs

        """
        self.relevance = relevance
        return relevance
        
class RotateScaleShift(nn.Module):
    r"""Rotate and then scale and shift layer for standardization.

    .. math::
       y = (x @ rotate_basis) \times \sigma + \mu

    Args:
        rotate_basis (torch.Tensor): 
        mean (torch.Tensor): mean value :math:`\mu`.
        stddev (torch.Tensor): standard deviation value :math:`\sigma`.

    """

    def __init__(self,
                 rotate_basis:torch.Tensor,
                 mean:torch.Tensor,
                 stddev:torch.Tensor):
        super(RotateScaleShift, self).__init__()
        
        self.register_buffer("rotate_basis", rotate_basis)
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)
        
        self.relevance = None

    def forward(self, input:Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute layer output.

        Args:
            input (torch.Tensor): input data.

        Returns:
            torch.Tensor: layer output.

        """
        y = (input @ self.rotate_basis) * self.stddev.reshape((1,-1)) + self.mean.reshape((1,-1))
        
        return y

        
    def relprop(self,
                inputs:Dict[str, torch.Tensor],
                relevance:torch.Tensor) -> torch.Tensor:
        """Compute layer relevance for its inputs.

        Args:
            inputs (dict of torch.Tensor): batch of input values. Just to match the syntax of other relprop functions 
            relevance (torch.tensor): relevance of outputs

        Returns:
            torch.Tensor : layer relevance of inputs

        """
        self.relevance = relevance
        return relevance

class Standardize(nn.Module):
    r"""Standardize layer for shifting and scaling.

    .. math::
       y = \frac{x - \mu}{\sigma}

    Args:
        mean (torch.Tensor): mean value :math:`\mu`.
        stddev (torch.Tensor): standard deviation value :math:`\sigma`.
        eps (float, optional): small offset value to avoid zero division.

    """

    def __init__(self,
                 mean:torch.Tensor,
                 stddev:torch.Tensor,
                 eps:float = 1e-9):
        super(Standardize, self).__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)
        self.register_buffer("eps", torch.ones_like(stddev) * eps)
        
        self.relevance = None

    def forward(self, input:Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute layer output.

        Args:
            input (torch.Tensor): input data.

        Returns:
            torch.Tensor: layer output.

        """
        # Add small number to catch divide by zero
        y = (input - self.mean) / (self.stddev + self.eps)
        return y
        

    def relprop(self,
                inputs:Dict[str, torch.Tensor],
                relevance:torch.Tensor) -> torch.Tensor:
        """Compute layer relevance for its inputs.

        Args:
            inputs (dict of torch.Tensor): batch of input values. Just to match the syntax of other relprop functions 
            relevance (torch.tensor): relevance of outputs

        Returns:
            torch.Tensor : layer relevance of inputs

        """
        self.relevance = relevance
        return relevance


class Aggregate(nn.Module):
    """Pooling layer based on sum or average with optional masking.

    Args:
        axis (int): axis along which pooling is done.
        mean (bool, optional): if True, use average instead for sum pooling.
        keepdim (bool, optional): whether the output tensor has dim retained or not.

    """

    def __init__(self,
                 axis:int,
                 mean:bool = False,
                 keepdim:bool = True):
        super(Aggregate, self).__init__()
        self.average = mean
        self.axis = axis
        self.keepdim = keepdim
        
        self.relevance = None

    def forward(self,
                input:Dict[str, torch.Tensor],
                mask:Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Compute layer output.

        Args:
            input (torch.Tensor): input data.
            mask (torch.Tensor, optional): mask to be applied; e.g. neighbors mask.

        Returns:
            torch.Tensor: layer output.

        """
        # mask input
        if mask is not None:
            input = input * mask[..., None]
        # compute sum of input along axis
        y = torch.sum(input, self.axis)
        # compute average of input along axis
        if self.average:
            # get the number of items along axis
            if mask is not None:
                N = torch.sum(mask, self.axis, keepdim=self.keepdim)
                N = torch.max(N, other=torch.ones_like(N))
            else:
                N = input.size(self.axis)
            y = y / N
        return y
        

    def relprop(self,
                input:Dict[str, torch.Tensor],
                relevance:torch.Tensor,
                epsilon:float = 0.0,
                mask:Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute layer relevance for its inputs.

        Args:
            input (dict of torch.Tensor): batch of input values.
            relevance (torch.tensor): relevance of outputs
            mask (torch.Tensor, optional): mask to be applied; e.g. neighbors mask (acts like the weights).

        Returns:
            torch.Tensor : layer relevance of inputs

        """
        if not input.is_leaf:    # Store gradient if not a leaf tensor
            input.retain_grad()
            
        if input.grad is not None:  # Let it not accumulate previous gradients
            input.grad = None
        
        y = self.forward(input, mask)       
        y_explain = y + epsilon
        
        s = relevance/(y_explain + 1e-9)
        (y_explain*s.data).sum().backward()
        
        self.relevance = input*input.grad    #input.grad should be 1/N or 0 if average aggregated
        
        return self.relevance
        
        
class Collect(nn.Module):
    """Pooling layer to just collect the output.

    Args:
        axis (int): axis along which pooling is done.
        keepdim (bool, optional): whether the output tensor has dim retained or not.

    """

    def __init__(self,
                 axis:int,
                 keepdim:bool = True):
        super(Collect, self).__init__()
        self.axis = axis
        self.keepdim = keepdim
        
        self.relevance = None

    def forward(self,
                input:Dict[str, torch.Tensor],
                mask:Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Compute layer output.

        Args:
            input (torch.Tensor): input data.
            mask (torch.Tensor, optional): mask to be applied; e.g. neighbors mask.

        Returns:
            torch.Tensor: layer output.

        """
        # mask input
        if mask is not None:
            input = input * mask[..., None]
        # collect the input
        y = input
        
        return y
        

    def relprop(self,
                input:Dict[str, torch.Tensor],
                relevance:torch.Tensor,
                epsilon:float = 0.0,
                mask:Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute layer relevance for its inputs.

        Args:
            input (dict of torch.Tensor): batch of input values.
            relevance (torch.tensor): relevance of outputs
            mask (torch.Tensor, optional): mask to be applied; e.g. neighbors mask (acts like the weights).

        Returns:
            torch.Tensor : layer relevance of inputs

        """
        if not input.is_leaf:    # Store gradient if not a leaf tensor
            input.retain_grad()
            
        if input.grad is not None:  # Let it not accumulate previous gradients
            input.grad = None
        
        y = self.forward(input, mask)       
        y_explain = y + epsilon
        
        s = relevance/(y_explain + 1e-9)
        (y_explain*s.data).sum().backward()
        
        self.relevance = input*input.grad    #input.grad should be 1/N or 0 if average aggregated
        
        return self.relevance


class MaxAggregate(nn.Module):
    """Pooling layer that computes the maximum for each feature over all atoms

    Args:
        axis (int): axis along which pooling is done.
    """

    def __init__(self, axis:int):
        super().__init__()
        self.axis = axis
        self.relevance  = None

    def forward(self,
                input:Dict[str, torch.Tensor],
                mask:Optional[torch.Tensor]=None) -> torch.Tensor:
        r"""Compute layer output.

        Args:
            input (torch.Tensor): input data.
            mask (torch.Tensor, optional): mask to be applied; e.g. neighbors mask.

        Returns:
            torch.Tensor: layer output.
        """
        # mask input
        if mask is not None:
            # If the mask is lower dimensional than the array being masked,
            #  inject an extra dimension to the end
            if mask.dim() < input.dim():
                mask = torch.unsqueeze(mask, -1)
            input = torch.where(mask > 0, input, torch.min(input))

        # compute sum of input along axis
        return torch.max(input, self.axis)[0]
                
    def relprop(self,
                input:Dict[str, torch.Tensor],
                relevance:torch.Tensor,
                epsilon:float = 0.0,
                mask:Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute layer relevance for its inputs.

        Args:
            input (dict of torch.Tensor): batch of input values.
            relevance (torch.tensor): relevance of outputs
            mask (torch.Tensor, optional): mask to be applied; e.g. neighbors mask.

        Returns:
            torch.Tensor : layer relevance of outputs

        """
        if not input.is_leaf:    # Store gradient if not a leaf tensor
            input.retain_grad()
            
        if input.grad is not None:  # Let it not accumulate previous gradients
            input.grad = None
        
        y = self.forward(input, mask)       
        y_explain = y + epsilon
        
        s = relevance/(y_explain + 1e-9)
        (y_explain*s.data).sum().backward()
        
        self.relevance = input*input.grad
        
        return self.relevance


class SoftmaxAggregate(nn.Module):
    """Pooling layer that computes the maximum for each feature over all atoms
    using the "softmax" function to weigh the contribution of each atom to
    the "maximum."

    Args:
        axis (int): axis along which pooling is done.
    """

    def __init__(self, axis:int):
        super().__init__()
        self.axis = axis
        self.relevance = None

    def forward(self,
                input:Dict[str, torch.Tensor],
                mask:Optional[torch.Tensor]=None) -> torch.Tensor:
        r"""Compute layer output.

        Args:
            input (torch.Tensor): input data.
            mask (torch.Tensor, optional): mask to be applied; e.g. neighbors mask.

        Returns:
            torch.Tensor: layer output.
        """

        # Compute the sum of exponentials for the desired axis
        exp_input = torch.exp(input)

        # Set the contributions of "masked" atoms to zero
        if mask is not None:
            # If the mask is lower dimensional than the array being masked,
            #  inject an extra dimension to the end
            if mask.dim() < input.dim():
                mask = torch.unsqueeze(mask, -1)
            exp_input = torch.where(mask > 0, exp_input, torch.zeros_like(exp_input))

        # Sum exponentials along the desired axis
        exp_input_sum = torch.sum(exp_input, self.axis, keepdim=True)

        # Normalize the exponential array by the
        weights = exp_input / exp_input_sum

        # compute sum of input along axis
        output = torch.sum(input * weights, self.axis)
        return output


    def relprop(self,
                input:Dict[str, torch.Tensor],
                relevance:torch.Tensor,
                epsilon:float = 0.0,
                mask:Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute layer relevance for its inputs.

        Args:
            input (dict of torch.Tensor): batch of input values.
            relevance (torch.tensor): relevance of outputs

        Returns:
            torch.Tensor : layer relevance of outputs

        """
        
        raise NotImplementedError