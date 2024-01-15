from torch import nn as nn
import torch

import schnetpack as spk
from schnetpack import Properties
from typing import Optional, Union, List, Dict

__all__ = ["AtomisticModel"]


class ModelError(Exception):
    pass


class AtomisticModel(nn.Module):
    """
    Join a representation model with output modules.

    Args:
        representation (torch.nn.Module): Representation block of the model.
        output_modules (list or nn.ModuleList or spk.output_modules.Atomwise): Output
            block of the model. Needed for predicting properties.
        analytical_modules (list or nn.ModuleList or spk.output_modules.Atomwise): Anaytical
            module which passes thourhg an analytical function on the inputs and results to finally get a new result

    Returns:
         dict: property predictions
    """

    def __init__(self,
                 representation:nn.Module,
                 output_modules:Union[List[nn.Module], nn.ModuleList, nn.Module],
                 analytical_modules:Optional[Union[List[nn.Module], nn.ModuleList, nn.Module]]=None):
        super(AtomisticModel, self).__init__()
        self.representation = representation
        if type(output_modules) not in [list, nn.ModuleList]:
            output_modules = [output_modules]
        if type(output_modules) == list:
            output_modules = nn.ModuleList(output_modules)
        self.output_modules = output_modules

        if(analytical_modules is None):
            self.analytical_modules = analytical_modules
        else:
            if type(analytical_modules) not in [list, nn.ModuleList]:
                analytical_modules = [analytical_modules]
            if type(analytical_modules) == list:
                analytical_modules = nn.ModuleList(analytical_modules)
            self.analytical_modules = analytical_modules
        # For gradients
        self.requires_dr = any([om.derivative for om in self.output_modules])
        # For stress tensor
        self.requires_stress = any([om.stress for om in self.output_modules])

        self.representation_relevance = []
        self.output_modules_relevance = []
        
    def forward(self, inputs:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward representation output through output modules.
        """
        if self.requires_dr:
            inputs[Properties.R].requires_grad_()
        if self.requires_stress:
            # Check if cell is present
            if inputs[Properties.cell] is None:
                raise ModelError("No cell found for stress computation.")

            # Generate Cartesian displacement tensor
            displacement = torch.zeros_like(inputs[Properties.cell]).to(
                inputs[Properties.R].device
            )
            displacement.requires_grad = True
            inputs["displacement"] = displacement

            # Apply to coordinates and cell
            inputs[Properties.R] = inputs[Properties.R] + torch.matmul(
                inputs[Properties.R], displacement
            )
            inputs[Properties.cell] = inputs[Properties.cell] + torch.matmul(
                inputs[Properties.cell], displacement
            )

        inputs["representation"] = self.representation(inputs)
        outs = {}
        for output_model in self.output_modules:
            outs.update(output_model(inputs))
        
        # Checking if the model has analytical modules attribute
        # This is done to be back compatible where model was saved without this attribute

        if hasattr(self, 'analytical_modules'):
            # Passing through the analytical layers
            if(self.analytical_modules is not None):   
                for analytical_model in self.analytical_modules:
                    outs.update(analytical_model(inputs, outs))
        return outs
        
    def explain_model(self, inputs:Dict[str, torch.Tensor]) -> None:
    
        if self.requires_dr:
            inputs[Properties.R].requires_grad_()
        if self.requires_stress:
            # Check if cell is present
            if inputs[Properties.cell] is None:
                raise ModelError("No cell found for stress computation.")

            # Generate Cartesian displacement tensor
            displacement = torch.zeros_like(inputs[Properties.cell]).to(
                inputs[Properties.R].device
            )
            displacement.requires_grad = True
            inputs["displacement"] = displacement

            # Apply to coordinates and cell
            inputs[Properties.R] = inputs[Properties.R] + torch.matmul(
                inputs[Properties.R], displacement
            )
            inputs[Properties.cell] = inputs[Properties.cell] + torch.matmul(
                inputs[Properties.cell], displacement
            )

        inputs["representation"] = self.representation(inputs)
        outs = {}
        for output_model in self.output_modules:
            outs.update(output_model(inputs))
            
        for output_model in self.output_modules: 
            out_relevance = output_model.relprop(inputs, 1.0)
            rep_relevance = self.representation.relprop(inputs, out_relevance)
            
            self.output_modules_relevance.append(out_relevance)
            self.representation_relevance.append(rep_relevance)
            
        
        
        
