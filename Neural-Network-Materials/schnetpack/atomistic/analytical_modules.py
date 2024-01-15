import torch
from torch import nn as nn
from typing import Tuple, List, Callable, Dict

__all__ = [
    "calc_injection_velocity",
    "AnalyticalLayer",
]

def calc_injection_velocity(Jx:torch.Tensor,
                            dos:torch.Tensor,
                            energies:torch.Tensor,
                            Ef:torch.Tensor,
                            kbT:float = 26 * 1e-3) -> Tuple[torch.Tensor]:
    '''
    Calculating injection velocity form Jx(E) and DOS(E) integration and average    
    Jx = vx X DOS

    Params
    ------
    
    Jx (np.array of size num_batch X num_E): The current density along the x,y,z direction as a function of energy
    dos (np.array of size num_batch X num_E): The density of states as a function of energy
    energies (np.array of size num_batch X num_E): The value of energies at which dos and v is calculated (in eV)
    Ef (num_batch X 1): The source fermi level relative the top of the valence band (at zero energy level)
    kbT (float): The value of kb * T value in eV
    
    Returns
    -------
    
    vinj (np.array of size num_batch X 1): The injection velocity along x,y and z direction
    N1 (np.array of size num_batch X 1): The number of electrons getting injected into the system 
    '''
    epsilon = 1e-12    
    f = 1 / (1 + torch.exp((energies - Ef)/kbT))       # size num_batch X num_E
    vinj = torch.trapz(Jx * f, energies, dim = 1)[:, None]       # size num_batch X 1
    N1 = 1/2 * torch.trapz(dos* f, energies, dim = 1)[:, None]    # size num_batch X 1 
    vinj = vinj/(N1 + epsilon)
    
    return vinj, N1   


class AnalyticalLayer(nn.Module):
    """ This layer is just an analytical function. [out] = f([in])
    Args:
        arg_strings(list): The list of argument strings. These are the arguments of the function
                             Must be in the same order as the arguments of the function
        arg_type(list): Whether the argument is from the inputs dictionary or the results dictionary
        output_strings(list): The list of output strings. Would be appended to the results dictionary
        function(pointer to the function): The analytical function which needs to be evaluated 
    """

    def __init__(self,
                arg_strings:List[str],
                arg_type:List[str],
                output_strings:List[str],
                function:Callable):
        super().__init__()
        self.arg_strings = arg_strings
        self.arg_type = arg_type
        self.output_strings = output_strings
        self.function = function

    def forward(self,
                inputs:Dict[str, torch.Tensor],
                results:torch.Tensor) -> Dict[str, torch.Tensor]:
        r"""Compute the analytical layer function.
        
        The arguments should be of size nbatch X (ndim of arguments)
        The result should be of size nbatch X (ndim of output)
        """

        args = []   # The list for function arguments
        
        for ii, arg_string in enumerate(self.arg_strings):
            if(self.arg_type[ii] == 'inputs'):
                args.append(inputs[arg_string])
            elif(self.arg_type[ii] == 'results'):
                args.append(results[arg_string])
            else:
                print('Please provide only "inputs" or "results"')
                raise TypeError

        outputs = self.function(*args)
        function_results = {}
        for ii, output_string in enumerate(self.output_strings):
            function_results.update({output_string: outputs[ii]})
         
        return function_results 
