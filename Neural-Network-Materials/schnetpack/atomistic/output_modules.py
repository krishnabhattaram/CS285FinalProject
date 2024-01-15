import numpy as np
import torch
from torch import nn as nn
from torch.autograd import grad

import schnetpack
from schnetpack import nn as L, Properties
from typing import Optional, Dict, Callable, Union, List

__all__ = [
    "Atomwise",
    "ElementalAtomwise",
    "DipoleMoment",
    "ElementalDipoleMoment",
    "Polarizability",
    "ElectronicSpatialExtent",
    "EnergyBandStructure",
    "DensityOfStates",
    "BondDensityOfStates",
    "LocalDensityOfStates",
    "Velocity",
    "Rotate_Vectorized",
]


class AtomwiseError(Exception):
    pass


class Atomwise(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the
    energy.

    Args:
        n_in (int): input dimension of representation
        n_out (int): output dimension of target property (default: 1)
        aggregation_mode (str): one of {sum, avg} (default: sum)
        n_layers (int): number of nn in output network (default: 2)
        n_neurons (list of int or None): number of neurons in each layer of the output
            network. If `None`, divide neurons by 2 in each layer. (default: None)
        activation (function): activation function for hidden nn
            (default: spk.nn.activations.shifted_softplus)
        property (str): name of the output property (default: "y")
        contributions (str or None): Name of property contributions in return dict.
            No contributions returned if None. (default: None)
        derivative_arg (str or None): Name of property for which derivate needs to be calculated
        derivative (str or None): Name of property derivative. No derivative
            returned if None. (default: None)
        negative_dr (bool): Multiply the derivative with -1 if True. (default: False)
        derivative_multiplier (float): Multiply the derivative by a multiplier. Used if use energyperatom as the property. (default: 1.0)
        stress (str or None): Name of stress property. Compute the derivative with
            respect to the cell parameters if not None. (default: None)
        create_graph (bool): If False, the graph used to compute the grad will be
            freed. Note that in nearly all cases setting this option to True is not
            needed and often can be worked around in a much more efficient way.
            Defaults to the value of create_graph. (default: False)
        mean (torch.Tensor or None): mean of property
        stddev (torch.Tensor or None): standard deviation of property (default: None)
        atomref (torch.Tensor or None): reference single-atom properties. Expects
            an (max_z + 1) x 1 array where atomref[Z] corresponds to the reference
            property of element Z. The value of atomref[0] must be zero, as this
            corresponds to the reference property for for "mask" atoms. (default: None)
        outnet (callable): Network used for atomistic outputs. Takes schnetpack input
            dictionary as input. Output is not normalized. If set to None,
            a pyramidal network is generated automatically. (default: None)

    Returns:
        tuple: prediction for property

        If contributions is not None additionally returns atom-wise contributions.

        If derivative is not None additionally returns derivative w.r.t. atom positions.

    """

    def __init__(
        self,
        n_in:int,
        n_out:int=1,
        aggregation_mode:str="sum",
        n_layers:int=2,
        n_neurons:Optional[int]=None,
        activation:Callable=schnetpack.nn.activations.shifted_softplus,
        property:str="y",
        contributions:Optional[str]=None,
        derivative_arg:Optional[str]=None,
        derivative:Optional[str]=None,
        derivative_multiplier:float=1.0,
        negative_dr:bool=False,
        stress:Optional[str]=None,
        create_graph:bool=False,
        mean:Optional[torch.Tensor]=None,
        stddev:Optional[torch.Tensor]=None,
        atomref:Optional[torch.Tensor]=None,
        outnet:Optional[Union[nn.Module, nn.Sequential]]=None,
    ):
        super(Atomwise, self).__init__()

        self.n_layers = n_layers
        self.create_graph = create_graph
        self.property = property
        self.contributions = contributions
        self.derivative_arg = derivative_arg
        self.derivative = derivative
        self.derivative_multiplier = derivative_multiplier
        self.negative_dr = negative_dr
        self.stress = stress

        mean = torch.FloatTensor([0.0]) if mean is None else mean
        stddev = torch.FloatTensor([1.0]) if stddev is None else stddev

        # initialize single atom energies
        if atomref is not None:
            self.atomref = nn.Embedding.from_pretrained(
                torch.from_numpy(atomref.astype(np.float32))
            )
        else:
            self.atomref = None

        # build output network
        if outnet is None:
            self.out_net = nn.Sequential(
                schnetpack.nn.base.GetItem("representation"),
                schnetpack.nn.blocks.MLP(n_in, n_out, n_neurons, n_layers, activation),
            )
        else:
            self.out_net = outnet

        # build standardization layer
        self.standardize = schnetpack.nn.base.ScaleShift(mean, stddev)

        # build aggregation layer
        if aggregation_mode == "sum":
            self.atom_pool = schnetpack.nn.base.Aggregate(axis=1, mean=False)
        elif aggregation_mode == "avg":
            self.atom_pool = schnetpack.nn.base.Aggregate(axis=1, mean=True)
        elif aggregation_mode == "avg_sum":    # This mode calculates both avg and sum. avg is the main output and sum is used for forces
            self.atom_pool = schnetpack.nn.base.Aggregate(axis=1, mean=True)
            self.atom_pool_sub = schnetpack.nn.base.Aggregate(axis=1, mean=False)          
        elif aggregation_mode == "max":
            self.atom_pool = schnetpack.nn.base.MaxAggregate(axis=1)
        elif aggregation_mode == "softmax":
            self.atom_pool = schnetpack.nn.base.SoftmaxAggregate(axis=1)
        elif aggregation_mode == "collect":
            self.atom_pool = schnetpack.nn.base.Collect(axis=1)
        else:
            raise AtomwiseError(
                "{} is not a valid aggregation " "mode!".format(aggregation_mode)
            )
            
        self.aggregation_mode = aggregation_mode

    def forward(self, inputs:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        r"""
        predicts atomwise property
        """
        atomic_numbers = inputs[Properties.Z]
        atom_mask = inputs[Properties.atom_mask]

        # run prediction
        yi = self.out_net(inputs)
        yi = self.standardize(yi)

        if self.atomref is not None:
            y0 = self.atomref(atomic_numbers)
            yi = yi + y0

        y = self.atom_pool(yi, atom_mask)
        # collect results
        result = {self.property: y}
        
        if(self.aggregation_mode  == "avg_sum"):
            ytotal = self.atom_pool_sub(yi, atom_mask)
            result[self.derivative_arg] = ytotal
        else:
            result[self.derivative_arg] = y
        
        if self.contributions is not None:
            result[self.contributions] = yi

        create_graph = True if self.training else self.create_graph

        if self.derivative is not None:
            sign = -1.0 if self.negative_dr else 1.0
            dy = grad(
                result[self.derivative_arg],
                inputs[Properties.R],
                grad_outputs=torch.ones_like(result[self.property]),
                create_graph=create_graph,
                retain_graph=True,
            )[0]
            result[self.derivative] = self.derivative_multiplier * sign * dy

        if self.stress is not None:
            cell = inputs[Properties.cell]
            # Compute derivative with respect to cell displacements
            stress = grad(
                result[self.property],
                inputs["displacement"],
                grad_outputs=torch.ones_like(result[self.property]),
                create_graph=create_graph,
                retain_graph=True,
            )[0]
            # Compute cell volume
            volume = torch.sum(
                cell[:, 0, :] * torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
                dim=1,
                keepdim=True,
            )[..., None]
            # Finalize stress tensor
            result[self.stress] = stress / volume

        return result


    def relprop(self,
                inputs:Dict[str, torch.Tensor],
                relevance:torch.Tensor,
                lrp_fn:Optional[Callable]=None,
                epsilon:float = 0.0) -> torch.Tensor:
        r"""
        predicts atomwise property
        """
        atomic_numbers = inputs[Properties.Z]
        atom_mask = inputs[Properties.atom_mask]

        # run prediction
        inputs_list = [inputs]
        for ii, nn in enumerate(self.out_net):
            inputs_list.append(nn(inputs_list[ii]))
                  
        yi = inputs_list[-1]
        yi_std = self.standardize(yi)

        if self.atomref is not None:
            y0 = self.atomref(atomic_numbers)
            yi_std = yi_std + y0

        y = self.atom_pool(yi_std, atom_mask)

        # collect results
        result = {self.property: y}

        if self.contributions is not None:
            result[self.contributions] = yi

        create_graph = True if self.training else self.create_graph

        if self.derivative is not None:
            sign = -1.0 if self.negative_dr else 1.0
            dy = grad(
                result[self.property],
                inputs[Properties.R],
                grad_outputs=torch.ones_like(result[self.property]),
                create_graph=create_graph,
                retain_graph=True,
            )[0]
            result[self.derivative] = self.derivative_multiplier * sign * dy

        if self.stress is not None:
            cell = inputs[Properties.cell]
            # Compute derivative with respect to cell displacements
            stress = grad(
                result[self.property],
                inputs["displacement"],
                grad_outputs=torch.ones_like(result[self.property]),
                create_graph=create_graph,
                retain_graph=True,
            )[0]
            # Compute cell volume
            volume = torch.sum(
                cell[:, 0, :] * torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
                dim=1,
                keepdim=True,
            )[..., None]
            # Finalize stress tensor
            result[self.stress] = stress / volume
            
        yi_std_relevance = self.atom_pool.relprop(yi_std, relevance, mask = atom_mask)
        yi_relevance = self.standardize.relprop(yi, yi_std_relevance)
        
        inputs_list = reversed(inputs_list)
        out_relevance = relevance
        for ii, nn in enumerate(reversed(list(self.out_net))):
            out_relevance = nn.relprop(inputs_list[ii + 1], out_relevance)
         
        return out_relevance  
        
            
        
        
        

class DipoleMoment(Atomwise):
    """
    Predicts latent partial charges and calculates dipole moment.

    Args:
        n_in (int): input dimension of representation
        n_layers (int): number of layers in output network (default: 2)
        n_neurons (list of int or None): number of neurons in each layer of the output
            network. If `None`, divide neurons by 2 in each layer. (default: None)
        activation (torch.Function): activation function for hidden nn
            (default: schnetpack.nn.activations.shifted_softplus)
        property (str): name of the output property (default: "y")
        contributions (str or None): Name of property contributions in return dict.
            No contributions returned if None. (default: None)
        charge_correction (str or None): Name of charge labels in dataset. If
            something is selected, the charge contributions are corrected according
            to the total charges in the dataset. No charge correction if None.
            (default: None)
        predict_magnitude (bool): if True, predict the magnitude of the dipole moment
            instead of the vector (default: False)
        mean (torch.FloatTensor or None): mean of dipole (default: None)
        stddev (torch.FloatTensor or None): stddev of dipole (default: None)

    Returns:
        dict: vector for the dipole moment

        If predict_magnitude is True returns the magnitude of the dipole moment
        instead of the vector.

        If contributions is not None latent atomic charges are added to the output
        dictionary.
    """

    def __init__(
        self,
        n_in:int,
        n_layers:int=2,
        n_neurons:Optional[Union[int, List[int]]]=None,
        activation:Callable=schnetpack.nn.activations.shifted_softplus,
        property:str="y",
        contributions:Optional[str]=None,
        charge_correction:Optional[str]=None,
        predict_magnitude:bool=False,
        mean:Optional[torch.Tensor]=None,
        stddev:Optional[torch.Tensor]=None,
        outnet:Optional[Union[nn.Module, nn.Sequential]]=None,
    ):
        self.predict_magnitude = predict_magnitude
        self.charge_correction = charge_correction
        super(DipoleMoment, self).__init__(
            n_in,
            1,
            "sum",
            n_layers,
            n_neurons,
            activation=activation,
            mean=mean,
            stddev=stddev,
            outnet=outnet,
            property=property,
            contributions=contributions,
        )

    def forward(self, inputs):
        """
        predicts dipole moment
        """
        positions = inputs[Properties.R]
        atom_mask = inputs[Properties.atom_mask]

        # run prediction
        charges = self.out_net(inputs) * atom_mask[:, :, None]

        # charge correction
        if self.charge_correction is not None:
            total_charges = inputs[self.charge_correction]
            charge_correction = total_charges - charges.sum(1)
            charges = charges + (
                charge_correction / atom_mask.sum(-1).unsqueeze(-1)
            ).unsqueeze(-1)
            charges *= atom_mask[:, :, None]

        yi = positions * charges
        y = self.atom_pool(yi)

        if self.predict_magnitude:
            y = torch.norm(y, dim=1, keepdim=True)

        # collect results
        result = {self.property: y}

        if self.contributions:
            result[self.contributions] = charges

        return result


class ElementalAtomwise(Atomwise):
    """
    Predicts properties in atom-wise fashion using a separate network for every chemical
    element of the central atom. Particularly useful for networks of
    Behler-Parrinello type.

    Args:
        n_in (int): input dimension of representation (default: 128)
        n_out (int): output dimension of target property (default: 1)
        aggregation_mode (str): one of {sum, avg} (default: sum)
        n_layers (int): number of nn in output network (default: 3)
        property (str): name of the output property (default: "y")
        derivative (str or None): Name of property derivative. No derivative
            returned if None. (default: None)
        negative_dr (bool): Multiply the derivative with -1 if True. (default: False)
        contributions (str or None): Name of property contributions in return dict.
            No contributions returned if None. (default: None)
        create_graph (bool): If False, the graph used to compute the grad will be
            freed. Note that in nearly all cases setting this option to True is not nee
            ded and often can be worked around in a much more efficient way. Defaults to
            the value of create_graph. (default: False)
        elements (set of int): List of atomic numbers present in the training set
            {1,6,7,8,9} for QM9. (default: frozenset(1,6,7,8,9))
        n_hidden (int): number of neurons in each layer of the output network.
            (default: 50)
        activation (function): activation function for hidden nn
            (default: spk.nn.activations.shifted_softplus)
        mean (torch.Tensor or None): mean of property
        stddev (torch.Tensor or None): standard deviation of property (default: None)
        atomref (torch.Tensor or None): reference single-atom properties. Expects
            an (max_z + 1) x 1 array where atomref[Z] corresponds to the reference
            property of element Z. The value of atomref[0] must be zero, as this
            corresponds to the reference property for for "mask" atoms. (default: None)
    """

    def __init__(
        self,
        n_in:int=128,
        n_out:int=1,
        aggregation_mode:str="sum",
        n_layers:int=3,
        property:str="y",
        derivative:Optional[str]=None,
        negative_dr:bool=False,
        contributions:Optional[str]=None,
        create_graph:bool=True,
        elements=frozenset((1, 6, 7, 8, 9)),
        n_hidden:int=50,
        activation:Callable=schnetpack.nn.activations.shifted_softplus,
        mean:Optional[torch.Tensor]=None,
        stddev:Optional[torch.Tensor]=None,
        atomref:Optional[torch.Tensor]=None,
    ):
        outnet = schnetpack.nn.blocks.GatedNetwork(
            n_in,
            n_out,
            elements,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )

        super(ElementalAtomwise, self).__init__(
            n_in=n_in,
            n_out=n_out,
            aggregation_mode=aggregation_mode,
            n_layers=n_layers,
            n_neurons=None,
            activation=activation,
            property=property,
            contributions=contributions,
            derivative=derivative,
            negative_dr=negative_dr,
            create_graph=create_graph,
            mean=mean,
            stddev=stddev,
            atomref=atomref,
            outnet=outnet,
        )


class ElementalDipoleMoment(DipoleMoment):
    """
    Predicts partial charges and computes dipole moment using serparate NNs for every different element.
    Particularly useful for networks of Behler-Parrinello type.

    Args:
        n_in (int): input dimension of representation
        n_out (int): output dimension of representation (default: 1)
        n_layers (int): number of layers in output network (default: 3)
        predict_magnitude (bool): if True, predict the magnitude of the dipole moment instead of the vector (default: False)
        elements (set of int): List of atomic numbers present in the training set {1,6,7,8,9} for QM9. (default: frozenset(1,6,7,8,9))
        n_hidden (int): number of neurons in each layer of the output network. (default: 50)
        activation (function): activation function for hidden nn (default: schnetpack.nn.activations.shifted_softplus)
        activation (function): activation function for hidden nn
        mean (torch.FloatTensor): mean of energy
        stddev (torch.FloatTensor): standard deviation of energy
    """

    def __init__(
        self,
        n_in:int,
        n_out:int=1,
        n_layers:int=3,
        contributions:bool=False,
        property:str="y",
        predict_magnitude:bool=False,
        elements=frozenset((1, 6, 7, 8, 9)),
        n_hidden:int=50,
        activation:Callable=schnetpack.nn.activations.shifted_softplus,
        mean:Optional[torch.Tensor]=None,
        stddev:Optional[torch.Tensor]=None,
    ):
        outnet = schnetpack.nn.blocks.GatedNetwork(
            n_in,
            n_out,
            elements,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )

        super(ElementalDipoleMoment, self).__init__(
            n_in,
            n_layers,
            None,
            activation=activation,
            property=property,
            contributions=contributions,
            outnet=outnet,
            predict_magnitude=predict_magnitude,
            mean=mean,
            stddev=stddev,
        )


class Polarizability(Atomwise):
    """
    Predicts polarizability of input molecules.

    Args:
        n_in (int): input dimension of representation (default: 128)
        aggregation_mode (str): one of {sum, avg} (default: sum)
        n_layers (int): number of nn in output network (default: 2)
        n_neurons (list of int or None): number of neurons in each layer of the output
            network. If `None`, divide neurons by 2 in each layer. (default: None)
        activation (function): activation function for hidden nn
            (default: spk.nn.activations.shifted_softplus)
        property (str): name of the output property (default: "y")
        isotropic (bool): return isotropic polarizability if True. (default: False)
        create_graph (bool): If False, the graph used to compute the grad will be
            freed. Note that in nearly all cases setting this option to True is not nee
            ded and often can be worked around in a much more efficient way. Defaults to
            the value of create_graph. (default: False)
        outnet (callable): Network used for atomistic outputs. Takes schnetpack input
            dictionary as input. Output is not normalized. If set to None,
            a pyramidal network is generated automatically. (default: None)
        cutoff_network (nn.Module): cutoff network (default: None)

    Returns:
        dict: Polarizability of molecules

        Adds isotropic polarizability if isotropic is not None.
    """

    def __init__(
        self,
        n_in:int=128,
        aggregation_mode:str="sum",
        n_layers:int=2,
        n_neurons:Optional[Union[int, List[int]]]=None,
        activation:Callable=L.shifted_softplus,
        property:str="y",
        isotropic:bool=False,
        create_graph:bool=True,
        outnet:Optional[Union[nn.Module, nn.Sequential]]=None,
        cutoff_network:nn.Module=None,
    ):
        super(Polarizability, self).__init__(
            n_in=n_in,
            n_out=2,
            n_layers=n_layers,
            aggregation_mode=aggregation_mode,
            n_neurons=n_neurons,
            activation=activation,
            property=property,
            derivative=None,
            create_graph=create_graph,
            outnet=outnet,
        )
        self.isotropic = isotropic
        self.nbh_agg = L.Aggregate(2)
        self.atom_agg = L.Aggregate(1)

        self.cutoff_network = cutoff_network

    def forward(self, inputs:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        positions = inputs[Properties.R]
        neighbors = inputs[Properties.neighbors]
        nbh_mask = inputs[Properties.neighbor_mask]
        atom_mask = inputs[Properties.atom_mask]

        # Get environment distances and positions
        distances, dist_vecs = L.atom_distances(positions, neighbors, return_vecs=True)

        # Get atomic contributions
        contributions = self.out_net(inputs)

        # Redistribute contributions to neighbors
        # B x A x N x 1
        # neighbor_contributions = L.neighbor_elements(c1, neighbors)
        neighbor_contributions = L.neighbor_elements(contributions, neighbors)

        if self.cutoff_network is not None:
            f_cut = self.cutoff_network(distances)[..., None]
            neighbor_contributions = neighbor_contributions * f_cut

        neighbor_contributions1 = neighbor_contributions[:, :, :, 0]
        neighbor_contributions2 = neighbor_contributions[:, :, :, 1]

        # B x A x N x 3
        atomic_dipoles = self.nbh_agg(
            dist_vecs * neighbor_contributions1[..., None], nbh_mask
        )
        # B x A x N x 3

        masked_dist = (distances ** 3 * nbh_mask) + (1 - nbh_mask)
        nbh_fields = (
            dist_vecs * neighbor_contributions2[..., None] / masked_dist[..., None]
        )
        atomic_fields = self.nbh_agg(nbh_fields, nbh_mask)
        field_norm = torch.norm(atomic_fields, dim=-1, keepdim=True)
        field_norm = field_norm + (field_norm < 1e-10).float()
        atomic_fields = atomic_fields / field_norm

        atomic_polar = atomic_dipoles[..., None] * atomic_fields[:, :, None, :]

        # Symmetrize
        atomic_polar = symmetric_product(atomic_polar)

        global_polar = self.atom_agg(atomic_polar, atom_mask[..., None])

        result = {
            # "y_i": atomic_polar,
            self.property: global_polar
        }

        if self.isotropic:
            result[self.property] = torch.mean(
                torch.diagonal(global_polar, dim1=-2, dim2=-1), dim=-1, keepdim=True
            )
        return result


def symmetric_product(tensor: torch.Tensor) -> torch.Tensor:
    """
    Symmetric outer product of tensor
    """
    shape = tensor.size()
    idx = list(range(len(shape)))
    idx[-1], idx[-2] = idx[-2], idx[-1]
    return 0.5 * (tensor + tensor.permute(*idx))


class ElectronicSpatialExtent(Atomwise):
    """
    Predicts the electronic spatial extent using a formalism close to the dipole moment layer.
    The electronic spatial extent is discretized as a sum of atomic latent contributions
    weighted by the squared distance of the atom from the center of mass (SchNetPack default).

    .. math:: ESE = \sum_i^N | R_{i0} |^2 q(R)

    Args:
        n_in (int): input dimension of representation
        n_layers (int): number of layers in output network (default: 2)
        n_neurons (list of int or None): number of neurons in each layer of the output
            network. If `None`, divide neurons by 2 in each layer. (default: None)
        activation (torch.Function): activation function for hidden nn
            (default: schnetpack.nn.activations.shifted_softplus)
        property (str): name of the output property (default: "y")
        contributions (str or None): Name of property contributions in return dict.
            No contributions returned if None. (default: None)
        mean (torch.FloatTensor or None): mean of dipole (default: None)
        stddev (torch.FloatTensor or None): stddev of dipole (default: None)

    Returns:
        dict: the electronic spatial extent

        If contributions is not None latent atomic charges are added to the output
        dictionary.
    """

    def __init__(
        self,
        n_in:int,
        n_layers:int=2,
        n_neurons:Optional[int]=None,
        activation:Callable=schnetpack.nn.activations.shifted_softplus,
        property:str="y",
        contributions:Optional[str]=None,
        mean:Optional[torch.Tensor]=None,
        stddev:Optional[torch.Tensor]=None,
        outnet:Optional[Union[nn.Module, nn.Sequential]]=None,
    ):
        super(ElectronicSpatialExtent, self).__init__(
            n_in,
            1,
            "sum",
            n_layers,
            n_neurons,
            activation=activation,
            mean=mean,
            stddev=stddev,
            outnet=outnet,
            property=property,
            contributions=contributions,
        )

    def forward(self, inputs:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Predicts the electronic spatial extent.
        """
        positions = inputs[Properties.R]
        atom_mask = inputs[Properties.atom_mask][:, :, None]

        # run prediction
        charges = self.out_net(inputs) * atom_mask
        yi = torch.norm(positions, 2, 2, keepdim=True) ** 2 * charges
        y = self.atom_pool(yi)

        # collect results
        result = {self.property: y}

        if self.contributions:
            result[self.contributions] = charges

        return result
        
class EnergyBandStructure(Atomwise):
  
  
  def __init__(
      self,
      n_in:int,
      n_out:int=50,
      aggregation_mode:str="avg",
      n_layers:int=2,
      n_neurons:Optional[Union[int, List[int]]]=None,
      activation:Callable=schnetpack.nn.activations.shifted_softplus,
      property:str="bandcoeffimp",
      mean:Optional[torch.Tensor]=None,
      stddev:Optional[torch.Tensor]=None,
      outnet:Optional[Union[nn.Module, nn.Sequential]]=None, 
      atomref:Optional[torch.Tensor]=None,):
      
      self.bandstructcoeff = np.array([])
      super(EnergyBandStructure, self).__init__(
            n_in,
            n_out,
            aggregation_mode,
            n_layers,
            n_neurons,
            activation=activation,
            mean=mean,
            stddev=stddev,
            outnet=outnet,
            property=property,
        )


  def forward(self, inputs:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
      r"""
      predicts energy band structure property
      """
      atomic_numbers = inputs[Properties.Z]
      atom_mask = inputs[Properties.atom_mask]

      # run prediction
      yi = self.out_net(inputs)
      yi = self.standardize(yi)

      if self.atomref is not None:
          y0 = self.atomref(atomic_numbers)
          yi = yi + y0
      
      y = self.atom_pool(yi, atom_mask)

      # collect results
      result = {self.property: y}


      return result
      
      
class DensityOfStates(Atomwise):
  
  
  def __init__(
      self,
      n_in:int,
      n_out:int=50,
      aggregation_mode:str="avg",
      n_layers:int=2,
      n_neurons:Optional[Union[int, List[int]]]=None,
      activation:Callable=schnetpack.nn.activations.shifted_softplus,
      property:str="threshold_pca_dosperatom",
      property_rotated:str = "smooth_dosperatom",
      remove_elements:List[int] = [],
      mean:Optional[torch.Tensor]=None,
      stddev:Optional[torch.Tensor]=None,
      outnet:Optional[Union[nn.Module, nn.Sequential]]=None, 
      atomref:Optional[torch.Tensor]=None,
      pca_basis:Optional[torch.Tensor] = None,
      std_dos = 1.0,
      mean_dos = 0.0):
      
            
      super(DensityOfStates, self).__init__(
            n_in,
            n_out,
            aggregation_mode,
            n_layers,
            n_neurons,
            activation=activation,
            mean=mean,
            stddev=stddev,
            outnet=outnet,
            property=property,
        )
      self.property = property
      self.property_rotated = property_rotated
      self.remove_elements = remove_elements
      
      if(self.property_rotated is not None):
          self.outnet_rotate =  schnetpack.nn.base.RotateScaleShift(pca_basis.T, mean_dos, std_dos)


  def forward(self, inputs:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
      r"""
      predicts density of states in pca basis property
      """
      atomic_numbers = inputs[Properties.Z]
      atom_mask = inputs[Properties.atom_mask]

      # run prediction
      yi = self.out_net(inputs)
      yi = self.standardize(yi)

      if self.atomref is not None:
          y0 = self.atomref(atomic_numbers)
          yi = yi + y0
      
      
      # Removing some input elements 
      remove_element_mask = torch.ones_like(atom_mask)
      if(len(self.remove_elements) > 0):
          for elem in self.remove_elements:
              remove_element_mask *= (~ (atomic_numbers == elem)) 
      
      y = self.atom_pool(yi, atom_mask * remove_element_mask)    # Removing the false atoms which was introduced for making equal size
      
      # Transforming from pca basis to energy basis
      if(self.property_rotated is not None):
          y_rotated = self.outnet_rotate(y)
                
          # collect results
          result = {self.property: y, self.property_rotated: y_rotated}
      
      else:               
          # collect results
          result = {self.property: y}
          

      return result
      
      
class BondDensityOfStates(Atomwise):
  
  
  def __init__(
      self,
      n_in:int,
      n_out:int=50,
      aggregation_mode:str="avg",
      n_layers:int=2,
      n_neurons:Optional[Union[int, List[int]]]=None,
      activation:Callable=schnetpack.nn.activations.shifted_softplus,
      property:str="threshold_pca_dosperatom",
      property_rotated:str = "smooth_dosperatom",
      remove_elements:List[int] = [],
      mean:Optional[torch.Tensor]=None,
      stddev:Optional[torch.Tensor]=None,
      outnet:Optional[Union[nn.Module, nn.Sequential]]=None, 
      atomref:Optional[torch.Tensor]=None,
      pca_basis:Optional[torch.Tensor] = None,
      std_dos = 1.0,
      mean_dos = 0.0):
      
            
      super(BondDensityOfStates, self).__init__(
            n_in,
            n_out,
            aggregation_mode,
            n_layers,
            n_neurons,
            activation=activation,
            mean=mean,
            stddev=stddev,
            outnet=outnet,
            property=property,
        )
        
        
      # build aggregation layer
      if aggregation_mode == "sum":
          self.atom_pool = schnetpack.nn.base.Aggregate(axis=(1,2), mean=False)
      elif aggregation_mode == "avg":
          self.atom_pool = schnetpack.nn.base.Aggregate(axis=(1,2), mean=True)
      elif aggregation_mode == "avg_sum":    # This mode calculates both avg and sum. avg is the main output and sum is used for forces
          self.atom_pool = schnetpack.nn.base.Aggregate(axis=(1,2), mean=True)
          self.atom_pool_sub = schnetpack.nn.base.Aggregate(axis=(1,2), mean=False)          
      elif aggregation_mode == "max":
          self.atom_pool = schnetpack.nn.base.MaxAggregate(axis=(1,2))
      elif aggregation_mode == "softmax":
          self.atom_pool = schnetpack.nn.base.SoftmaxAggregate(axis=(1,2))
      elif aggregation_mode == "collect":
          self.atom_pool = schnetpack.nn.base.Collect(axis=(1,2))
      else:
          raise AtomwiseError(
              "{} is not a valid aggregation " "mode!".format(aggregation_mode)
          )
            
      self.property = property
      self.property_rotated = property_rotated
      self.remove_elements = remove_elements
      
      if(self.property_rotated is not None):
          self.outnet_rotate =  schnetpack.nn.base.RotateScaleShift(pca_basis.T, mean_dos, std_dos)


  def forward(self, inputs:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
      r"""
      predicts density of states in pca basis property
      """
      atomic_numbers = inputs[Properties.Z]
      atom_mask = inputs[Properties.atom_mask]
      neighbor_mask = inputs[Properties.neighbor_mask]

      # run prediction
      yi = self.out_net(inputs)
      yi = self.standardize(yi)
      
      '''
      if self.atomref is not None:
          y0 = self.atomref(atomic_numbers)
          yi = yi + y0
      '''
      
      # Removing some input elements 
      remove_element_mask = torch.ones_like(atom_mask)
      if(len(self.remove_elements) > 0):
          for elem in self.remove_elements:
              remove_element_mask *= (~ (atomic_numbers == elem)) 
      
      mask = atom_mask[:, :, None] * remove_element_mask[:, :, None] * neighbor_mask
      y = self.atom_pool(yi, mask)    # Removing the false atoms which was introduced for making equal size
      
      # Transforming from pca basis to energy basis
      if(self.property_rotated is not None):
          y_rotated = self.outnet_rotate(y)
                
          # collect results
          result = {self.property: y, self.property_rotated: y_rotated}
      
      else:               
          # collect results
          result = {self.property: y}
          

      return result
      
      
class Velocity(Atomwise):
  
  
    def __init__(
        self,
        n_in:int,
        n_out:int=50,
        aggregation_mode:str="avg",
        n_layers:int=2,
        n_neurons:Optional[Union[int, List[int]]]=None,
        activation:Callable=schnetpack.nn.activations.shifted_softplus,
        property:str="threshold_pca_vel",
        property_rotated:str = "smooth_vel",
        mean:Optional[torch.Tensor]=None,
        stddev:Optional[torch.Tensor]=None,
        outnet:Optional[torch.Tensor]=None, 
        atomref:Optional[torch.Tensor]=None,
        pca_basis:Optional[torch.Tensor] = None,
        std_dos = 1.0,
        mean_dos = 0.0):
      
            
        super(Velocity, self).__init__(
            n_in,
            n_out,
            aggregation_mode,
            n_layers,
            n_neurons,
            activation=activation,
            mean=mean,
            stddev=stddev,
            outnet=outnet,
            property=property,
        )
        self.property = property
        self.property_rotated = property_rotated
        
        if(self.property_rotated is not None):
            self.outnet_rotate =  schnetpack.nn.base.RotateScaleShift(pca_basis.T, mean_dos, std_dos)


    def forward(self, inputs:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        r"""
        predicts density of states in pca basis property
        """
        atomic_numbers = inputs[Properties.Z]
        atom_mask = inputs[Properties.atom_mask]

        # run prediction
        yi = self.out_net(inputs)
        yi = self.standardize(yi)

        if self.atomref is not None:
            y0 = self.atomref(atomic_numbers)
            yi = yi + y0
        
        y = self.atom_pool(yi, atom_mask)    # Removing the false atoms which was introduced for making equal size
        
        # Transforming from pca basis to energy basis
        if(self.property_rotated is not None):
            y_rotated = self.outnet_rotate(y)
                    
            # collect results
            result = {self.property: y, self.property_rotated: y_rotated}
        
        else:               
            # collect results
            result = {self.property: y}
            

        return result
      
      
      
class LocalDensityOfStates(Atomwise):
  
  
    def __init__(
        self,
        n_in:int,
        n_out:int=50,
        aggregation_mode:str="collect",
        n_layers:int=2,
        n_neurons:Optional[Union[int, List[int]]]=None,
        activation:Callable=schnetpack.nn.activations.shifted_softplus,
        property:str="threshold_pca_ldos",
        property_rotated:str = "smooth_ldos",
        total_property:str = "smooth_dosperatom",
        mean:Optional[torch.Tensor]=None,
        stddev:Optional[torch.Tensor]=None,
        outnet:Optional[Union[nn.Module, nn.Sequential]]=None, 
        atomref:Optional[torch.Tensor]=None,
        pca_basis:Optional[torch.Tensor] = None,
        std_ldos = 1.0,
        mean_ldos = 0.0):
      
            
        super(LocalDensityOfStates, self).__init__(
            n_in,
            n_out,
            aggregation_mode,
            n_layers,
            n_neurons,
            activation=activation,
            mean=mean,
            stddev=stddev,
            outnet=outnet,
            property=property,
        )
        self.property = property
        self.property_rotated = property_rotated
        self.total_property = total_property
        
        if(self.property_rotated is not None):
            self.outnet_rotate =  schnetpack.nn.base.RotateScaleShift(pca_basis.T, mean_ldos, std_ldos)


    def forward(self, inputs:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        r"""
        predicts density of states in pca basis property
        """
        atomic_numbers = inputs[Properties.Z]
        atom_mask = inputs[Properties.atom_mask]

        # run prediction
        yi = self.out_net(inputs)
        yi = self.standardize(yi)
        
        if self.atomref is not None:
            y0 = self.atomref(atomic_numbers)
            yi = yi + y0
      
      
        y = self.atom_pool(yi, atom_mask)    # Removing the false atoms which was introduced for making equal size
      
        # Transforming from pca basis to energy basis
        if(self.property_rotated is not None):
            y_rotated = self.outnet_rotate(y)
            num_atoms = torch.sum(atom_mask, axis = 1, keepdims = True)
            y_total = torch.sum(y_rotated, axis = 1)/num_atoms
            # collect results
            result = {self.property: y, self.property_rotated: y_rotated, self.total_property: y_total}
      
        else:               
            # collect results
            result = {self.property: y}
          

        return result


class Rotate_Vectorized(Atomwise):
    """
    Predicts a vector output. It can also rotate depending on the indicator
    rotate_vector = rotate_std * (vector @ rotate_basis) + rotate_mean 

    Args:
        property: string label for the vector property to be predicted
        mean: the mean value of the vector output training data
        stddev: The standard deviation value of the vector output training data
        rotate_indicator: Whether to rotate the predicted output or not
        rotate_property: string label for the vector property after rotation
        rotate_basis: the rotation matrix
        rotate_std: the scaling value after the rotation 
        rotate_mean: the shift after the scaling and rotation
        remove_elements_list: the list of elements which have to be removed using the atom mask
                              e.g. this is used for the case of hydrogen atoms on the dangling bonds of nanoslab silicon

    Returns:
        result: result dictionary
    """
  
  
    def __init__(
        self,
        n_in:int,
        n_out:int=50,
        aggregation_mode:str="avg",
        n_layers:int=2,
        n_neurons:Optional[Union[int, List[int]]]=None,
        activation:Callable=schnetpack.nn.activations.shifted_softplus,
        property:Optional[str]=None,
        mean:Optional[torch.Tensor]=None,
        stddev:Optional[torch.Tensor]=None,
        outnet:Optional[Union[nn.Module, nn.Sequential]]=None, 
        atomref:Optional[torch.Tensor]=None,
        rotate_indicator:bool=True,
        rotate_property:Optional[str]=None,
        rotate_basis:Optional[torch.Tensor] = None,
        rotate_std = 1.0,
        rotate_mean = 0.0,
        remove_elements_list = []
    ):
      
            
        super(Rotate_Vectorized, self).__init__(
            n_in,
            n_out,
            aggregation_mode,
            n_layers,
            n_neurons,
            activation=activation,
            mean=mean,
            stddev=stddev,
            outnet=outnet,
            property=property,
            atomref = atomref,
        )
        self.property = property
        self.rotate_property = rotate_property
        self.rotate_indicator = rotate_indicator
        self.remove_elements_list = remove_elements_list

        if(self.rotate_indicator is True):
            if(rotate_property is None):
                print("Provide the label for rotated property")
            self.outnet_rotate =  schnetpack.nn.base.RotateScaleShift(rotate_basis, rotate_mean, rotate_std)


    def forward(self, inputs:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        r"""
        predicts a vector and another rotated vector
        """
        atomic_numbers = inputs[Properties.Z]
        atom_mask = inputs[Properties.atom_mask]

        # run prediction
        yi = self.out_net(inputs)
        yi = self.standardize(yi)

        '''
        if self.atomref is not None:
            y0 = self.atomref(atomic_numbers)
            yi = yi + y0
        '''

        # Removing some input elements 
        remove_element_mask = torch.ones_like(atom_mask)
        if(len(self.remove_elements_list) > 0):
            for elem in self.remove_elements_list:
                remove_element_mask *= (~ (atomic_numbers == elem)) 
      
        mask = atom_mask * remove_element_mask
        y = self.atom_pool(yi, mask)    # Removing the false atoms which was introduced for making equal size
      
        # Rotating the predicted vector
        if(self.rotate_indicator is True):
            rotate_y = self.outnet_rotate(y)
                    
            # collect results
            result = {self.property: y, self.rotate_property: rotate_y}
        
        else:               
            # collect results
            result = {self.property: y}
          
        return result