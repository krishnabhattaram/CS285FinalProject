import torch
import torch.nn as nn

from schnetpack.nn.base import Dense
from schnetpack.nn.attention import Attention
from schnetpack import Properties
from schnetpack.nn.cfconv import CFConv, AngularCFConv
from schnetpack.nn.cutoff import CosineCutoff
from schnetpack.nn.acsf import GaussianSmearing, ExponentialBernsteinPolynomials, SphericalHarmonics
from schnetpack.nn.neighbors import AtomDistances, AtomVectors
from schnetpack.nn.activations import shifted_softplus
from typing import Optional, Dict, Callable, Union, List


__all__ = ["TbpNet"]


class NonLocalInteraction(nn.Module):
    r"""
    Non-Local Interaction Block for updating atomic features through nonlocal interactions with all
    atoms.
    Arguments:
        num_atom_features (int):
            Dimensions of the atomic features.
        num_basis_functions (int):
            Number of radial basis functions.
        num_residual_q (int):
            Number of residual blocks applied to atomic features in q branch
            (central atoms) before computing the interaction.
        num_residual_k (int):
            Number of residual blocks applied to atomic features in k branch
            (neighbouring atoms) before computing the interaction.
        num_residual_v (int):
            Number of residual blocks applied to interaction features.
        activation (str):
            Kind of activation function. Possible values:
            (1) Shifted softplus activation function.
    """

    def __init__(
        self,
        num_atom_features:int,
        num_residual_q:Optional[int] = None,
        num_residual_k:Optional[int] = None,
        num_residual_v:Optional[int] = None,
        activation:Optional[Callable] = None,
    ):
        """ Initializes the NonlocalInteraction class. """
        super(NonLocalInteraction, self).__init__()
        self.resblock_q = Dense(
            num_atom_features, num_atom_features, activation=activation, bias = False
        )
        self.resblock_k = Dense(
            num_atom_features, num_atom_features, activation=activation, bias = False
        )
        self.resblock_v = Dense(
            num_atom_features, num_atom_features, activation=activation, bias = False
        )
        self.attention = Attention(num_atom_features, num_atom_features, num_atom_features)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ For compatibility with other modules. """
        pass

    def forward(
        self,
        x:torch.Tensor,
        atom_mask:torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate interaction block.
        N: Number of atoms.
       
            
        Args:
            x (FloatTensor [n_batch, n_atoms, num_atom_features]): Atomic feature vectors.
            atom_mask (Float Tensor [n_batch, n_atoms]): The mask to remove the false atoms
            
        Returns:
            attention_matrix (torch.Tensor)
            
        """
        x_masked = x * atom_mask.unsqueeze(-1)    # Masking the false atoms
        q = self.resblock_q(x_masked)  # queries
        k = self.resblock_k(x_masked)  # keys
        v = self.resblock_v(x_masked)  # values
        n = self.attention(q, k, v, atom_mask)
                
        return n
        
        
class LocalInteraction(nn.Module):
    r"""Local interaction block for modeling interactions of atomistic systems including spherical harmonics

    Args:
        n_atom_basis (int): number of features to describe atomic environments.
        n_spatial_basis (int): number of input features of filter-generating networks.
        n_filters (int): number of filters used in continuous-filter convolution.
        cutoff (float): cutoff radius.
        cutoff_network (nn.Module, optional): cutoff layer.
        normalize_filter (bool, optional): if True, divide aggregated filter by number
            of neighbors over which convolution is applied.
        sblock_indicator (bool, optional): if True evalute using s orbitals
        pblock_indicator (bool, optional): if True evalute using p orbitals
        dblock_indicator (bool, optional): if True evalute using d orbitals
    """

    def __init__(
        self,
        n_atom_basis:int,
        n_spatial_basis:int,
        n_filters:int,
        cutoff:float,
        cutoff_network:nn.Module=CosineCutoff,
        normalize_filter:bool=False,
        sblock_indicator:bool = True,
        pblock_indicator:bool = True,
        dblock_indicator:bool = False,
        projection_indicator:bool = False,
    ):
        super(LocalInteraction, self).__init__()
        
        # filter block used in interaction block
        self.filter_network_sblock = nn.Sequential(
            Dense(n_spatial_basis, n_filters, bias = False),
        )
        
        self.filter_network_pblock = nn.Sequential(
            Dense(n_spatial_basis, n_filters, bias = False),
        )
        
        self.filter_network_dblock = nn.Sequential(
            Dense(n_spatial_basis, n_filters, bias = False),
        )
                      
        # cutoff layer used in interaction block
        self.cutoff_network = cutoff_network(cutoff)
        # interaction block
        self.cfconv = AngularCFConv(
            n_atom_basis,
            n_filters,
            n_atom_basis,
            self.filter_network_sblock,
            self.filter_network_pblock,
            self.filter_network_dblock,
            cutoff_network=self.cutoff_network,
            activation=shifted_softplus,
            normalize_filter=normalize_filter,
            sblock_indicator = sblock_indicator,
            pblock_indicator = pblock_indicator,
            dblock_indicator = dblock_indicator,
            projection_indicator = projection_indicator,
        )

        # dense layer
        self.dense = Dense(n_atom_basis, n_atom_basis, bias=True, activation=None)

    def forward(self,
                x:torch.Tensor,
                r_ij:torch.Tensor,
                neighbors:torch.Tensor,
                neighbor_mask:torch.Tensor,
                fsblock_ij:Optional[torch.Tensor]=None,
                fpblock_ij:Optional[torch.Tensor]=None,
                fdblock_ij:Optional[torch.Tensor]=None) -> torch.Tensor:
        """Compute interaction output.

        Args:
            x (torch.Tensor): input representation/embedding of atomic environments
                with (N_b, N_a, n_atom_basis) shape.
            r_ij (torch.Tensor): interatomic distances of (N_b, N_a, N_nbh) shape.
            neighbors (torch.Tensor): indices of neighbors of (N_b, N_a, N_nbh) shape.
            neighbor_mask (torch.Tensor): mask to filter out non-existing neighbors
                introduced via padding.
            fsblock_ij (torch.Tensor, optional): expanded interatomic distances in a s-orbital basis.
            fpblock_ij (torch.Tensor, optional): expanded interatomic distances in a p-orbital basis.
            fdblock_ij (torch.Tensor, optional): expanded interatomic distances in a d-orbital basis.

        Returns:
            torch.Tensor: block output with (N_b, N_a, n_atom_basis) shape.

        """
        
        # continuous-filter convolution interaction block followed by Dense layer
        v = self.cfconv(x, r_ij, neighbors, neighbor_mask, fsblock_ij, fpblock_ij, fdblock_ij)
        v = self.dense(v)
        return v


class TbpNet(nn.Module):

    
    def __init__(
        self,
        n_atom_basis:int=128,
        n_filters:int=128,
        n_interactions:int=3,
        cutoff:float=5.0,
        environment_cutoff:float=5.0,
        n_basis_functions:int=25,
        normalize_filter:bool=False,
        coupled_interactions:bool=False,
        return_intermediate:bool=False,
        max_z:int=100,
        cutoff_network:nn.Module=CosineCutoff,
        sblock_indicator:bool = True,
        pblock_indicator:bool = True,
        dblock_indicator:bool = False,
        projection_indicator:bool = False,
        attention_indicator:bool = True,
        trainable_embedding:bool = True,
        trainable_gaussians:bool = False,
        distance_expansion:bool = None,
        charged_systems:bool = False,
    ):
        super(TbpNet, self).__init__()

        self.n_atom_basis = n_atom_basis
        # make a lookup table to store embeddings for each element (up to atomic
        # number max_z) each of which is a vector of size n_atom_basis
        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0) 
        self.embedding.weight.requires_grad = trainable_embedding 

        # layer for computing interatomic distances and vectors
        self.distances = AtomDistances(return_directions=True)
            
        # layer for expanding interatomic distances in a basis
        if distance_expansion is None:
            self.distance_expansion = ExponentialBernsteinPolynomials(n_basis_functions)
        else:
            self.distance_expansion = distance_expansion
            
        # layer for calculating spherical harmonics
        self.spherical_harmonics = SphericalHarmonics(sblock_indicator, pblock_indicator, dblock_indicator)
        

        # block for computing interaction
        if coupled_interactions:
            # use the same SchNetInteraction instance (hence the same weights)

            self.interactions = nn.ModuleList(
                [
                    LocalInteraction(
                        n_atom_basis=n_atom_basis,
                        n_spatial_basis=n_basis_functions,
                        n_filters=n_filters,
                        cutoff_network=cutoff_network,
                        cutoff=cutoff,
                        normalize_filter=normalize_filter,
                        sblock_indicator = sblock_indicator,
                        pblock_indicator = pblock_indicator,
                        dblock_indicator = dblock_indicator,
                        projection_indicator = projection_indicator,
                    )
                ]
                * n_interactions
            )
            
            self.non_local_interactions = nn.ModuleList(
                [    
                    NonLocalInteraction(
                        num_atom_features=n_atom_basis,
                        activation=shifted_softplus
                    )
                ]
                * n_interactions            
            )
        else:
            # use one Interaction instance for each interaction
            self.interactions = nn.ModuleList(
                [
                    LocalInteraction(
                        n_atom_basis=n_atom_basis,
                        n_spatial_basis=n_basis_functions,
                        n_filters=n_filters,
                        cutoff_network=cutoff_network,
                        cutoff=cutoff,
                        normalize_filter=normalize_filter,
                        sblock_indicator = sblock_indicator,
                        pblock_indicator = pblock_indicator,
                        dblock_indicator = dblock_indicator,
                        projection_indicator = projection_indicator,
                    )
                    for _ in range(n_interactions)
                ]
            )
                
            self.non_local_interactions = nn.ModuleList(
                [
                    NonLocalInteraction(
                        num_atom_features=n_atom_basis,
                        activation=shifted_softplus
                    )
                    for _ in range(n_interactions)
                ]
            )
            
        

        # set attributes
        self.sblock_indicator = sblock_indicator
        self.pblock_indicator = pblock_indicator
        self.dblock_indicator = dblock_indicator
        self.attention_indicator = attention_indicator
        self.return_intermediate = return_intermediate
        self.charged_systems = charged_systems
        
        if charged_systems:
            self.charge = nn.Parameter(torch.Tensor(1, n_atom_basis))
            self.charge.data.normal_(0, 1.0 / n_atom_basis ** 0.5)

    def forward(self, inputs:Dict[str, torch.Tensor]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Compute atomic representations/embeddings.

        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        Returns:
            torch.Tensor: atom-wise representation.
            list of torch.Tensor: intermediate atom-wise representations, if
            return_intermediate=True was used.

        """
        
        # get tensors from input dictionary
        atomic_numbers = inputs[Properties.Z]
        positions = inputs[Properties.R]
        cell = inputs[Properties.cell]
        cell_offset = inputs[Properties.cell_offset]
        neighbors = inputs[Properties.neighbors]
        neighbor_mask = inputs[Properties.neighbor_mask]
        atom_mask = inputs[Properties.atom_mask]
        
        
        # get atom embeddings for the input atomic numbers
        x = self.embedding(atomic_numbers)

        if self.charged_systems and Properties.charge in inputs.keys():
            n_atoms = torch.sum(atom_mask, dim=1, keepdim=True)
            charge = inputs[Properties.charge] / n_atoms  # B
            charge = charge[:, None] * self.charge  # B x F
            x = x + charge
        
        # compute interatomic distance of every atom to its neighbors
        r_ij, rvec_ij = self.distances(
            positions, neighbors, cell, cell_offset, neighbor_mask=neighbor_mask
        )
        
        # expand interatomic distances (for example, Exponential Bernstein polynomials)
        fradial_ij = self.distance_expansion(r_ij)
        fangular_list_ij = self.spherical_harmonics(rvec_ij)
        
        # get the spherical harmonics evaluation 
        
        if self.sblock_indicator:
            fsblock_ij = fradial_ij.unsqueeze(3)*fangular_list_ij[0].unsqueeze(4)
        else:
            fsblock_ij = None 
            
        if self.pblock_indicator:
            fpblock_ij = fradial_ij.unsqueeze(3)*fangular_list_ij[1].unsqueeze(4)
        else:
            fpblock_ij = None
        
        if self.dblock_indicator:
            fdblock_ij = fradial_ij.unsqueeze(3)*fangular_list_ij[2].unsqueeze(4)
        else:
            fdblock_ij = None

        # store intermediate representations
        if self.return_intermediate:
            xs = [x]
        # compute interaction block to update atomic embeddings
        for ii, local_interaction in enumerate(self.interactions):
            v = local_interaction(x, r_ij, neighbors, neighbor_mask, fsblock_ij=fsblock_ij, fpblock_ij = fpblock_ij, fdblock_ij = fdblock_ij)
            
            if(self.attention_indicator == True):
                n_num_atoms = torch.sum(atom_mask, axis = 1, keepdim = True)[:, :, None]
                n = self.non_local_interactions[ii](x, atom_mask)/n_num_atoms
                x = x + v + n
            else:
                x = x + v   
                         
            if self.return_intermediate:
                xs.append(x)
        if self.return_intermediate:
            return x, xs
        
        return x