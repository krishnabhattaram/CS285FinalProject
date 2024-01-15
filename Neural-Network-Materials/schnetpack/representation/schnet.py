import torch
import torch.nn as nn

from schnetpack.nn.base import Dense
from schnetpack import Properties
from schnetpack.nn.cfconv import CFConv, AngularCFConv
from schnetpack.nn.cutoff import CosineCutoff
from schnetpack.nn.acsf import GaussianSmearing, ExponentialBernsteinPolynomials, SphericalHarmonics
from schnetpack.nn.neighbors import AtomDistances, AtomVectors
from schnetpack.nn.activations import shifted_softplus
from typing import Dict, Optional, Callable, Tuple, Union, List


__all__ = ["SchNetInteraction", "SchNet", "AngularSchNetInteraction", "AngularSchNet"]


class SchNetInteraction(nn.Module):
    r"""SchNet interaction block for modeling interactions of atomistic systems.

    Args:
        n_atom_basis (int): number of features to describe atomic environments.
        n_spatial_basis (int): number of input features of filter-generating networks.
        n_filters (int): number of filters used in continuous-filter convolution.
        cutoff (float): cutoff radius.
        cutoff_network (nn.Module, optional): cutoff layer.
        normalize_filter (bool, optional): if True, divide aggregated filter by number
            of neighbors over which convolution is applied.
    """

    def __init__(
        self,
        n_atom_basis:int,
        n_spatial_basis:int,
        n_filters:int,
        cutoff:float,
        cutoff_network:nn.Module=CosineCutoff,
        normalize_filter:bool=False,
    ):
        super(SchNetInteraction, self).__init__()
        
        # filter block used in interaction block
        self.filter_network = nn.Sequential(
            Dense(n_spatial_basis, n_filters, activation=shifted_softplus),
            Dense(n_filters, n_filters),
        )
                      
        # cutoff layer used in interaction block
        self.cutoff_network = cutoff_network(cutoff)
        # interaction block
        self.cfconv = CFConv(
            n_atom_basis,
            n_filters,
            n_atom_basis,
            self.filter_network,
            cutoff_network=self.cutoff_network,
            activation=shifted_softplus,
            normalize_filter=normalize_filter,
        )
        # dense layer
        self.dense = Dense(n_atom_basis, n_atom_basis, bias=True, activation=None)
        
        self.x_relevance = None
        self.f_relevance = None
        self.W_relevance = None
  
    def forward(self,
                x:torch.Tensor,
                r_ij:torch.Tensor,
                neighbors:torch.Tensor,
                neighbor_mask:torch.Tensor,
                f_ij:Optional[torch.Tensor]=None) -> torch.Tensor:
        """Compute interaction output.

        Args:
            x (torch.Tensor): input representation/embedding of atomic environments
                with (N_b, N_a, n_atom_basis) shape.
            r_ij (torch.Tensor): interatomic distances of (N_b, N_a, N_nbh) shape.
            neighbors (torch.Tensor): indices of neighbors of (N_b, N_a, N_nbh) shape.
            neighbor_mask (torch.Tensor): mask to filter out non-existing neighbors
                introduced via padding.
            f_ij (torch.Tensor, optional): expanded interatomic distances in a basis.
                If None, r_ij.unsqueeze(-1) is used.

        Returns:
            torch.Tensor: block output with (N_b, N_a, n_atom_basis) shape.

        """
        # continuous-filter convolution interaction block followed by Dense layer
        v = self.cfconv(x, r_ij, neighbors, neighbor_mask, f_ij)
        v = self.dense(v)
        
        return v
        
    def relprop(self,
                x:torch.Tensor,
                r_ij:torch.Tensor,
                neighbors:torch.Tensor,
                neighbor_mask:torch.Tensor,
                v_relevance:torch.Tensor,
                f_ij:Optional[torch.Tensor]=None,
                lrp_fn:Optional[Callable] = None,
                epsilon:float = 0.0) -> Tuple[torch.Tensor]:
        """Compute layer relevance for its inputs (x and f_ij).

        Args:
            inputs (dict of torch.Tensor): batch of input values.
            relevance (torch.tensor): relevance of output interactions

        Returns:
            torch.Tensor : layer relevance of inputs (x and f_ij).

        """
        
        # continuous-filter convolution interaction block followed by Dense layer
        v_temp = self.cfconv(x, r_ij, neighbors, neighbor_mask, f_ij)
        v = self.dense(v_temp)
        
        v_temp_relevance = self.dense.relprop(v_temp, v_relevance)
        x_relevance, f_relevance, W_relevance = self.cfconv.relprop(x, r_ij, neighbors, neighbor_mask, v_temp_relevance, f_ij)
        
        self.x_relevance = x_relevance
        self.f_relevance = f_relevance
        self.W_relevance = W_relevance
        
        return x_relevance, f_relevance, W_relevance
        
        


class SchNet(nn.Module):
    """SchNet architecture for learning representations of atomistic systems.

    Args:
        n_atom_basis (int, optional): number of features to describe atomic environments.
            This determines the size of each embedding vector; i.e. embeddings_dim.
        n_filters (int, optional): number of filters used in continuous-filter convolution
        n_interactions (int, optional): number of interaction blocks.
        cutoff (float, optional): cutoff radius.
        n_gaussians (int, optional): number of Gaussian functions used to expand
            atomic distances.
        normalize_filter (bool, optional): if True, divide aggregated filter by number
            of neighbors over which convolution is applied.
        coupled_interactions (bool, optional): if True, share the weights across
            interaction blocks and filter-generating networks.
        return_intermediate (bool, optional): if True, `forward` method also returns
            intermediate atomic representations after each interaction block is applied.
        max_z (int, optional): maximum nuclear charge allowed in database. This
            determines the size of the dictionary of embedding; i.e. num_embeddings.
        cutoff_network (nn.Module, optional): cutoff layer.
        trainable_gaussians (bool, optional): If True, widths and offset of Gaussian
            functions are adjusted during training process.
        distance_expansion (nn.Module, optional): layer for expanding interatomic
            distances in a basis.
        charged_systems (bool, optional):

    References:
    .. [#schnet1] Schütt, Arbabzadah, Chmiela, Müller, Tkatchenko:
       Quantum-chemical insights from deep tensor neural networks.
       Nature Communications, 8, 13890. 2017.
    .. [#schnet_transfer] Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller:
       SchNet: A continuous-filter convolutional neural network for modeling quantum
       interactions.
       In Advances in Neural Information Processing Systems, pp. 992-1002. 2017.
    .. [#schnet3] Schütt, Sauceda, Kindermans, Tkatchenko, Müller:
       SchNet - a deep learning architecture for molceules and materials.
       The Journal of Chemical Physics 148 (24), 241722. 2018.

    """

    def __init__(
        self,
        n_atom_basis:int=128,
        n_filters:int=128,
        n_interactions:int=3,
        cutoff:float=5.0,
        environment_cutoff:float=5.0,
        n_gaussians:int=25,
        normalize_filter:bool=False,
        coupled_interactions:bool=False,
        return_intermediat:bool=False,
        max_z:int=100,
        cutoff_network:nn.Module=CosineCutoff,
        trainable_embedding:bool = True,
        trainable_gaussians:bool = False,
        distance_expansion:Optional[nn.Module]=None,
        charged_systems:bool=False,
    ):
        super(SchNet, self).__init__()

        self.n_atom_basis = n_atom_basis
        # make a lookup table to store embeddings for each element (up to atomic
        # number max_z) each of which is a vector of size n_atom_basis
        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0) 
        self.embedding.weight.requires_grad = trainable_embedding 

        # layer for computing interatomic distances
        self.distances = AtomDistances()
            
        # layer for expanding interatomic distances in a basis
        if distance_expansion is None:
            self.distance_expansion = GaussianSmearing(
                0.0, environment_cutoff, n_gaussians, trainable=trainable_gaussians)
        else:
            self.distance_expansion = distance_expansion
        
        # block for computing interaction
        if coupled_interactions:
            # use the same SchNetInteraction instance (hence the same weights)
            self.interactions = nn.ModuleList(
                [
                    SchNetInteraction(
                        n_atom_basis=n_atom_basis,
                        n_spatial_basis=n_gaussians,
                        n_filters=n_filters,
                        cutoff_network=cutoff_network,
                        cutoff=cutoff,
                        normalize_filter=normalize_filter,
                    )
                ]
                * n_interactions
            )
        else:
            # use one SchNetInteraction instance for each interaction
            self.interactions = nn.ModuleList(
                [
                    SchNetInteraction(
                        n_atom_basis=n_atom_basis,
                        n_spatial_basis=n_gaussians,
                        n_filters=n_filters,
                        cutoff_network=cutoff_network,
                        cutoff=cutoff,
                        normalize_filter=normalize_filter,
                    )
                    for _ in range(n_interactions)
                ]
            )

        # set attributes
        self.return_intermediate = return_intermediate
        self.charged_systems = charged_systems
        
        if charged_systems:
            self.charge = nn.Parameter(torch.Tensor(1, n_atom_basis))
            self.charge.data.normal_(0, 1.0 / n_atom_basis ** 0.5)
        
        self.xs_relevance = []
        self.fs_relevance = []
        self.vs_relevance = []
        self.Ws_relevance = []
        self.x_relevance = None
        self.rij_relevance = None
        

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
        r_ij = self.distances(
            positions, neighbors, cell, cell_offset, neighbor_mask=neighbor_mask
        )
                
        # expand interatomic distances (for example, Gaussian smearing)
        f_ij = self.distance_expansion(r_ij)
        # store intermediate representations
        if self.return_intermediate:
            xs = [x]
        # compute interaction block to update atomic embeddings
        for interaction in self.interactions:
            v = interaction(x, r_ij, neighbors, neighbor_mask, f_ij=f_ij)
            x = x + v
            if self.return_intermediate:
                xs.append(x)
        if self.return_intermediate:
            return x, xs
        
        return x
        
    def relprop(self,
                inputs:Dict[str, torch.Tensor],
                representation_relevance:torch.Tensor,
                lrp_fn:Optional[Callable] = None,
                epsilon:float = 0.0) -> torch.Tensor:
        
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
        r_ij = self.distances(
            positions, neighbors, cell, cell_offset, neighbor_mask=neighbor_mask
        )
                
        # expand interatomic distances (for example, Gaussian smearing)
        f_ij = self.distance_expansion(r_ij)
        # store intermediate representations
        xs = [x]
        vs = []
        # compute interaction block to update atomic embeddings
        for interaction in self.interactions:
            v = interaction(x, r_ij, neighbors, neighbor_mask, f_ij=f_ij)
            vs.append(v)
            x = x + v
            xs.append(x)
        
        xnext_relevance = representation_relevance
        self.xs_relevance = [representation_relevance]
        for ii, interaction in enumerate(reversed(self.interactions)):
            x = reversed(xs)[ii + 1]
            v = reversed(vs)[ii]
            
            x_relevance = x/(x + v)*xnext_relevance
            v_relevance = v/(x + v)*xnext_relevance
            
            xtemp_relevance, f_relevance, W_relevance = interaction.relprop(x, r_ij, neighbors, neighbor_mask, v_relevance, f_ij = f_ij) 
            x_relevance += xtemp_relevance
            
            
            self.fs_relevance.append(f_relevance)
            self.vs_relevance.append(v_relevance)
            self.xs_relevance.append(x_relevance)
            self.Ws_relevance.append(W_relevance)
                      
            xnext_relevance = x_relevance
            
        self.xs_relevance = reversed(self.xs_relevance)
        self.fs_relevance = reversed(self.fs_relevance)
        self.vs_relevance = reversed(self.vs_relevance)  
        self.Ws_relevance = reversed(self.Ws_relevance)             
            
            
        if coupled_interactions:
            ''' Same filter network used for all interaction blocks. Therefore add up relevance of weights from all interaction blocks'''
            '''The way it has been set up is that during backward relevance propagation, then the first interaction block returns the summed up 
                relevances of the weights and the first f in the list represents the true relevance of that
            ''' 
            self.rij_relevance = self.distance_expansion.relprop(r_ij, self.fs_relevance[0])
            
        else:
            '''
            Have to sum up all the fs relevance to get the original f relevance. Different filter network is used for each interaction block.
            '''
            
            self.rij_relevance = self.distance_expansion.relprop(r_ij, sum(self.fs_relevance))
         
        self.x_relevance = self.xs_relevance[0]
                     
        return self.x_relevance, self.rij_relevance
        
        
        
        
class AngularSchNetInteraction(nn.Module):
    r"""SchNet interaction block for modeling interactions of atomistic systems including spherical harmonics

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
        super(AngularSchNetInteraction, self).__init__()
        
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


class AngularSchNet(nn.Module):
    """SchNet architecture for learning representations of atomistic systems including spherical harmonics

    Args:
        n_atom_basis (int, optional): number of features to describe atomic environments.
            This determines the size of each embedding vector; i.e. embeddings_dim.
        n_filters (int, optional): number of filters used in continuous-filter convolution
        n_interactions (int, optional): number of interaction blocks.
        cutoff (float, optional): cutoff radius.
        n_gaussians (int, optional): number of Gaussian functions used to expand
            atomic distances.
        normalize_filter (bool, optional): if True, divide aggregated filter by number
            of neighbors over which convolution is applied.
        coupled_interactions (bool, optional): if True, share the weights across
            interaction blocks and filter-generating networks.
        return_intermediate (bool, optional): if True, `forward` method also returns
            intermediate atomic representations after each interaction block is applied.
        max_z (int, optional): maximum nuclear charge allowed in database. This
            determines the size of the dictionary of embedding; i.e. num_embeddings.
        cutoff_network (nn.Module, optional): cutoff layer.
        sblock_indicator (bool, optional): if True evalute using s orbitals
        pblock_indicator (bool, optional): if True evalute using p orbitals
        dblock_indicator (bool, optional): if True evalute using d orbitals
        trainable_gaussians (bool, optional): If True, widths and offset of Gaussian
            functions are adjusted during training process.
        distance_expansion (nn.Module, optional): layer for expanding interatomic
            distances in a basis.
        charged_systems (bool, optional):

    References:
    .. [#schnet1] Schütt, Arbabzadah, Chmiela, Müller, Tkatchenko:
       Quantum-chemical insights from deep tensor neural networks.
       Nature Communications, 8, 13890. 2017.
    .. [#schnet_transfer] Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller:
       SchNet: A continuous-filter convolutional neural network for modeling quantum
       interactions.
       In Advances in Neural Information Processing Systems, pp. 992-1002. 2017.
    .. [#schnet3] Schütt, Sauceda, Kindermans, Tkatchenko, Müller:
       SchNet - a deep learning architecture for molceules and materials.
       The Journal of Chemical Physics 148 (24), 241722. 2018.

    """

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
        trainable_embedding:bool = True,
        trainable_gaussians:bool = False,
        distance_expansion:Optional[nn.Module] = None,
        charged_systems:bool = False,
    ):
        super(AngularSchNet, self).__init__()

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
                    AngularSchNetInteraction(
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
        else:
            # use one SchNetInteraction instance for each interaction
            self.interactions = nn.ModuleList(
                [
                    AngularSchNetInteraction(
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

        # set attributes
        self.sblock_indicator = sblock_indicator
        self.pblock_indicator = pblock_indicator
        self.dblock_indicator = dblock_indicator
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
        for interaction in self.interactions:
            v = interaction(x, r_ij, neighbors, neighbor_mask, fsblock_ij=fsblock_ij, fpblock_ij = fpblock_ij, fdblock_ij = fdblock_ij)
            x = x + v            
            if self.return_intermediate:
                xs.append(x)
        if self.return_intermediate:
            return x, xs
        
        return x
