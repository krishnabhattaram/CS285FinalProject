import numpy as np
import torch
import copy
import os
import sys

from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list
from torch.autograd import grad

from schnetpack.nn import atom_distances, triple_distances
from schnetpack.environment import collect_atom_triples, AseEnvironmentProvider
from schnetpack.utils.analytical_utils.SiliconConstants import *

import scipy.linalg as splin

__all__ = [
    "Stillinger_Weber",
    "Heterogenous_Stillinger_Weber",
    "SiO2_Stillinger_Weber",
    "Pedone",
    "Tight_Binding_Silicon",
]





class Stillinger_Weber(Calculator):
    r'''
    Class for calculating the stillinger weber energy of the atom. The formula is given by:
    Upair(r_ij) = A(B/(r_ij^p) - 1/r_ij^q)h_\beta(r_ij)
    Utrip(r_ij, r_ik) = \lambda*h_\gamma(r_ij)*h_\gamma(r_ik)(cos \theta_jik + 1/3)^2
    h_\delta(r) = e^(\delta(r - a)) r < a
    
    a is the cutoff value
    '''

    implemented_properties = ['energy', 'energies', 'forces']
    default_parameters = {
        'unit_to_angstrom': 2.0951,    # Angstrom/unit
        'unit_to_ergs': 3.4723*1e-12,    # ergs/unit
        'angstrom_to_unit': 1/2.0951,    # unit/Angstrom
        'ergs_to_ev':6.242*1e11,   # ev/ergs
        'A': 7.04956,
        'B': 0.602225,
        'p': 4,
        'q':0,
        'beta':1,
        'lambda_c':21,
        'gamma':1.2,
        'a': 1.8*2.0951,    # Angstrom
        'neighbor_indices':None,
        'neighbor_mask':None,
        'cell_offset':None,
        'cell': None,
        'use_environment': True
    }
    nolabel = True

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)


    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
    
        r'''
        Calculates the total potential energy, local energies and forces with the Stillinger Weber potential
        
        a) local energies for all atom in the atoms
        b) total potential energy
        c) forces on each atom
        
        Args:
            atoms (Ase atoms object): The input atomic structure
            properties (Optional): Some parameter required by Ase calculator
            system_changes (Optional): Some parameter required by Ase calculator

        
        '''
    
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)

        A = np.float32(self.parameters.A)
        B = np.float32(self.parameters.B)
        p = np.float32(self.parameters.p)
        q = np.float32(self.parameters.q)
        beta = np.float32(self.parameters.beta)
        lambda_c = np.float32(self.parameters.lambda_c)
        gamma = np.float32(self.parameters.gamma)
        a = np.float32(self.parameters.a*self.parameters.angstrom_to_unit)
        positions = self.atoms.positions*self.parameters.angstrom_to_unit
        natoms = len(self.atoms)
        
        if self.parameters.use_environment is True:
            neighbor_indices, neighbor_mask, cell, cell_offset = self.create_environment(a*self.parameters.unit_to_angstrom, natoms)
            self.parameters.neighbor_indices = neighbor_indices    # For probing the class
            self.parameters.neighbor_mask = neighbor_mask    # For probing the class
            self.parameters.cell = cell    # For probing the class
            self.parameters.cell_offset = cell_offset    # For probing the class
                   
        else:
            neighbor_indices = self.parameters.neighbor_indices
            neighbor_mask = self.parameters.neighbor_mask
            cell = self.parameters.cell*self.parameters.angstrom_to_unit
            cell_offset = self.parameters.cell_offset*self.parameters.angstrom_to_unit    
        
        if self.atoms.get_pbc() is False:
            cell_offset = np.zeros((neighbor_indices.shape[0], neighbor_indices.shape[1], 3))
        
        nbh_idx_j, nbh_idx_k, offset_idx_j, offset_idx_k = collect_atom_triples(neighbor_indices)
            
        # torchifying the numpy data 
        positions = torch.from_numpy(positions)[None,:,:].float()
        neighbor_indices = torch.from_numpy(neighbor_indices)[None,:,:]
        nbh_idx_j = torch.from_numpy(nbh_idx_j)[None,:,:]
        nbh_idx_k = torch.from_numpy(nbh_idx_k)[None,:,:]
        offset_idx_j = torch.from_numpy(offset_idx_j)[None,:,:]
        offset_idx_k = torch.from_numpy(offset_idx_k)[None,:,:]
        neighbor_mask = torch.from_numpy(neighbor_mask)[None,:,:]
        cell = torch.from_numpy(cell)[None,:,:].float()
        cell_offset = torch.from_numpy(cell_offset)[None,:,:,:].float()
        energies = torch.zeros((1, natoms)).float()
        forces = torch.zeros((1, natoms, 3)).float()

        # Enabling calculating gradients for necessary variables
        positions.requires_grad = True
        energies.requires_grad = True
        forces.requires_grad = True
        
        rr_ij, rr_ij_vec = atom_distances(positions, neighbor_indices, cell=cell, cell_offsets=cell_offset, return_vecs = True)
        R_ij, R_ik, R_jk = triple_distances(positions, nbh_idx_j, nbh_idx_k, offset_idx_j, offset_idx_k, cell=cell, cell_offsets=cell_offset, return_vecs=True)

        nbatch, natoms, nneigh = neighbor_indices.shape
        triu_idx_row, triu_idx_col = np.triu_indices(nneigh, k=1)
        triu_idx_flat = triu_idx_row * nneigh + triu_idx_col
        
        neighbor_mask_ij = neighbor_mask.repeat(1, 1, nneigh) 
        neighbor_mask_ij = neighbor_mask_ij[:,:,triu_idx_flat]
        neighbor_mask_ik = neighbor_mask.repeat_interleave(nneigh, dim=2).reshape((1, natoms, -1))
        neighbor_mask_ik = neighbor_mask_ik[:,:,triu_idx_flat]

        energies = (self.calc_pairwise_energy(rr_ij, A, B, p, q, beta, a, neighbor_mask) + self.calc_triplets_energy(R_ij, R_ik, lambda_c, gamma, a, neighbor_mask_ij, neighbor_mask_ik)).squeeze()*self.parameters.unit_to_ergs*self.parameters.ergs_to_ev
           
        energy = energies.sum()
        forces = -grad(energy, positions, grad_outputs=torch.ones_like(energy), retain_graph=True,)[0].squeeze()/self.parameters.unit_to_angstrom

        self.results['energy'] = energy.detach().cpu().numpy()    # eV
        self.results['energies'] = energies.detach().cpu().numpy()    # eV
        self.results['forces'] = forces.detach().cpu().numpy()    # eV/Ao
     
     
     
    def cutoff(self, constant , r, a, neighbor_mask):
        '''
        The exponential cutoff function used in the stillinger weber potential for smooth decay
        h_\delta(r) = e^(\delta(r - a)) r < a
        Args:
            constant (float) : The value of delta
            r (torch.tensor) : The neighbor distances for the central atom
            a (float) : The cutoff value
            neighbor mask (torch.tensor) : mask which removes the non exitsent neigbhors from neighbor indices
            
        Returns:
            value (np.array) : Cutoff values
        '''
        
        temp_index = torch.where(r < a)
        value = torch.zeros_like(r)
        value[temp_index] = torch.exp(constant/(r[temp_index] - a))*neighbor_mask[temp_index]
        return value   # Adding a small constant for when r = a. r cannot be greater than a as it is taken care of when finding out the environment around the atom
     
             
    def calc_pairwise_energy(self, rr_ij, A, B, p, q, beta, a, neighbor_mask):
        r'''
        Calculates the pairwise energy for the Stillinger Weber potential
        Upair(r_ij) = A(B/(r_ij^p) - 1/r_ij^q)h_\beta(r_ij)
        
        Args:
          
            rr_ij (torch.tensor of float) : The distances to the neighbors of the cental atom
            A (float) : The value of A
            B (float) : The value of B
            p (float) : The value of p
            q (float) : The value of q
            beta (float) : The value of beta for the cutoff function
            a (float) : The value of cutoff
            neighbor_mask (torch.tensor) : mask which removes the non exitsent neigbhors from neighbor indices
            
        Returns:
          
            U_pairwise (torch.tensor) : The pairwise energy of every atom in the atoms Ase object. Need to sum to getthe total radial potential energy
        
        '''
        cutoff = self.cutoff(beta, rr_ij, a, neighbor_mask)
        U_pairwise = 1/2*torch.sum(A*(B/(torch.pow(rr_ij, p) + 1e-12) - 1/(torch.pow(rr_ij,q) + 1e-12))*cutoff, dim = 2)
        return U_pairwise
        
    def calc_triplets_energy(self, R_ij, R_ik, lambda_c, gamma, a, neighbor_mask_ij, neighbor_mask_ik):
        r'''
        Calculates the triplet energy for the Stillinger Weber potential
        Utrip(r_ij, r_ik) = \lambda*h_\gamma(r_ij)*h_\gamma(r_ac)(cos \theta_jik + 1/3)^2
        
        Args:
            
            R_ij (torch.tensor) : The distance vectors to the first neighbor
            R_ik (torch.tensor) : The distance vectors to the second neighbor
            lambda_c (torch.tensor) : The value of lambda
            gamma (torch.tensor) : The value of gamma for the cutoff function
            a (float) : The value of cutoff 
            neighbor_mask_ij (torch.tensor) : The neighbor mask applied for the first neighbors
            neighbor_mask_ik (torch.tensor) : The neighbor mask applied for the second neighbors
            
        Returns:
        
            U_triplet (torch.tensor) : The triplet local energies for all the atoms. Need to sum it up to get the total angular energy
            
        '''
        r_ij = torch.norm(R_ij, dim=3)
        r_ik = torch.norm(R_ik, dim=3)
        costheta = torch.sum(R_ij*R_ik, dim=3)/(r_ij*r_ik + 1e-12)
        
        cutoff = self.cutoff(gamma, r_ij, a, neighbor_mask_ij)*self.cutoff(gamma, r_ik, a, neighbor_mask_ik)
        U_triplet = torch.sum(lambda_c*torch.pow(costheta + 1/3, 2)*cutoff, dim = 2) 
        return U_triplet
     
     
    def create_environment(self, rc, natoms):
        r'''
        Creates the neighbor environment for all atoms.
        
        Args:
           
           rc (float) : The value of cutoff within which neighbors are found
           natoms (int) : The number of atoms in the structure
           
        Returns:
         
           neighborhood_idx (np.array): The neighbor indices for all the atoms 
           mask (np.array) : Mask to remove the non existent neighbors
           cell (np.array) : The basis of the unit cell of the structure
           offset (np.array) : Offset for all the neighbors to calculate the distance vectors
        ''' 
        
        idx_i, idx_j, idx_S = neighbor_list(
            "ijS", self.atoms, rc, self_interaction=False
        )
        if idx_i.shape[0] > 0:
            uidx, n_nbh = np.unique(idx_i, return_counts=True)
            n_max_nbh = np.max(n_nbh)

            n_nbh = np.tile(n_nbh[:, np.newaxis], (1, n_max_nbh))
            nbh_range = np.tile(
                np.arange(n_max_nbh, dtype=np.int)[np.newaxis], (n_nbh.shape[0], 1)
            )

            mask = np.zeros((natoms, np.max(n_max_nbh)), dtype=np.bool)
            mask[uidx, :] = nbh_range < n_nbh
            neighborhood_idx = -np.ones((natoms, np.max(n_max_nbh)), dtype=np.int64)
            neighborhood_idx[mask] = idx_j
            
            offset = np.zeros((natoms, np.max(n_max_nbh), 3), dtype=np.float32)
            offset[mask] = idx_S
            
        else:
            neighborhood_idx = -np.ones((natoms, 1), dtype=np.int64)
            mask = np.zeros((natoms, 1), dtype=np.int64)
            offset = np.zeros((natoms, 1, 3), dtype=np.float32)
        
        cell = np.array(self.atoms.get_cell())*self.parameters.angstrom_to_unit
        
        return neighborhood_idx, mask, cell, offset
        

class Pedone(Calculator):
    r'''
    Class for calculating the pedone energy and force fields of the atom. The formula is given by:
    Upair(r_ij) = ZiZj e^2/r + Dij[ {1 - exp(-aij (r - r0))} ^2 - 1] + Cij/r^12
    First is a long range coulomb repulsion. Should be calculated using Ewald sum method
    The other two parts are short range attractive forces
    
    a is the cutoff value
    '''

    implemented_properties = ['energy', 'energies', 'forces']
    default_parameters = {
        'D_SiO': 0.340554,    # eV
        'a_SiO': 2.006700,    # A^-2
        'ro_SiO': 2.1,    # A
        'C_SiO': 1.0,    # eV A^12
        'D_HfO': 0.324143,    # eV
        'a_HfO': 2.5702,    # A^-2
        'ro_HfO': 2.34720,    # A
        'C_HfO': 1.0,    # eV A^12
        'D_OO': 0.042395,    # eV
        'a_OO': 1.3793,    # A^-2
        'ro_OO': 3.61870,    # A
        'C_OO': 22,    # eV A^12
        'charge_O': -1.2,
        'charge_Hf': 2.4,
        'charge_Si':2.4,  
        'short_cutoff': 5.5,    # A
        'long_cutoff': 7.0,    # A
        'neighbor_indices':None,
        'neighbor_mask':None,
        'cell_offset':None,
        'cell': None,
        'use_environment': True
    }
    nolabel = True

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)
        
        self.D_SiO = self.parameters.D_SiO
        self.a_SiO = self.parameters.a_SiO
        self.ro_SiO = self.parameters.ro_SiO
        self.C_SiO = self.parameters.C_SiO
        
        self.D_HfO = self.parameters.D_HfO
        self.a_HfO = self.parameters.a_HfO
        self.ro_HfO = self.parameters.ro_HfO
        self.C_HfO = self.parameters.C_HfO
        
        self.D_OO = self.parameters.D_OO
        self.a_OO = self.parameters.a_OO
        self.ro_OO = self.parameters.ro_OO
        self.C_OO = self.parameters.C_OO
        
        self.charge_O = self.parameters.charge_O
        self.charge_Hf = self.parameters.charge_Hf
        self.charge_Si = self.parameters.charge_Si
        
        self.short_cutoff = self.parameters.short_cutoff
        self.long_cutoff = self.parameters.long_cutoff
        
        self.Si_atomic_number = 14.0
        self.Hf_atomic_number = 72.0
        self.O_atomic_number = 8.0
        
        self.D_matrix = None
        self.a_matrix = None
        self.C_matrix = None
        self.ro_matrix = None
        self.ZiZj_matrix = None

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
    
        r'''
        Calculates the total potential energy, local energies and forces with the Pedone potential
        
        a) local energies for all atom in the atoms
        b) total potential energy
        c) forces on each atom
        
        Args:
            atoms (Ase atoms object): The input atomic structure
            properties (Optional): Some parameter required by Ase calculator
            system_changes (Optional): Some parameter required by Ase calculator

        
        '''
    
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)
        
        positions = self.atoms.positions
        natoms = len(self.atoms)
        
        if self.parameters.use_environment is True:
            
            neighbor_indices, neighbor_mask, cell, cell_offset = self.create_environment(self.short_cutoff, natoms)    # For short range forces
            neighbor_indices_long, neighbor_mask_long, cell_long, cell_offset_long = self.create_environment(self.long_cutoff, natoms)  # For long range forces
            self.parameters.neighbor_indices = neighbor_indices    # For probing the class
            self.parameters.neighbor_mask = neighbor_mask    # For probing the class
            self.parameters.cell = cell    # For probing the class
            self.parameters.cell_offset = cell_offset    # For probing the class
                   
        else:
            neighbor_indices = self.parameters.neighbor_indices
            neighbor_indices_long = self.parameters.neighbor_indices
            neighbor_mask = self.parameters.neighbor_mask
            neighbor_mask_long = self.parameters.neighbor_mask
            cell = self.parameters.cell
            cell_long = self.parameters.cell
            cell_offset = self.parameters.cell_offset    
            cell_offset_long = self.parameters.cell_offset
            
        if self.atoms.get_pbc() is False:
            cell_offset = np.zeros((neighbor_indices.shape[0], neighbor_indices.shape[1], 3))
            cell_offset_long = np.zeros((neighbor_indices_long.shape[0], neighbor_indices_long.shape[1], 3))
        
        # torchifying the numpy data 
        positions = torch.from_numpy(positions)[None,:,:].float()
        neighbor_indices = torch.from_numpy(neighbor_indices)[None,:,:]
        neighbor_indices_long = torch.from_numpy(neighbor_indices_long)[None,:,:]
        neighbor_mask = torch.from_numpy(neighbor_mask)[None,:,:]
        neighbor_mask_long= torch.from_numpy(neighbor_mask_long)[None,:,:]
        cell = torch.from_numpy(cell)[None,:,:].float()
        cell_long = torch.from_numpy(cell_long)[None,:,:].float()
        cell_offset = torch.from_numpy(cell_offset)[None,:,:,:].float()
        cell_offset_long = torch.from_numpy(cell_offset_long)[None,:,:,:].float()
        energies = torch.zeros((1, natoms)).float()
        forces = torch.zeros((1, natoms, 3)).float()

        # Enabling calculating gradients for necessary variables
        positions.requires_grad = True
        energies.requires_grad = True
        forces.requires_grad = True
        
        rr_ij, rr_ij_vec = atom_distances(positions, neighbor_indices, cell=cell, cell_offsets=cell_offset, return_vecs = True)
        rr_ij_long, rr_ij_long_vec = atom_distances(positions, neighbor_indices_long, cell=cell_long, cell_offsets=cell_offset_long, return_vecs = True)  
            
        self.create_neighbor_matrices(atoms, neighbor_indices, neighbor_indices_long)
        energies = self.calc_pairwise_short_energy(rr_ij, neighbor_mask) + self.calc_pairwise_long_energy(rr_ij_long, neighbor_mask_long)
           
        energy = energies.sum()
        forces = -grad(energy, positions, grad_outputs=torch.ones_like(energy), retain_graph=True,)[0].squeeze()

        self.results['energy'] = energy.detach().cpu().numpy()    # eV
        self.results['energies'] = energies.detach().cpu().numpy()    # eV
        self.results['forces'] = forces.detach().cpu().numpy()    # eV/Ao
     

    def create_neighbor_matrices(self, atoms, neighbor_indices, neighbor_indices_long):
        '''
        Creates the constant neighbor valued matrices required to calculate the Pedone Potential. It calculates:
        
        D_matrix (torch.Tensor 1 X natoms X n_max_neigh) : The constant value D_ij
        a_matrix (torch.Tensor 1 X natoms X n_max_neigh) : The constant value a_ij
        C_matrix (torch.Tensor 1 X natoms X n_max_neigh) : The constant value C_ij
                
        Args:
            
            atoms (Ase object) : The input atoms object
            neighbor_indices (torch.Tensor 1 X natoms X n_max_neigh) : The indices of the neighbors for all the atoms
            neighbor_indices_long (torch.Tensor 1 X natoms X n_max_neigh) : The indices of the neighbors for all the atoms for long range interactions
        '''
        atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype = torch.float32)    # Size : (natoms)
        
        indicator_matrix = atomic_numbers[neighbor_indices[0]] + atomic_numbers[:, None]
        SiO_indicator = (indicator_matrix == self.Si_atomic_number + self.O_atomic_number)
        OO_indicator = (indicator_matrix == self.O_atomic_number + self.O_atomic_number)
        HfO_indicator = (indicator_matrix == self.Hf_atomic_number + self.O_atomic_number)
                
        D_matrix = torch.zeros_like(indicator_matrix)
        D_matrix[SiO_indicator] = self.D_SiO
        D_matrix[OO_indicator] = self.D_OO
        D_matrix[HfO_indicator] = self.D_HfO
        
        a_matrix = torch.zeros_like(indicator_matrix)
        a_matrix[SiO_indicator] = self.a_SiO
        a_matrix[OO_indicator] = self.a_OO
        a_matrix[HfO_indicator] = self.a_HfO
        
        C_matrix = torch.zeros_like(indicator_matrix)
        C_matrix[SiO_indicator] = self.C_SiO
        C_matrix[OO_indicator] = self.C_OO
        C_matrix[HfO_indicator] = self.C_HfO
        
        ro_matrix = torch.zeros_like(indicator_matrix)
        ro_matrix[SiO_indicator] = self.ro_SiO
        ro_matrix[OO_indicator] = self.ro_OO
        ro_matrix[HfO_indicator] = self.ro_HfO
        
        atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype = torch.float32)    # Size : (natoms)
        
        indicator_matrix = atomic_numbers[neighbor_indices_long[0]] + atomic_numbers[:, None]
        SiO_indicator = (indicator_matrix == self.Si_atomic_number + self.O_atomic_number)
        OO_indicator = (indicator_matrix == self.O_atomic_number + self.O_atomic_number)
        HfO_indicator = (indicator_matrix == self.Hf_atomic_number + self.O_atomic_number)
        SiSi_indicator = (indicator_matrix == self.Si_atomic_number + self.Si_atomic_number) 
        HfHf_indicator = (indicator_matrix == self.Hf_atomic_number + self.Hf_atomic_number) 
        SiHf_indicator = (indicator_matrix == self.Si_atomic_number + self.Hf_atomic_number)
        
        ZiZj_matrix = torch.zeros_like(indicator_matrix)
        ZiZj_matrix[SiO_indicator] = self.charge_Si * self.charge_O
        ZiZj_matrix[OO_indicator] = self.charge_O * self.charge_O
        ZiZj_matrix[HfO_indicator] = self.charge_Hf * self.charge_O
        ZiZj_matrix[SiSi_indicator] = self.charge_Si * self.charge_Si
        ZiZj_matrix[HfHf_indicator] = self.charge_Hf * self.charge_Hf
        ZiZj_matrix[SiHf_indicator] = self.charge_Si * self.charge_Hf
        
        self.D_matrix = D_matrix[None, :, :]
        self.a_matrix = a_matrix[None, :, :]
        self.C_matrix = C_matrix[None, :, :]
        self.ro_matrix = ro_matrix[None, :, :]
        self.ZiZj_matrix = ZiZj_matrix[None, :, :]
    
     
    def calc_pairwise_short_energy(self, rr_ij, neighbor_mask):
        r'''
        Calculates the pairwise energy with short attractions for the Pedone potential
        Upair(r_ij) = Dij[ {1 - exp(-aij (r - r0))} ^2 - 1] + Cij/r^12
        
        Args:
          
            rr_ij (torch.tensor of float) : The distances to the neighbors of the cental atom
            neighbor_mask (torch.tensor) : mask which removes the non exitsent neigbhors from neighbor indices
            
        Returns:
          
            U_pairwise_short (torch.tensor) : The pairwise energy of every atom in the atoms Ase object. Need to sum to getthe total radial potential energy
        
        '''
        bond_dissociation = self.D_matrix * (torch.pow(1 - torch.exp(-self.a_matrix * (rr_ij - self.ro_matrix)), 2) - 1)
        
        short_repulsion = torch.zeros_like(bond_dissociation)
        indicator = (neighbor_mask == True)
        short_repulsion[indicator] = self.C_matrix[indicator] / torch.pow(rr_ij[indicator]  + 1e-12, 12)
        
        U_pairwise_short = 1/2*torch.sum((bond_dissociation + short_repulsion) * neighbor_mask, dim = 2)
        
        return U_pairwise_short
        
    def calc_pairwise_long_energy(self, rr_ij, neighbor_mask):
        r'''
        Calculates the electrostatic pairwise energy for the Pedone potential
        Upair(r_ij) = ZiZj e^2/r
        
        Args:
          
            rr_ij (torch.tensor of float) : The distances to the neighbors of the cental atom
            neighbor_mask (torch.tensor) : mask which removes the non exitsent neigbhors from neighbor indices
            
        Returns:
          
            U_pairwise_long (torch.tensor) : The pairwise energy of every atom in the atoms Ase object. Need to sum to get the total radial potential energy
        
        '''
        sigma = 2
        
        indicator = (neighbor_mask == True)
        short_electrostatic = torch.zeros_like(indicator, dtype = torch.float32)
        long_electrostatic = torch.zeros_like(indicator, dtype = torch.float32)
        
        short_electrostatic[indicator] = self.ZiZj_matrix[indicator] / (rr_ij[indicator] + 1e-12) * torch.erfc(rr_ij[indicator] / (np.sqrt(2) * sigma))
        long_electrostatic[indicator] = self.ZiZj_matrix[indicator] / (rr_ij[indicator] + 1e-12) * torch.erf(rr_ij[indicator] / (np.sqrt(2) * sigma))
        
        U_pairwise_long = 1/2*torch.sum((short_electrostatic + 0*long_electrostatic) * neighbor_mask, dim = 2)
        
        return U_pairwise_long
        
         
     
    def create_environment(self, rc, natoms):
        r'''
        Creates the neighbor environment for all atoms.
        
        Args:
           
           rc (float) : The value of cutoff within which neighbors are found
           natoms (int) : The number of atoms in the structure
           
        Returns:
         
           neighborhood_idx (np.array): The neighbor indices for all the atoms 
           mask (np.array) : Mask to remove the non existent neighbors
           cell (np.array) : The basis of the unit cell of the structure
           offset (np.array) : Offset for all the neighbors to calculate the distance vectors
        ''' 
        
        idx_i, idx_j, idx_S = neighbor_list(
            "ijS", self.atoms, rc, self_interaction=False
        )
        if idx_i.shape[0] > 0:
            uidx, n_nbh = np.unique(idx_i, return_counts=True)
            n_max_nbh = np.max(n_nbh)

            n_nbh = np.tile(n_nbh[:, np.newaxis], (1, n_max_nbh))
            nbh_range = np.tile(
                np.arange(n_max_nbh, dtype=np.int)[np.newaxis], (n_nbh.shape[0], 1)
            )

            mask = np.zeros((natoms, np.max(n_max_nbh)), dtype=np.bool)
            mask[uidx, :] = nbh_range < n_nbh
            neighborhood_idx = -np.ones((natoms, np.max(n_max_nbh)), dtype=np.int64)
            neighborhood_idx[mask] = idx_j
            
            offset = np.zeros((natoms, np.max(n_max_nbh), 3), dtype=np.float32)
            offset[mask] = idx_S
            
        else:
            neighborhood_idx = -np.ones((natoms, 1), dtype=np.int64)
            mask = np.zeros((natoms, 1), dtype=np.int64)
            offset = np.zeros((natoms, 1, 3), dtype=np.float32)
        
        cell = np.array(self.atoms.get_cell())
        
        return neighborhood_idx, mask, cell, offset

 

class Heterogenous_Stillinger_Weber(Calculator):
    r'''
    Class for calculating the stillinger weber energy of the molecule containing oxygen and silicon atoms. The formula is given by:
    Upair(r_ij) = A_ij * exp(-r_ij/beta_ij) + Z_iZ_j/r_ij * erfc(r_ij/beta_ij)
    Utrip(r_ij, r_ik) = \lambda_i*h_\gamma_i(r_ij)*h_\gamma(r_ik)(cos \theta_jik + cos \theta^c_jik)^2
    h_\delta(r) = e^(\delta/(r - rc)) r < rc
    
    rc is the cutoff value
    '''

    implemented_properties = ['energy', 'energies', 'forces']
    default_parameters = {
        'ergs_to_ev':6.242*1e11,   # ev/ergs
        'angstrom_to_cm' : 1e-8 ,   # cm/A
        'electronic_charge': 4.80325 * 1e-10,    # ESU
        'A_SiO': 3.00 * 1e-9,    # ergs
        'A_OO': 1.10 * 1e-9,    # ergs
        'A_SiSi': 1.88 * 1e-9,    # ergs
        'beta_SiO': 2.29,    # Angstrom (A)
        'beta_OO': 2.34,    # A
        'beta_SiSi': 2.34,    # A
        'lambda_Si': 18.0 * 1e-11,    # ergs
        'lambda_O':  0.3 * 1e-11,    # ergs
        'gamma_Si' : 2.6,    # A
        'gamma_O' : 2.0,    # A 
        'rc_Si': 3.0,    # A
        'rc_O' : 2.6,    # A
        'costhetac_Si' : -1/3,
        'costhetac_O' : -1/3, 
        'rho' : 0.29,    # A
        'global_radial_cutoff' : 6.0,    #A
        'global_angular_cutoff' : 3.0,    #A
        'neighbor_indices': None,
        'neighbor_mask': None,
        'cell_offset': None,
        'cell': None,
        'use_environment': True
    }
    nolabel = True

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)
        
        self.Si_atomic_number = 14.0
        self.O_atomic_number = 8.0
        self.Si_ionic_charge = 4.0
        self.O_ionic_charge = -2.0
        
        self.A_SiO = self.parameters.A_SiO
        self.A_OO = self.parameters.A_OO
        self.A_SiSi = self.parameters.A_SiSi
        
        self.beta_SiO = self.parameters.beta_SiO
        self.beta_OO = self.parameters.beta_OO
        self.beta_SiSi = self.parameters.beta_SiSi
        
        self.lambda_Si = self.parameters.lambda_Si
        self.lambda_O = self.parameters.lambda_O
        
        self.gamma_Si = self.parameters.gamma_Si
        self.gamma_O = self.parameters.gamma_O
        
        self.rc_Si = self.parameters.rc_Si
        self.rc_O = self.parameters.rc_O
        
        self.rho = self.parameters.rho
        self.global_radial_cutoff = self.parameters.global_radial_cutoff
        self.global_angular_cutoff = self.parameters.global_angular_cutoff
        
        self.costhetac_Si = self.parameters.costhetac_Si
        self.costhetac_O = self.parameters.costhetac_O

        self.A_matrix = None    # 'A' value for the radial part of the potential 
        self.beta_matrix = None    # '\beta' value for the radial part of the potential
        self.ZiZj_matrix = None    # 'ZiZj' the multiplication of charges for the atoms
        self.lambda_matrix = None    # 'lambda' constant depending on the atom for the potential
        self.rc_matrix = None    # 'Cutoff' value for the neighborhood for different atoms

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
    
        r'''
        Calculates the total potential energy, local energies and forces with the Heterogenous Stillinger Weber potential
        
        a) local energies for all atom in the atoms
        b) total potential energy
        c) forces on each atom
        
        Args:
            atoms (Ase atoms object): The input atomic structure
            properties (Optional): Some parameter required by Ase calculator
            system_changes (Optional): Some parameter required by Ase calculator

        
        '''
    
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)
        
        positions = self.atoms.positions
        natoms = len(self.atoms)
        self.create_atom_matrices(atoms)
        
        if self.parameters.use_environment is True:
            
            neighbor_indices, neighbor_mask, cell, cell_offset = self.create_environment(self.global_radial_cutoff, natoms)
            neighbor_indices_angular, neighbor_mask_angular, cell_angular, cell_offset_angular = self.create_environment(self.global_angular_cutoff, natoms)    # These neighbors used for calculating the 3 body term only
            self.parameters.neighbor_indices = neighbor_indices    # For probing the class
            self.parameters.neighbor_mask = neighbor_mask    # For probing the class
            self.parameters.cell = cell    # For probing the class
            self.parameters.cell_offset = cell_offset    # For probing the class
                   
        else:
            neighbor_indices = self.parameters.neighbor_indices
            neighbor_indices_angular = self.parameters.neighbor_indices
            neighbor_mask = self.parameters.neighbor_mask
            neighbor_mask_angular = self.parameters.neighbor_mask
            cell = self.parameters.cell
            cell_angular = self.parameters.cell
            cell_offset = self.parameters.cell_offset    
            cell_offset_angular = self.parameters.cell_offset
            
        if self.atoms.get_pbc() is False:
            cell_offset = np.zeros((neighbor_indices.shape[0], neighbor_indices.shape[1], 3))
        
        nbh_idx_j, nbh_idx_k, offset_idx_j, offset_idx_k = collect_atom_triples(neighbor_indices_angular)
            
        # torchifying the numpy data 
        positions = torch.from_numpy(positions)[None,:,:].float()
        neighbor_indices = torch.from_numpy(neighbor_indices)[None,:,:]
        neighbor_indices_angular = torch.from_numpy(neighbor_indices_angular)[None,:,:]
        nbh_idx_j = torch.from_numpy(nbh_idx_j)[None,:,:]
        nbh_idx_k = torch.from_numpy(nbh_idx_k)[None,:,:]
        offset_idx_j = torch.from_numpy(offset_idx_j)[None,:,:]
        offset_idx_k = torch.from_numpy(offset_idx_k)[None,:,:]
        neighbor_mask = torch.from_numpy(neighbor_mask)[None,:,:]
        neighbor_mask_angular = torch.from_numpy(neighbor_mask_angular)[None,:,:]
        cell = torch.from_numpy(cell)[None,:,:].float()
        cell_angular = torch.from_numpy(cell_angular)[None,:,:].float()
        cell_offset = torch.from_numpy(cell_offset)[None,:,:,:].float()
        cell_offset_angular = torch.from_numpy(cell_offset_angular)[None,:,:,:].float()
        energies = torch.zeros((1, natoms)).float()
        forces = torch.zeros((1, natoms, 3)).float()

        # Enabling calculating gradients for necessary variables
        positions.requires_grad = True
        energies.requires_grad = True
        forces.requires_grad = True
        
        rr_ij, rr_ij_vec = atom_distances(positions, neighbor_indices, cell=cell, cell_offsets=cell_offset, return_vecs = True)
        R_ij, R_ik, R_jk = triple_distances(positions, nbh_idx_j, nbh_idx_k, offset_idx_j, offset_idx_k, cell=cell_angular, cell_offsets=cell_offset_angular, return_vecs=True)

        nbatch, natoms, nneigh = neighbor_indices_angular.shape
        triu_idx_row, triu_idx_col = np.triu_indices(nneigh, k=1)
        triu_idx_flat = triu_idx_row * nneigh + triu_idx_col
        
        neighbor_mask_ij = neighbor_mask_angular.repeat(1, 1, nneigh) 
        neighbor_mask_ij = neighbor_mask_ij[:,:,triu_idx_flat]
        neighbor_mask_ik = neighbor_mask_angular.repeat_interleave(nneigh, dim=2).reshape((1, natoms, -1))
        neighbor_mask_ik = neighbor_mask_ik[:,:,triu_idx_flat]
        
        self.create_neighbor_matrices(atoms, neighbor_indices)
        energies = (self.calc_pairwise_energy(rr_ij, neighbor_mask) + self.calc_triplets_energy(R_ij, R_ik, neighbor_mask_ij, neighbor_mask_ik)).squeeze()*self.parameters.ergs_to_ev
             
        energy = energies.sum()
        forces = -grad(energy, positions, grad_outputs=torch.ones_like(energy), retain_graph=True,)[0].squeeze()

        self.results['energy'] = energy.detach().cpu().numpy()    # eV
        self.results['energies'] = energies.detach().cpu().numpy()    # eV
        self.results['forces'] = forces.detach().cpu().numpy()    # eV/Ao
    
    def create_atom_matrices(self, atoms):
        '''
        Creates the constant atom valued matrices required to calculate the Heterogenous Stillinger Weber Potential. It calculates:

        lambda_matrix (torch.Tensor 1 X natoms) : The lambda value for the heteregenous atoms
        gamma_matrix (torch.Tensor 1 X natoms) : The gamma value for the heterogenous atoms
        rc_matrix (torch.Tensor 1 X natoms) : The rc values for each of the different atoms 
        
        Args:
            
            atoms (Ase object) : The input atoms object
        '''
        
        
        atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype = torch.float32)    # Size : (natoms)
        
        lambda_matrix = copy.deepcopy(atomic_numbers)
        lambda_matrix[lambda_matrix == self.Si_atomic_number] = self.lambda_Si
        lambda_matrix[lambda_matrix == self.O_atomic_number] = self.lambda_O
        
        gamma_matrix = copy.deepcopy(atomic_numbers)
        gamma_matrix[gamma_matrix == self.Si_atomic_number] = self.gamma_Si
        gamma_matrix[gamma_matrix == self.O_atomic_number] = self.gamma_O
        
        rc_matrix = copy.deepcopy(atomic_numbers)
        rc_matrix[rc_matrix == self.Si_atomic_number] = self.rc_Si
        rc_matrix[rc_matrix == self.O_atomic_number] = self.rc_O
        
        self.lambda_matrix = lambda_matrix[None, :]
        self.gamma_matrix = gamma_matrix[None, :]
        self.rc_matrix = rc_matrix[None, :]
        
    
    def create_neighbor_matrices(self, atoms, neighbor_indices):
        '''
        Creates the constant neighbor valued matrices required to calculate the Heterogenous Stillinger Weber Potential. It calculates:
        
        A_matrix (torch.Tensor 1 X natoms X n_max_neigh) : The constant value A
        beta_matrix (torch.Tensor 1 X natoms X n_max_neigh) : The constant value \beta
        ZiZj_matrix (torch.Tensor 1 X natoms X n_max_neigh) : The ZiZj value for the heterogenous atoms
                
        Args:
            
            atoms (Ase object) : The input atoms object
            neighbor_indices (torch.Tensor 1 X natoms X n_max_neigh) : The indices of the neighbors for all the atoms
        '''
        atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype = torch.float32)    # Size : (natoms)
        
        A_matrix = atomic_numbers[neighbor_indices[0]] + atomic_numbers[:, None]    # Adding the atomic numbers of the central atom and its neighbors
        A_matrix[A_matrix == self.Si_atomic_number + self.O_atomic_number] = self.A_SiO
        A_matrix[A_matrix == self.O_atomic_number + self.O_atomic_number] = self.A_OO
        A_matrix[A_matrix == self.Si_atomic_number + self.Si_atomic_number] = self.A_SiSi
        
        beta_matrix = atomic_numbers[neighbor_indices[0]] + atomic_numbers[:, None] 
        beta_matrix[beta_matrix == self.Si_atomic_number + self.O_atomic_number] = self.beta_SiO
        beta_matrix[beta_matrix == self.O_atomic_number + self.O_atomic_number] = self.beta_OO
        beta_matrix[beta_matrix == self.Si_atomic_number + self.Si_atomic_number] = self.beta_SiSi
        
        ZiZj_matrix = atomic_numbers[neighbor_indices[0]] + atomic_numbers[:, None]
        ZiZj_matrix[ZiZj_matrix == self.Si_atomic_number + self.O_atomic_number] = self.Si_ionic_charge*self.O_ionic_charge
        ZiZj_matrix[ZiZj_matrix == self.O_atomic_number + self.O_atomic_number] = self.O_ionic_charge*self.O_ionic_charge
        ZiZj_matrix[ZiZj_matrix == self.Si_atomic_number + self.Si_atomic_number] = self.Si_ionic_charge*self.Si_ionic_charge
        
        self.A_matrix = A_matrix[None, :, :]
        self.beta_matrix = beta_matrix[None, :, :]
        self.ZiZj_matrix = ZiZj_matrix[None, :, :]
        
     
    def cutoff(self, constant , r_ij, r_ik, rc, neighbor_mask_ij, neighbor_mask_ik):
        '''
        The exponential cutoff function used in the stillinger weber potential for smooth decay
        h_\delta(r) = e^(\delta/(r - rc)) r < rc
        Args:
            constant (torch.tensor 1 X natoms) : The value of delta for all the variety of atoms
            rij (torch.tensor 1 X natoms X n_neigh_max) : The neighbor distances for the central atom
            rik (torch.tensor 1 X natoms X n_neigh_max) : The neighbor distances for the central atom
            rc (torch.tensor 1 X natoms) : The cutoff value
            neighbor_mask_ij (torch.tensor 1 X natoms X n_neigh_max) : mask which removes the non existent neigbhors from neighbor indices
            neighbor_mask_ik (torch.tensor 1 X natoms X n_neigh_max) : mask which removes the non existent neigbhors from neighbor indices
            
        Returns:
            value (torch.tensor 1 X natoms X n_neigh_max) : Cutoff values
        '''
        rc_matrix = rc[:, :, None].expand(-1, -1, r_ij.shape[2])
        constant_matrix = constant[:, :, None].expand(-1, -1, r_ij.shape[2])
        
        temp_index = (r_ij < rc_matrix)*(r_ik < rc_matrix)    
        value = torch.zeros_like(r_ij)
        cutoff1 = constant_matrix[temp_index]/(r_ij[temp_index] - rc_matrix[temp_index] + 1e-12)
        cutoff2 = constant_matrix[temp_index]/(r_ik[temp_index] - rc_matrix[temp_index] + 1e-12)
        
        value[temp_index] = torch.exp(cutoff1 + cutoff2)*neighbor_mask_ij[temp_index]*neighbor_mask_ik[temp_index]
                
        return value   # Adding a small constant for when r = a. r cannot be greater than a as it is taken care of when finding out the environment around the atom
     
             
    def calc_pairwise_energy(self, rr_ij, neighbor_mask):
        r'''
        Calculates the pairwise energy for the Heterogenous Stillinger Weber potential
        Upair(r_ij) = A_ij * exp(-r_ij/beta_ij) + Z_iZ_j/r_ij * erfc(r_ij/beta_ij)
        
        Args:
          
            rr_ij (torch.tensor of float 1 X natoms X n_neigh_max) : The distances to the neighbors of the cental atom
            neighbor_mask (torch.tensor 1 X natoms X n_neigh_max) : mask which removes the non exitsent neigbhors from neighbor indices
            
        Returns:
          
            U_pairwise (torch.tensor 1 X natoms) : The pairwise energy of every atom in the atoms Ase object. 
                                                   Need to sum to get the total radial potential energy
        
        '''
        
        electronic_charge = self.parameters.electronic_charge
        angstrom_to_cm = self.parameters.angstrom_to_cm
        
        repulsive_part = self.A_matrix*torch.exp(-rr_ij/self.rho)
        attractive_part = self.ZiZj_matrix*(electronic_charge**2)/((rr_ij + 1e-12)*angstrom_to_cm)*torch.erfc(rr_ij/self.beta_matrix)
        
        U_pairwise = 1/2*torch.sum((repulsive_part + attractive_part)*neighbor_mask, dim = 2)
 
        return U_pairwise
        
    def calc_triplets_energy(self, R_ij, R_ik, neighbor_mask_ij, neighbor_mask_ik):
        r'''
        Calculates the triplet energy for the Stillinger Weber potential
        Utrip(r_ij, r_ik) = \lambda*h_\gamma(r_ij)*h_\gamma(r_ik)(cos \theta_jik - cos \theta^c_jik)^2
        
        Args:
            
            R_ij (torch.tensor 1 X natoms X n_max_angles) : The distance vectors to the first neighbor
            R_ik (torch.tensor 1 X natoms X n_max_angles) : The distance vectors to the second neighbor
            neighbor_mask_ij (torch.tensor 1 X natoms X n_max_angles) : The neighbor mask applied for the first neighbors
            neighbor_mask_ik (torch.tensor 1 X natoms X n_max_angles) : The neighbor mask applied for the second neighbors
            
        Returns:
        
            U_triplet (torch.tensor 1 X natoms) : The triplet local energies for all the atoms. Need to sum it up to get the total angular energy
            
        '''
        
        r_ij = torch.norm(R_ij, dim=3)
        r_ik = torch.norm(R_ik, dim=3)
        costheta = torch.sum(R_ij*R_ik, dim=3)/(r_ij*r_ik + 1e-12)
        
        cutoff = self.cutoff(self.gamma_matrix, r_ij, r_ik, self.rc_matrix, neighbor_mask_ij, neighbor_mask_ik)      
        ''' Assumed that every angle wants to be a tetrahedron irrespective of the type of atom '''
        U_triplet = torch.sum(self.lambda_matrix[:, :, None]*torch.pow(costheta + 1/3, 2)*cutoff, dim = 2)     
        
        
        return U_triplet
     
     
    def create_environment(self, rc, natoms):
        r'''
        Creates the neighbor environment for all atoms.
        
        Args:
           
           rc (float) : The value of cutoff within which neighbors are found
           natoms (int) : The number of atoms in the structure
           
        Returns:
         
           neighborhood_idx (np.array): The neighbor indices for all the atoms 
           mask (np.array) : Mask to remove the non existent neighbors
           cell (np.array) : The basis of the unit cell of the structure
           offset (np.array) : Offset for all the neighbors to calculate the distance vectors
        '''

        idx_i, idx_j, idx_S = neighbor_list(
            "ijS", self.atoms, rc, self_interaction=False
        )
        if idx_i.shape[0] > 0:
            uidx, n_nbh = np.unique(idx_i, return_counts=True)
            n_max_nbh = np.max(n_nbh)

            n_nbh = np.tile(n_nbh[:, np.newaxis], (1, n_max_nbh))
            nbh_range = np.tile(
                np.arange(n_max_nbh, dtype=np.int)[np.newaxis], (n_nbh.shape[0], 1)
            )

            mask = np.zeros((natoms, np.max(n_max_nbh)), dtype=np.bool)
            mask[uidx, :] = nbh_range < n_nbh
            neighborhood_idx = -np.ones((natoms, np.max(n_max_nbh)), dtype=np.int64)
            neighborhood_idx[mask] = idx_j
            
            offset = np.zeros((natoms, np.max(n_max_nbh), 3), dtype=np.float32)
            offset[mask] = idx_S
            
        else:
            neighborhood_idx = -np.ones((natoms, 1), dtype=np.int64)
            mask = np.zeros((natoms, 1), dtype=np.int64)
            offset = np.zeros((natoms, 1, 3), dtype=np.float32)
        
        cell = np.array(self.atoms.get_cell())
        
        return neighborhood_idx, mask, cell, offset
       

                    
class SiO2_Stillinger_Weber(Calculator):
    r'''
    Class for calculating the stillinger weber energy of the molecule containing oxygen and silicon atoms. The formula is given by:
    Upair(r_ij) = g_ij * A_ij * (B_ij * r_ij^-p_ij - r_ij^-q_ij) * exp((r_ij - a_ij)^-1)
    g_ij and h_1 are defined in the powerpoint
    
    Utrip(r_ij, r_ik) = \lambda_i*exp(\gamma_i^ij/(r_ij - a_i^ij) + \gamma_i^{ik}/(r_ik - a_i^ik))(cos \theta_jik + cos \theta^c_jik)^2
    h_\delta(r) = e^(\delta/(r - rc)) r < rc
    
    rc is the cutoff value
    '''

    implemented_properties = ['energy', 'energies', 'forces']
    default_parameters = {
        'kcalmol_to_ev': 0.043,   # ev/kcalpermol
        'lengthunit_to_angstrom' : 2.0951 ,   # A/length_unit
        'angstrom_to_lengthunit' : 1/2.0951,    # length_unit/A
        'A_SiSi': 7.049556277,    # kcal/mol
        'B_SiSi': 0.6022245584,    # kcal/mol
        'p_SiSi': 4,    
        'q_SiSi': 0,
        'a_SiSi': 1.8,
        'A_SiO': 115.364065913,
        'B_SiO': 0.9094442793,
        'p_SiO': 2.58759,
        'q_SiO': 2.39370,
        'a_SiO': 1.4,
        'A_OO': -12.292427744,
        'B_OO': 0,
        'p_OO': 0,
        'q_OO': 2.24432,
        'a_OO': 1.25,
        'lambda_SiSiSi': 16.404,
        'gamma_SiSiSi': 1.0473,
        'a_SiSiSi': 1.8,
        'costhetac_SiSiSi': -1/3,
        'lambda_SiSiO': 10.667,
        'gammaSiSi_SiSiO': 1.93973,
        'gammaSiO_SiSiO': 0.25,
        'aSiSi_SiSiO': 1.9,
        'aSiO_SiSiO': 1.4,
        'costhetac_SiSiO': -1/3,
        'lambda_SiOSi': 2.9572,
        'a_SiOSi': 1.4,
        'costhetac_SiOSi': -0.6155238,
        'gamma_SiOSi': 0.71773,
        'lambda_OSiO': 3.1892,
        'gamma_OSiO': 0.3220,
        'a_OSiO': 1.65,
        'costhetac_OSiO': -1/3,
        'm1': 0.097,
        'm2': 1.6,
        'm3': 0.3654,
        'm4': 0.1344,
        'm5': 6.4176,
        'R': 1.3,
        'D': 0.1,
        'epsilon': 50,    # kcal/mol
        'global_radial_cutoff' : 1.8,    # length_unit
        'global_angular_cutoff' : 1.4,    # length_unit (Only approx 4 nearest neighbors)
        'neighbor_indices':None,
        'neighbor_mask':None,
        'cell_offset':None,
        'cell': None,
        'use_environment': True
    }
    nolabel = True

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)
        
        self.Si_atomic_number = 14.0
        self.O_atomic_number = 8.0
        
        self.A_SiSi = self.parameters.A_SiSi
        self.B_SiSi = self.parameters.B_SiSi
        self.p_SiSi = self.parameters.p_SiSi
        self.q_SiSi = self.parameters.q_SiSi
        self.a_SiSi = self.parameters.a_SiSi
        
        self.A_SiO = self.parameters.A_SiO
        self.B_SiO = self.parameters.B_SiO
        self.p_SiO = self.parameters.p_SiO
        self.q_SiO = self.parameters.q_SiO
        self.a_SiO = self.parameters.a_SiO
        
        self.A_OO = self.parameters.A_OO
        self.B_OO = self.parameters.B_OO
        self.p_OO = self.parameters.p_OO
        self.q_OO = self.parameters.q_OO
        self.a_OO = self.parameters.a_OO
        
        self.lambda_SiSiSi = self.parameters.lambda_SiSiSi
        self.gamma_SiSiSi = self.parameters.gamma_SiSiSi
        self.a_SiSiSi = self.parameters.a_SiSiSi
        self.costhetac_SiSiSi = self.parameters.costhetac_SiSiSi
        
        self.lambda_SiSiO = self.parameters.lambda_SiSiO
        self.gammaSiSi_SiSiO = self.parameters.gammaSiSi_SiSiO
        self.gammaSiO_SiSiO = self.parameters.gammaSiO_SiSiO
        self.aSiSi_SiSiO = self.parameters.aSiSi_SiSiO
        self.aSiO_SiSiO = self.parameters.aSiO_SiSiO
        self.costhetac_SiSiO = self.parameters.costhetac_SiSiO
        
        self.lambda_SiOSi = self.parameters.lambda_SiOSi
        self.a_SiOSi = self.parameters.a_SiOSi
        self.costhetac_SiOSi = self.parameters.costhetac_SiOSi
        self.gamma_SiOSi = self.parameters.gamma_SiOSi
        
        self.lambda_OSiO = self.parameters.lambda_OSiO
        self.gamma_OSiO = self.parameters.gamma_OSiO
        self.a_OSiO = self.parameters.a_OSiO
        self.costhetac_OSiO = self.parameters.costhetac_OSiO
        
        self.m1 = self.parameters.m1
        self.m2 = self.parameters.m2
        self.m3 = self.parameters.m3
        self.m4 = self.parameters.m4
        self.m5 = self.parameters.m5
        self.R = self.parameters.R
        self.D = self.parameters.D
        self.epsilon = self.parameters.epsilon
                
        self.global_radial_cutoff = self.parameters.global_radial_cutoff
        self.global_angular_cutoff = self.parameters.global_angular_cutoff
        
        self.A_matrix = None    # 'A' value for the radial part of the potential 
        self.B_matrix = None    # 'B' value for the radial part of the potential
        self.a_matrix = None    # a cutoff for the radial part of the potential
        self.g_matrix = None    # g cutoff for the radial part of the potential
        self.p_matrix = None    # p power for the radial part of the potential
        self.q_matrix = None    # q power for the radial part of the potential
        
        
        self.lambda_matrix = None    # lambda constant value for the angular part of the potential
        self.costhetac_matrix = None    # cos thetac value for the angular part of the potential
        self.j_gamma_matrix = None    # gamma^{ij} value for the angular part of the potential
        self.k_gamma_matrix = None    # gamma^{ik} value for the angular part of the potential
        self.j_a_matrix = None    # a^{ij} value for the angular part of the potential
        self.k_a_matrix = None    # a^{ik} value for the angular part of the potential
        
    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
    
        r'''
        Calculates the total potential energy, local energies and forces with the Heterogenous Stillinger Weber potential
        
        a) local energies for all atom in the atoms
        b) total potential energy
        c) forces on each atom
        
        Args:
            atoms (Ase atoms object): The input atomic structure
            properties (Optional): Some parameter required by Ase calculator
            system_changes (Optional): Some parameter required by Ase calculator

        
        '''
    
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)
        
        positions = self.atoms.positions
        natoms = len(self.atoms)
        
        if self.parameters.use_environment is True:
            
            neighbor_indices, neighbor_mask, cell, cell_offset = self.create_environment(self.global_radial_cutoff * self.parameters.lengthunit_to_angstrom, natoms)
            neighbor_indices_angular, neighbor_mask_angular, cell_angular, cell_offset_angular = self.create_environment(self.global_angular_cutoff * self.parameters.lengthunit_to_angstrom, natoms)    # These neighbors used for calculating the 3 body term only
            self.parameters.neighbor_indices = neighbor_indices    # For probing the class
            self.parameters.neighbor_mask = neighbor_mask    # For probing the class
            self.parameters.cell = cell # For probing the class
            self.parameters.cell_offset = cell_offset   # For probing the class
                   
        else:
            neighbor_indices = self.parameters.neighbor_indices
            neighbor_indices_angular = self.parameters.neighbor_indices
            neighbor_mask = self.parameters.neighbor_mask
            neighbor_mask_angular = self.parameters.neighbor_mask
            cell = self.parameters.cell
            cell_angular = self.parameters.cell
            cell_offset = self.parameters.cell_offset 
            cell_offset_angular = self.parameters.cell_offset
            
        if self.atoms.get_pbc() is False:
            cell_offset = np.zeros((neighbor_indices.shape[0], neighbor_indices.shape[1], 3))
                   
        nbh_idx_j, nbh_idx_k, offset_idx_j, offset_idx_k = collect_atom_triples(neighbor_indices_angular)
            
        # torchifying the numpy data 
        positions = torch.from_numpy(positions)[None,:,:].float() * self.parameters.angstrom_to_lengthunit
        neighbor_indices = torch.from_numpy(neighbor_indices)[None,:,:]
        neighbor_indices_angular = torch.from_numpy(neighbor_indices_angular)[None,:,:]
        nbh_idx_j = torch.from_numpy(nbh_idx_j)[None,:,:]
        nbh_idx_k = torch.from_numpy(nbh_idx_k)[None,:,:]
        offset_idx_j = torch.from_numpy(offset_idx_j)[None,:,:]
        offset_idx_k = torch.from_numpy(offset_idx_k)[None,:,:]
        neighbor_mask = torch.from_numpy(neighbor_mask)[None,:,:]
        neighbor_mask_angular = torch.from_numpy(neighbor_mask_angular)[None,:,:]
        cell = torch.from_numpy(cell)[None,:,:].float() * self.parameters.angstrom_to_lengthunit
        cell_angular = torch.from_numpy(cell_angular)[None,:,:].float() * self.parameters.angstrom_to_lengthunit
        cell_offset = torch.from_numpy(cell_offset)[None,:,:,:].float()
        cell_offset_angular = torch.from_numpy(cell_offset_angular)[None,:,:,:].float()
        energies = torch.zeros((1, natoms)).float()
        forces = torch.zeros((1, natoms, 3)).float()

        # Enabling calculating gradients for necessary variables
        positions.requires_grad = True
        energies.requires_grad = True
        forces.requires_grad = True
        
        rr_ij, rr_ij_vec = atom_distances(positions, neighbor_indices, cell=cell, cell_offsets=cell_offset, return_vecs = True)
        R_ij, R_ik, R_jk = triple_distances(positions, nbh_idx_j, nbh_idx_k, offset_idx_j, offset_idx_k, cell=cell_angular, cell_offsets=cell_offset_angular, return_vecs=True)

        nbatch, natoms, nneigh = neighbor_indices_angular.shape
        triu_idx_row, triu_idx_col = np.triu_indices(nneigh, k=1)
        triu_idx_flat = triu_idx_row * nneigh + triu_idx_col
        
        nbh_mask_ij = neighbor_mask_angular.repeat(1, 1, nneigh) 
        nbh_mask_ij = nbh_mask_ij[:,:,triu_idx_flat]
        nbh_mask_ik = neighbor_mask_angular.repeat_interleave(nneigh, dim=2).reshape((1, natoms, -1))
        nbh_mask_ik = nbh_mask_ik[:,:,triu_idx_flat]
        
        self.create_neighbor_matrices_radial(atoms, neighbor_indices, neighbor_mask, rr_ij)
        self.create_neighbor_matrices_angular(atoms, nbh_idx_j, nbh_idx_k)
        
        energies = (self.calc_pairwise_energy(rr_ij, neighbor_mask) + self.calc_triplets_energy(R_ij, R_ik, nbh_mask_ij, nbh_mask_ik)).squeeze() * self.epsilon * self.parameters.kcalmol_to_ev  
        
        energy = energies.sum()
        forces = -grad(energy, positions, grad_outputs=torch.ones_like(energy), retain_graph=True,)[0].squeeze() / self.parameters.lengthunit_to_angstrom
        forces = forces
        
        self.results['energy'] = energy.detach().cpu().numpy()    # eV
        self.results['energies'] = energies.detach().cpu().numpy()    # eV
        self.results['forces'] = forces.detach().cpu().numpy()    # eV/Ao
    
    
    def calc_g(self, atoms, z, neighbor_indices, neighbor_mask):
        '''
        Calculates the g value for the pair potential as 
        g(z) = m_1/(exp[(m_2 - z)/m_3] + 1) * exp(m_4 * (z - m_5)^2)
        
        g_ij  = g(z_i)    i = O and j = Si
        g_ij = g(z_j)    i = Si and j = O
        g_ij = 1          otherwise
        
        Args:
        
            z (torch.Tensor 1 X natoms) : The z vector calculated from fc
            neighbor_indices (torch.Tensor 1 X natoms X n_max_neigh) : The indices of the neighbors for all the atoms
            neighbor_mask (torch.tensor 1 X natoms X n_neigh_max) : mask which removes the non existent neighbors from neighbor indices
            
        Returns:
            
            g (torch.Tensor 1 X natoms X n_max_neigh) : The g matrix 
        
        '''
        
        atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype = torch.float32)    # natoms
        indicator_matrix = atomic_numbers[neighbor_indices[0]] + atomic_numbers[:, None]
        SiO_indicator = (indicator_matrix == self.Si_atomic_number + self.O_atomic_number)
        i_Si_indicator = (atomic_numbers == self.Si_atomic_number)[None, :, None].expand(-1, -1, neighbor_indices.shape[2])    # 1 X natoms X n_neigh_max
        i_O_indicator = (atomic_numbers == self.O_atomic_number)[None, :, None].expand(-1, -1, neighbor_indices.shape[2])    # 1 X natoms X n_neigh_max
        
        zi = z[:, :, None].expand(-1, -1, neighbor_indices.shape[2])    # 1 X natoms X n_neigh_max
        zj = z[:, neighbor_indices[0]]    # 1 X natoms X n_neigh_max
        
        first_index = i_O_indicator * SiO_indicator    # If i = O and j = Si
        second_index = i_Si_indicator * SiO_indicator  # If i = Si and j = O
                
        g = torch.ones_like(neighbor_indices, dtype = torch.float32) * neighbor_mask 
        g[first_index] = self.m1/(torch.exp((self.m2 - zi[first_index])/self.m3) + 1) * torch.exp(self.m4 * torch.pow(zi[first_index] - self.m5, 2)) * neighbor_mask[first_index]
        g[second_index] = self.m1/(torch.exp((self.m2 - zj[second_index])/self.m3) + 1) * torch.exp(self.m4 * torch.pow(zj[second_index] - self.m5, 2)) * neighbor_mask[second_index]
        
        return g
        
    def create_neighbor_matrices_angular(self, atoms, nbh_idx_j, nbh_idx_k):
        '''
        Creates the constant neighbor valued matrices for angular potential required to calculate the Stillinger Weber Potential. It calculates:
        
        lambda_matrix (torch.Tensor 1 X natoms X n_max_neigh) : The constant value lambda
        gamma_matrix (torch.Tensor 1 X natoms X n_max_neigh) : The constant value gamma
        a_matrix (torch.Tensor 1 X natoms X n_max_neigh) : The a_ij value for the heterogenous atoms (cutoff)
        costhetac_matrix (torch.Tensor 1 X natoms X n_max_neigh) : The costhetac value for the heterogenous atoms
                
        Args:
            
            atoms (Ase object) : The input atoms object
            nbh_idx_j (torch.Tensor 1 X natoms X n_max_neigh) : The indices of the neighbors for all the atoms (j)
            nbh_idx_k (torch.Tensor 1 X natoms X n_max_neigh) : The indices of the neighbors for all the atoms (k)
        '''
        
        atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype = torch.float32)    # Size : (natoms)
        
        indicator_j_matrix = atomic_numbers[nbh_idx_j[0]] + atomic_numbers[:, None]    # natoms X n_max_neigh
        indicator_k_matrix = atomic_numbers[nbh_idx_k[0]] + atomic_numbers[:, None]    # natoms X n_max_neigh
        
        j_SiO_indicator = (indicator_j_matrix == self.Si_atomic_number + self.O_atomic_number)
        j_OO_indicator = (indicator_j_matrix == self.O_atomic_number + self.O_atomic_number)
        j_SiSi_indicator = (indicator_j_matrix == self.Si_atomic_number + self.Si_atomic_number)
        
        k_SiO_indicator = (indicator_k_matrix == self.Si_atomic_number + self.O_atomic_number)
        k_OO_indicator = (indicator_k_matrix == self.O_atomic_number + self.O_atomic_number)
        k_SiSi_indicator = (indicator_k_matrix == self.Si_atomic_number + self.Si_atomic_number)
        
        SiSiSi_indicator = j_SiSi_indicator*k_SiSi_indicator
        SiSiO_indicator = j_SiO_indicator*k_SiSi_indicator + j_SiSi_indicator*k_SiO_indicator
        OSiO_indicator = j_SiO_indicator*k_SiO_indicator*(atomic_numbers[:, None] == self.Si_atomic_number)
        SiOSi_indicator = j_SiO_indicator*k_SiO_indicator*(atomic_numbers[:, None] == self.O_atomic_number)
        
        lambda_matrix = torch.zeros_like(indicator_j_matrix, dtype = torch.float32)
        lambda_matrix[SiSiSi_indicator] = self.lambda_SiSiSi
        lambda_matrix[SiSiO_indicator] = self.lambda_SiSiO
        lambda_matrix[OSiO_indicator] = self.lambda_OSiO
        lambda_matrix[SiOSi_indicator] = self.lambda_SiOSi
        
        costhetac_matrix = torch.zeros_like(indicator_j_matrix, dtype = torch.float32)
        costhetac_matrix[SiSiSi_indicator] = self.costhetac_SiSiSi
        costhetac_matrix[SiSiO_indicator] = self.costhetac_SiSiO
        costhetac_matrix[OSiO_indicator] = self.costhetac_OSiO
        costhetac_matrix[SiOSi_indicator] = self.costhetac_SiOSi
        
        j_gamma_matrix = torch.zeros_like(indicator_j_matrix, dtype = torch.float32)
        j_gamma_matrix[SiSiSi_indicator] = self.gamma_SiSiSi
        j_gamma_matrix[SiSiO_indicator * j_SiO_indicator] = self.gammaSiO_SiSiO
        j_gamma_matrix[SiSiO_indicator * j_SiSi_indicator] = self.gammaSiSi_SiSiO
        j_gamma_matrix[OSiO_indicator] = self.gamma_OSiO
        j_gamma_matrix[SiOSi_indicator] = self.gamma_SiOSi
        
        
        k_gamma_matrix = torch.zeros_like(indicator_j_matrix, dtype = torch.float32)
        k_gamma_matrix[SiSiSi_indicator] = self.gamma_SiSiSi
        k_gamma_matrix[SiSiO_indicator * k_SiO_indicator] = self.gammaSiO_SiSiO
        k_gamma_matrix[SiSiO_indicator * k_SiSi_indicator] = self.gammaSiSi_SiSiO
        k_gamma_matrix[OSiO_indicator] = self.gamma_OSiO
        k_gamma_matrix[SiOSi_indicator] = self.gamma_SiOSi
        
        j_a_matrix = torch.zeros_like(indicator_j_matrix, dtype = torch.float32)
        j_a_matrix[SiSiSi_indicator] = self.a_SiSiSi
        j_a_matrix[SiSiO_indicator * j_SiO_indicator] = self.aSiO_SiSiO
        j_a_matrix[SiSiO_indicator * j_SiSi_indicator] = self.aSiSi_SiSiO
        j_a_matrix[OSiO_indicator] = self.a_OSiO
        j_a_matrix[SiOSi_indicator] = self.a_SiOSi
        
        k_a_matrix = torch.zeros_like(indicator_j_matrix, dtype = torch.float32)
        k_a_matrix[SiSiSi_indicator] = self.a_SiSiSi
        k_a_matrix[SiSiO_indicator * j_SiO_indicator] = self.aSiO_SiSiO
        k_a_matrix[SiSiO_indicator * j_SiSi_indicator] = self.aSiSi_SiSiO
        k_a_matrix[OSiO_indicator] = self.a_OSiO
        k_a_matrix[SiOSi_indicator] = self.a_SiOSi
        
        self.lambda_matrix = lambda_matrix[None, :, :]
        self.costhetac_matrix = costhetac_matrix[None, :, :]
        self.j_gamma_matrix = j_gamma_matrix[None, :, :]
        self.k_gamma_matrix = k_gamma_matrix[None, :, :]
        self.j_a_matrix = j_a_matrix[None, :, :]
        self.k_a_matrix = k_a_matrix[None, :, :]
    
    
    def create_neighbor_matrices_radial(self, atoms, neighbor_indices, neighbor_mask, rr_ij):
        '''
        Creates the constant neighbor valued matrices for radial potential required to calculate the Stillinger Weber Potential. It calculates:
        
        A_matrix (torch.Tensor 1 X natoms X n_max_neigh) : The constant value A
        B_matrix (torch.Tensor 1 X natoms X n_max_neigh) : The constant value B
        a_matrix (torch.Tensor 1 X natoms X n_max_neigh) : The a_ij value for the heterogenous atoms
        g_matrix (torch.Tensor 1 X natoms X n_max_neigh) : The g_ij value for the heterogenous atoms
                
        Args:
            
            atoms (Ase object) : The input atoms object
            neighbor_indices (torch.Tensor 1 X natoms X n_max_neigh) : The indices of the neighbors for all the atoms
            neighbor_mask (torch.Tensor 1 X natoms X n_max_neigh) : mask which removes the non existent neigbhors from neighbor indices
            rr_ij (torch.Tensor 1 X natoms X n_max_neigh) : The neighbor distances for all the i atoms
        '''
        atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype = torch.float32)    # Size : (natoms)
        indicator_matrix = atomic_numbers[neighbor_indices[0]] + atomic_numbers[:, None]
        SiO_indicator = (indicator_matrix == self.Si_atomic_number + self.O_atomic_number)
        OO_indicator = (indicator_matrix == self.O_atomic_number + self.O_atomic_number)
        SiSi_indicator = (indicator_matrix == self.Si_atomic_number + self.Si_atomic_number)
        
        
        A_matrix = torch.zeros_like(indicator_matrix, dtype = torch.float32)
        A_matrix[SiO_indicator] = self.A_SiO
        A_matrix[OO_indicator] = self.A_OO
        A_matrix[SiSi_indicator] = self.A_SiSi
        
        B_matrix = torch.zeros_like(indicator_matrix, dtype = torch.float32)
        B_matrix[SiO_indicator] = self.B_SiO
        B_matrix[OO_indicator] = self.B_OO
        B_matrix[SiSi_indicator] = self.B_SiSi
        
        a_matrix = torch.zeros_like(indicator_matrix, dtype = torch.float32)
        a_matrix[SiO_indicator] = self.a_SiO
        a_matrix[OO_indicator] = self.a_OO
        a_matrix[SiSi_indicator] = self.a_SiSi
        
        p_matrix = torch.zeros_like(indicator_matrix, dtype = torch.float32)
        p_matrix[SiO_indicator] = self.p_SiO
        p_matrix[OO_indicator] = self.p_OO
        p_matrix[SiSi_indicator] = self.p_SiSi
        
        q_matrix = torch.zeros_like(indicator_matrix, dtype = torch.float32)
        q_matrix[SiO_indicator] = self.q_SiO
        q_matrix[OO_indicator] = self.q_OO
        q_matrix[SiSi_indicator] = self.q_SiSi
        
        Si_indicator = atomic_numbers[neighbor_indices] == self.Si_atomic_number
        z = self.Tersoff_cutoff(rr_ij, neighbor_mask, Si_indicator)
        self.g_matrix = self.calc_g(atoms, z, neighbor_indices, neighbor_mask)
               
        self.A_matrix = A_matrix[None, :, :]
        self.B_matrix = B_matrix[None, :, :]
        self.a_matrix = a_matrix[None, :, :]
        self.p_matrix = p_matrix[None, :, :]
        self.q_matrix = q_matrix[None, :, :]
        
    
    def Tersoff_cutoff(self, rr_ij, neighbor_mask, Si_indicator):
        '''
        It effective calculates the cordination number of its neighbors given by:
        fc(r) = 1    r < R - D
        fc(r) = 1 - (r - R + D)/2D + sin(pi(r - R + D)/D)/(2pi)    R - D <= r < R + D
        fc(r) = 0    r >= R + D
            
        Only count cordination number only if the neighbor is Si
        
        Args:
            rr_ij (torch.tensor 1 X natoms X n_neigh_max) : The neighbor distances for the central atom
            neighbor_mask (torch.tensor 1 X natoms X n_neigh_max) : mask which removes the non existent neigbhors from neighbor indices
            Si_indicator (torch.tensor 1 X natoms X n_neigh_max) : mask which indicates if the neighbor atom is Si
            
        Returns:
            
            z (torch.tensor 1 X natoms) : Effective Si cordination numbers
        '''
        
        cutoff = torch.zeros_like(rr_ij)
        first_index = rr_ij < self.R - self.D
        second_index = (rr_ij >= self.R - self.D) * (rr_ij < self.R + self.D)
        
        cutoff[first_index] = 1.0 * neighbor_mask[first_index] * Si_indicator[first_index]
        cutoff[second_index] = (1.0 - (rr_ij[second_index] - self.R + self.D)/(2 * self.D) + torch.sin(np.pi * (rr_ij[second_index] - self.R + self.D)/self.D)/(2 * np.pi))  * neighbor_mask[second_index] * Si_indicator[second_index]
        
        z = torch.sum(cutoff, dim = 2)
        
        return z
        
    
    def Exponential_cutoff(self, j_gamma_matrix, k_gamma_matrix, r_ij, r_ik, j_a_matrix, k_a_matrix, nbh_mask_ij, nbh_mask_ik):
        '''
        The exponential cutoff function used in the stillinger weber potential for smooth decay
        h_\delta(r) = e^(\delta/(r - rc)) r < rc
        Args:
            j_gamma_matrix (torch.tensor 1 X natoms X n_neigh_max) : The value of gamma for ij bonds
            k_gamma_matrix (torch.tensor 1 X natoms X n_neigh_max) : The value of gamma for ik bonds
            r_ij (torch.tensor 1 X natoms X n_neigh_max) : The neighbor distances for the central atom
            r_ik (torch.tensor 1 X natoms X n_neigh_max) : The neighbor distances for the central atom
            j_a_matrix (torch.tensor 1 X natoms X n_neigh_max) : The cutoff value for ij bonds
            k_a_matrix (torch.tensor 1 X natoms X n_neigh_max) : The cutoff value for ik bonds
            nbh_mask_ij (torch.tensor 1 X natoms X n_neigh_max) : mask which removes the non existent neigbhors from neighbor indices (ij)
            nbh_mask_ik (torch.tensor 1 X natoms X n_neigh_max) : mask which removes the non existent neigbhors from neighbor indices (ik)
            
        Returns:
            cutoff (torch.tensor 1 X natoms X n_neigh_max) : Cutoff values
        '''
               
        temp_index = (r_ij < j_a_matrix)*(r_ik < k_a_matrix)    
        cutoff = torch.zeros_like(r_ij)
        value1 = j_gamma_matrix[temp_index]/(r_ij[temp_index] - j_a_matrix[temp_index] + 1e-12)
        value2 = k_gamma_matrix[temp_index]/(r_ik[temp_index] - k_a_matrix[temp_index] + 1e-12)
        
        cutoff[temp_index] = torch.exp(value1 + value2) * nbh_mask_ij[temp_index] * nbh_mask_ik[temp_index]
                
        return cutoff
     
             
    def calc_pairwise_energy(self, rr_ij, neighbor_mask):
        r'''
        Calculates the pairwise energy for the Heterogenous Stillinger Weber potential
        Upair(r_ij) = A_ij * exp(-r_ij/beta_ij) + Z_iZ_j/r_ij * erfc(r_ij/beta_ij)
        
        Args:
          
            rr_ij (torch.tensor of float 1 X natoms X n_neigh_max) : The distances to the neighbors of the cental atom
            neighbor_mask (torch.tensor 1 X natoms X n_neigh_max) : mask which removes the non exitsent neigbhors from neighbor indices
            
        Returns:
          
            U_pairwise (torch.tensor 1 X natoms) : The pairwise energy of every atom in the atoms Ase object. 
                                                   Need to sum to get the total radial potential energy
        
        '''
        
        value = torch.zeros_like(neighbor_mask, dtype = torch.float32)
        indicator = (neighbor_mask == True) * (rr_ij < self.a_matrix)
        
        value[indicator] = self.g_matrix[indicator] * self.A_matrix[indicator] * (self.B_matrix[indicator] * torch.pow(rr_ij[indicator], -self.p_matrix[indicator]) - torch.pow(rr_ij[indicator], -self.q_matrix[indicator])) * torch.exp(torch.pow(rr_ij[indicator] - self.a_matrix[indicator], -1))
           
        U_pairwise = 1/2*torch.sum(value * neighbor_mask, dim = 2)
        
        return U_pairwise
        
    def calc_triplets_energy(self, R_ij, R_ik, nbh_mask_ij, nbh_mask_ik):
        r'''
        Calculates the triplet energy for the Stillinger Weber potential
        Utrip(r_ij, r_ik) = \lambda*h_\gamma(r_ij)*h_\gamma(r_ik)(cos \theta_jik - cos \theta^c_jik)^2
        
        Args:
            
            R_ij (torch.tensor 1 X natoms X n_max_angles) : The distance vectors to the first neighbor
            R_ik (torch.tensor 1 X natoms X n_max_angles) : The distance vectors to the second neighbor
            nbh_mask_ij (torch.tensor 1 X natoms X n_max_angles) : The neighbor mask applied for the first neighbors
            nbh_mask_ik (torch.tensor 1 X natoms X n_max_angles) : The neighbor mask applied for the second neighbors
            
        Returns:
        
            U_triplet (torch.tensor 1 X natoms) : The triplet local energies for all the atoms. Need to sum it up to get the total angular energy
            
        '''
        
        r_ij = torch.norm(R_ij, dim=3)
        r_ik = torch.norm(R_ik, dim=3)
        costheta = torch.sum(R_ij*R_ik, dim=3)/(r_ij*r_ik + 1e-12)
        
        cutoff = self.Exponential_cutoff(self.j_gamma_matrix, self.k_gamma_matrix, r_ij, r_ik, self.j_a_matrix, self.k_a_matrix, nbh_mask_ij, nbh_mask_ik)      
        ''' Assumed that every angle wants to be a tetrahedron irrespective of the type of atom '''
        U_triplet = torch.sum(self.lambda_matrix * torch.pow(costheta - self.costhetac_matrix, 2) * cutoff, dim = 2)
        
        return U_triplet
     
     
    def create_environment(self, rc, natoms):
        r'''
        Creates the neighbor environment for all atoms.
        
        Args:
           
           rc (float) : The value of cutoff within which neighbors are found
           natoms (int) : The number of atoms in the structure
           
        Returns:
         
           neighborhood_idx (np.array): The neighbor indices for all the atoms 
           mask (np.array) : Mask to remove the non existent neighbors
           cell (np.array) : The basis of the unit cell of the structure
           offset (np.array) : Offset for all the neighbors to calculate the distance vectors
        '''

        idx_i, idx_j, idx_S = neighbor_list(
            "ijS", self.atoms, rc, self_interaction=False
        )
        if idx_i.shape[0] > 0:
            uidx, n_nbh = np.unique(idx_i, return_counts=True)
            n_max_nbh = np.max(n_nbh)

            n_nbh = np.tile(n_nbh[:, np.newaxis], (1, n_max_nbh))
            nbh_range = np.tile(
                np.arange(n_max_nbh, dtype=np.int)[np.newaxis], (n_nbh.shape[0], 1)
            )

            mask = np.zeros((natoms, np.max(n_max_nbh)), dtype=np.bool)
            mask[uidx, :] = nbh_range < n_nbh
            neighborhood_idx = -np.ones((natoms, np.max(n_max_nbh)), dtype=np.int64)
            neighborhood_idx[mask] = idx_j
            
            offset = np.zeros((natoms, np.max(n_max_nbh), 3), dtype=np.float32)
            offset[mask] = idx_S
            
        else:
            neighborhood_idx = -np.ones((natoms, 1), dtype=np.int64)
            mask = np.zeros((natoms, 1), dtype=np.int64)
            offset = np.zeros((natoms, 1, 3), dtype=np.float32)
        
        cell = np.array(self.atoms.get_cell())
        
        return neighborhood_idx, mask, cell, offset
        

class Tight_Binding_Silicon(Calculator):   
    r'''
    Class to calculate the NRL tight binding Hamitlonian and Overlap for the Silicon Hamiltonian
    The NRL TB Hamiltonian is defined in https://link.springer.com/content/pdf/10.1557/PROC-491-221.pdf
    In the original paper energy was calculated in rydbers and distances in bohrs
    
    In this code energy is calculated in ev and distances in angstroms.
    
    Args:
    
        environment_provider (Optional ): The schnet encironment provider which gives the neighbor indices and offset for all atoms
                                          If not provided then Ase environment is assumed
    
    '''     

    implemented_properties = ['energy', 'energies', 'forces']
    default_parameters = {
        'bohr_to_angstrom': 0.529177,    # Angstrom/Bohr
        'ryd_to_ev': 13.6056980,    # electron-volt/Rydberg
        'angstrom_to_bohr': 1/0.529177,    # Bohr/Angstrom
        'ev_to_ryd': 1/13.6056980,   # Rydberg/ev
        'rc': 12.5,           # Cutoff in bohr
        'structure': 'diamond',    # Structure of the unit cell of Si
    }
    
    def __init__(self, environment_provider = None, **kwargs):
        Calculator.__init__(self, **kwargs)
        
        if environment_provider is None:
            cutoff = self.parameters.rc*self.parameters.bohr_to_angstrom
            self.environment_provider = AseEnvironmentProvider(cutoff) 
        else:
            self.environment_provider = environment_provider
            
        self.Hamilt = None
        self.Overlap = None
        self.eigvals = None
        
    def generate_Hamiltonian_submatrix(self, l, m, n, rc):
        r'''
        Calcultes the Hamiltonian submatric between two atoms where the cosine distance values are given by l, m, n
        
        Args:
            l (float) : cosine distance along x
            m (float) : cosine distance along y
            n (float) : cosine distance along z
            rc (float) : cutoff value 
        
        '''
        ss = E_ss(l, m, n, rc)
        sx = E_sx(l, m, n, rc)
        sy = E_sy(l, m, n, rc)
        sz = E_sz(l, m, n, rc)
        xx = E_xx(l, m, n, rc)
        yy = E_yy(l, m, n, rc)
        zz = E_zz(l, m, n, rc)
        xy = E_xy(l, m, n, rc)
        yz = E_yz(l, m, n, rc)
        xz = E_xz(l, m, n ,rc)
      
        submatrix = \
        np.array([[         ss,            sx,              sy,             sz],\
                  [        -sx,            xx,              xy,             xz],\
                  [        -sy,            xy,              yy,             yz],\
                  [        -sz,            xz,              yz,             zz]    
          ])
        
        return submatrix

    def generate_overlap_submatrix(self, l, m, n, rc):
        r'''
        Calcultes the Overlap submatric between two atoms where the cosine distance values are given by l, m, n
        
        Args:
            l (float) : cosine distance along x
            m (float) : cosine distance along y
            n (float) : cosine distance along z
            rc (float) : cutoff value 
        
        '''
        ss = S_ss(l, m, n, rc)
        sx = S_sx(l, m, n, rc)
        sy = S_sy(l, m, n, rc)
        sz = S_sz(l, m, n, rc)
        xx = S_xx(l, m, n, rc)
        yy = S_yy(l, m, n, rc)
        zz = S_zz(l, m, n, rc)
        xy = S_xy(l, m, n, rc)
        yz = S_yz(l, m, n, rc)
        xz = S_xz(l, m, n ,rc)
      
        submatrix = \
        np.array([[         ss,            sx,              sy,             sz],\
                  [        -sx,            xx,              xy,             xz],\
                  [        -sy,            xy,              yy,             yz],\
                  [        -sz,            xz,              yz,             zz]    
          ])
        
        return submatrix

    def generate_real_space_hamiltonian(self, atoms = None):
        r'''
        Calculates the total real space TB Hamiltonian for the atoms Ase object
                     
        Args:
            atoms (atoms Ase object) : the input atomic structure. Constants used are for silicon
        
        '''
        
        natoms = len(atoms)
        Hamilt = np.zeros((4*natoms, 4*natoms))
        Overlap = np.zeros((4*natoms, 4*natoms))
        neighbor_indices, offsets = self.environment_provider.get_environment(atoms)
        
        
        for index in range(natoms):
            
            neighbor_index = neighbor_indices[index, :]
            mask = (neighbor_index != -1)    #  mask to remove the non-existent neighbors
            neighbor_index = neighbor_index[mask].astype(np.int32)

            R = (atoms.positions[neighbor_index, :] + offsets[index, mask, :]@atoms.get_cell() - atoms.positions[index, :])    # Calculate distances in Angstrom
            R = R*self.parameters.angstrom_to_bohr
            neighbors_distance = np.sqrt(np.sum(R**2, axis = 1))
            h_is, h_ip = h_il(neighbors_distance, self.parameters.rc)
            Hamilt[4*index:4*(index + 1), 4*index:4*(index + 1)] = np.array([[h_is, 0, 0, 0], [0, h_ip, 0, 0], [0, 0, h_ip, 0], [0 ,0 ,0 ,h_ip]])
            Overlap[4*index:4*(index + 1), 4*index:4*(index + 1)] = np.eye(4)
        
            for ii, neighbor in enumerate(neighbor_index):       
                l, m, n = R[ii]
                Hamilt[4*index:4*(index + 1), 4*neighbor:4*(neighbor + 1)] = self.generate_Hamiltonian_submatrix(l, m, n, rc)
                Overlap[4*index:4*(index + 1), 4*neighbor:4*(neighbor + 1)] = self.generate_overlap_submatrix(l, m, n, rc)
            
            
        self.Hamilt = Hamilt
        self.Overlap = Overlap
        
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        r'''
        Calculates the eigen energy values, total ground state energy and energy per atom by diagonalising the TB Hamiltonian
        
        a) DOS (real space energy eigen values)
        b) total ground state energy
        c) energy per atom 
        
        Args:
            atoms (Ase atoms object): The input atomic structure
            properties (Optional): Some parameter required by Ase calculator
            system_changes (Optional): Some parameter required by Ase calculator
        '''
        
        if properties is None:
            properties = self.implemented_properties
        
        Calculator.calculate(self, atoms, properties, system_changes)
        
        
        self.generate_real_space_hamiltonian(atoms)
        self.eigvals = splin.eigvalsh(self.Hamilt, self.Overlap)
        
        self.results['energy'] = 2*np.sum(self.eigvals[0:len(self.eigvals)//2])*self.parameters.ryd_to_ev
        self.results['energyperatom'] = self.results['energy']/len(atoms)
            
