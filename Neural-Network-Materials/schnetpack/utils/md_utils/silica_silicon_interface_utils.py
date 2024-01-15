import numpy as np
import torch

from ase.optimize import BFGS, FIRE
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.md.andersen import Andersen
from ase.md.nptberendsen import NPTBerendsen
from ase.md.npt import NPT
from ase import units
from ase.neighborlist import neighbor_list


from schnetpack.nn import atom_distances
from .amorphous_silicon_utils import reset_positions

__all__ = ["generate_Si_by_layers",
           "insert_oxygen_atoms",
           ]
           
           
def generate_Si_by_layers(Na, Nb, Nc):
    r'''
    Generate the structures of crystalline Si atom where the basis is simple cubic. Creates a slab which is periodic
    in x and y direction. (001 slabs)
    Structure is always diamond
    
      Args:
          
          Na (float) : The number of unit cells along the x axis
          Nb (float) : The number of unit cells along the y axis
          Nc (float) : The number of unit cells along the z axis (not periodic in this direction)
      
      Returns:
        
          Si (Atoms ase object) : atoms object containing the atomic positions
    
    '''
    
    a0 = 5.43
    
    x_layer1 = np.tile(a0 * np.array([[0, 0, 0], [1/2, 1/2, 0]]), (Na, 1)) 
    addition = np.pad(np.repeat(a0 * np.arange(Na), 2)[:, None], ((0, 0), (0, 2)))
    x_layer1 += addition
    
    xy_layer1 = np.tile(x_layer1, (Nb, 1))
    addition = np.pad(np.repeat(a0 * np.arange(Nb), 2 * Na)[:, None], ((0, 0), (1, 1)))
    xy_layer1 += addition
    
    xy_layer2 = xy_layer1 + a0 * np.array([[1/4, 1/4, 1/4]])
    
    x_layer3 = np.tile(a0 * np.array([[0, 1/2, 1/2], [1/2, 0, 1/2]]), (Na, 1)) 
    addition = np.pad(np.repeat(a0 * np.arange(Na), 2)[:, None], ((0, 0), (0, 2)))
    x_layer3 += addition
    
    xy_layer3 = np.tile(x_layer3, (Nb, 1))
    addition = np.pad(np.repeat(a0 * np.arange(Nb), 2 * Na)[:, None], ((0, 0), (1, 1)))
    xy_layer3 += addition
    
    xy_layer4 = xy_layer3 + a0 * np.array([[1/4, 1/4, 1/4]])
    
    
    positions_array = np.concatenate((xy_layer1, xy_layer2, xy_layer3, xy_layer4))
    positions_array = np.tile(positions_array, (Nc , 1))
    addition = np.pad(np.repeat(a0 * np.arange(Nc), 2 * Na * Nb * 4)[:, None], ((0, 0), (2, 0)))
    positions_array += addition
      
    # Adding the top final layer along the z axis for the Si slab
    positions_array = np.concatenate((positions_array, xy_layer1 + a0 * np.array([[0, 0, Nc]])))
    
    natoms = positions_array.shape[0]
    Si = Atoms("Si" + str(natoms), positions = positions_array, cell = [Na*a0, Nb*a0, Nc*a0], pbc = [1, 1, 0])
    
    return Si
   
def get_neighbors(atoms):
        r'''
        Gives the distance vectors of the bonds to the next layer
        
        Args:
           
           atoms (Atoms ase object): The Si atoms in the layer whose bonds to the next layer needs to be calculated
           
        Returns:
         
           rr_ij_vec (np.array): The vectors pointing to the bonded Si atoms
           mask (np.array): The mask to remove the non-existent neighbors
        '''
        rc = 2.5 
        natoms = len(atoms)
        idx_i, idx_j, idx_S = neighbor_list(
            "ijS", atoms, rc, self_interaction=False
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
        
        cell = np.array(atoms.get_cell())
        
        positions = torch.from_numpy(atoms.positions)[None,:,:].float()
        neighbor_indices = torch.from_numpy(neighborhood_idx)[None,:,:]
        cell = torch.from_numpy(cell)[None,:,:].float()
        offset = torch.from_numpy(offset)[None,:,:,:].float()
        
        rr_ij, rr_ij_vec = atom_distances(positions, neighbor_indices, cell=cell, cell_offsets=offset, return_vecs = True)
        rr_ij_vec = rr_ij_vec.cpu().numpy().squeeze()
        neighbor_indices = neighbor_indices.cpu().numpy().squeeze()
        
        return rr_ij_vec, neighbor_indices, mask
        
        
    
def insert_oxygen_atoms(atoms, layers, num_sheet_atoms):
    r'''
    Inserts oxygen atoms into crystalline silicon at each Si-Si bond
    
      Args:
          
          atoms (Atoms ase object): The input crystalline Si slab layer by layer
          layers (int): The number of layers in which oxygen needs to be added
          num_sheet_atoms (int): The number of atoms in one layer of crystalline Si

      Returns:
      
          SiO (Atoms ase object): after addition of the oxygen atom into crystalline silicon
    '''
    
    num_atoms = len(atoms)
    rr_ij_vec, neighbor_indices, neighbor_mask = get_neighbors(atoms)
    connectivity_matrix = np.zeros((len(atoms), len(atoms)))    # to keep account th
    oxygen_positions = np.empty((0 , 3))

    
    for ii in range(num_atoms - 1, num_atoms - 1 - num_sheet_atoms * layers, -1):
        for jj in range(4):
            if((neighbor_mask[ii, jj] == True) and (connectivity_matrix[ii, neighbor_indices[ii , jj]] == 0)):   
                oxygen_positions = np.append(oxygen_positions, (atoms.positions[ii, :] + rr_ij_vec[ii, jj, :]/2)[None,:], axis = 0)
                connectivity_matrix[ii ,neighbor_indices[ii , jj]] = 1
                connectivity_matrix[neighbor_indices[ii , jj] ,ii] = 1
    
    positions = atoms.positions
    natoms_Si = len(atoms)
    natoms_O = oxygen_positions.shape[0]

    total_positions = np.append(positions, oxygen_positions, axis = 0)
    SiO = Atoms("Si" + str(natoms_Si) + "O" + str(natoms_O), positions = total_positions, cell = atoms.get_cell(), pbc = [True, True, False])
    SiO = reset_positions(SiO)
    
    return SiO
    
    
    
    
    
