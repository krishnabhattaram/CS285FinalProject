import numpy as np
import torch
from ase.neighborlist import neighbor_list
from schnetpack.environment import collect_atom_triples
from schnetpack.nn import atom_distances, triple_distances


__all__ = ["plot_pair_distribution",
           "calc_bond_angle_dist",
           "calc_deviations",
           "create_environment",
          ]

def plot_pair_distribution(atoms, cutoff = 10.0):
    r'''
    Calculates g(r) for the input atomic structure. Neighbors are calculated within the cutoff
    
    Args:
        
        atoms (Ase atoms object): input atomic structure containing atomic positions
        cutoff (float): cutoff within which neighbors are calculated to calculate g(r)
    
    Returns:
        
        left_edges (np.array of float): the left edges of histogram or plotting
        pdf (np.array of float): the value of g(r) corresponding to the left edges (normalised)
        width (float): the width of the bar for the histogram
    '''

    d = neighbor_list('d', atoms, cutoff)
    h, bin_edges = np.histogram(d, bins=100)
    pdf = h/(4*np.pi/3*(bin_edges[1:]**3 - bin_edges[:-1]**3)) * atoms.get_volume()/len(atoms)
    pdf = pdf/np.max(pdf)
    
    
    left_edges = bin_edges[:-1]
    width = left_edges[1] - left_edges[0]
    
    return left_edges, pdf, width
    
def calc_bond_angle_dist(atoms, radial_rc, angular_rc):
    r'''
    Calculates the distances to its neighbors.
    Also calculates the bond angles between two neighbors from a tetrahedron.
    
    Args:
    
        atoms (Ase atoms object): input atomic strcuture containing atomic positions
        radial_rc (float): the cutoff within which the radial neighbors are calculated
        angular_rc (float): the cutoff within which the angular neighbors are calculated
    
    Returns:
      
        distance_array (np.array of float): All the bond lengths
        angle_array (np.array of float): All the angles between pairs of neighbors from tetrahadron (Absolute value)
        
        Note: Mask is applied by making the values np.nan
        
    '''
    

    positions = atoms.positions
    neighbor_indices, neighbor_mask, cell, cell_offset = create_environment(atoms, radial_rc)    
    ang_neighbor_indices, ang_neighbor_mask, ang_cell, ang_cell_offset = create_environment(atoms, angular_rc)  
    nbh_idx_j, nbh_idx_k, offset_idx_j, offset_idx_k = collect_atom_triples(ang_neighbor_indices)
    
    # torchifying the numpy data 
    positions = torch.from_numpy(positions)[None,:,:].float()
    neighbor_indices = torch.from_numpy(neighbor_indices)[None,:,:]
    ang_neighbor_indices = torch.from_numpy(ang_neighbor_indices)[None,:,:]
    nbh_idx_j = torch.from_numpy(nbh_idx_j)[None,:,:]
    nbh_idx_k = torch.from_numpy(nbh_idx_k)[None,:,:]
    offset_idx_j = torch.from_numpy(offset_idx_j)[None,:,:]
    offset_idx_k = torch.from_numpy(offset_idx_k)[None,:,:]
    neighbor_mask = torch.from_numpy(neighbor_mask)[None,:,:]
    ang_neighbor_mask = torch.from_numpy(ang_neighbor_mask)[None,:,:]
    cell = torch.from_numpy(cell)[None,:,:].float()
    ang_cell = torch.from_numpy(ang_cell)[None,:,:].float()
    cell_offset = torch.from_numpy(cell_offset)[None,:,:,:].float()
    ang_cell_offset = torch.from_numpy(ang_cell_offset)[None,:,:,:].float()
    
    nbatch, natoms, nneigh = ang_neighbor_indices.shape
    triu_idx_row, triu_idx_col = np.triu_indices(nneigh, k=1)
    triu_idx_flat = triu_idx_row * nneigh + triu_idx_col
    
    neighbor_mask_ij = ang_neighbor_mask.repeat(1, 1, nneigh) 
    neighbor_mask_ij = neighbor_mask_ij[:,:,triu_idx_flat]
    neighbor_mask_ik = ang_neighbor_mask.repeat_interleave(nneigh, dim=2).reshape((1, natoms, -1))
    neighbor_mask_ik = neighbor_mask_ik[:,:,triu_idx_flat]   
    
    rr_ij, rr_ij_vec = atom_distances(positions, neighbor_indices, cell=cell, cell_offsets=cell_offset, return_vecs = True)
    R_ij, R_ik, R_jk = triple_distances(positions, nbh_idx_j, nbh_idx_k, offset_idx_j, offset_idx_k, cell=ang_cell, cell_offsets=ang_cell_offset, return_vecs=True)
    
    r_ij = torch.norm(R_ij, dim=3)
    r_ik = torch.norm(R_ik, dim=3)
    costheta = (torch.sum(R_ij*R_ik, dim=3)/(r_ij*r_ik + 1e-12)*neighbor_mask_ij*neighbor_mask_ik).detach().cpu().numpy()
    costheta[costheta > 1.0 ] = 1.0
    costheta[costheta < -1.0] = -1.0
    theta = np.arccos(costheta)
    rr_ij = rr_ij.detach().cpu().numpy()
    neighbor_mask = neighbor_mask.detach().cpu().numpy()
    neighbor_mask_ij = neighbor_mask_ij.detach().cpu().numpy()
    neighbor_mask_ik = neighbor_mask_ik.detach().cpu().numpy()
    
    '''
    print('Radial matrix : ',rr_ij.shape)
    print('Theta matrix : ',theta.shape)
    print('Neighbor mask matrix : ',neighbor_mask.shape)
    print('Neighbor mask ij matrix : ', neighbor_mask_ij.shape)
    print('Neighbor mask ik matrix : ', neighbor_mask_ik.shape)
    '''
    
    ### Masking
    rr_ij[~neighbor_mask] = np.nan
    angle_array = theta * 180/np.pi
    angle_array[~(neighbor_mask_ij * neighbor_mask_ik)] = np.nan
        
    return rr_ij, angle_array 
 
 
    
def calc_deviations(atoms, bond_eq, rc):
    r'''
    Calculates the deviation of distances to its neighbors from the equilibrium bond length.
    Also calculates the deviation in bond angles between two neighbors from a tetrahedron (cos \theta = -1/3).
    
    Args:
    
        atoms (Ase atoms object): input atomic strcuture containing atomic positions
        bond_eq (float): the equilibrium bond length from which the deviation is calculated
        rc (float): the cutoff within which the neighbors are calculated
    
    Returns:
      
        dev_distance_array (np.array of float): All the deviations of bond lengths from the equilibrium value (Returned in percentage form)
        dev_angle_array (np.array of float): All the deviations of angles between pairs of neighbors from tetrahadron (Absolute value)
        
    '''
    

    positions = atoms.positions
    neighbor_indices, neighbor_mask, cell, cell_offset = create_environment(atoms, rc)
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
    
    nbatch, natoms, nneigh = neighbor_indices.shape
    triu_idx_row, triu_idx_col = np.triu_indices(nneigh, k=1)
    triu_idx_flat = triu_idx_row * nneigh + triu_idx_col
    
    neighbor_mask_ij = neighbor_mask.repeat(1, 1, nneigh) 
    neighbor_mask_ij = neighbor_mask_ij[:,:,triu_idx_flat]
    neighbor_mask_ik = neighbor_mask.repeat_interleave(nneigh, dim=2).reshape((1, natoms, -1))
    neighbor_mask_ik = neighbor_mask_ik[:,:,triu_idx_flat]   
    
    rr_ij, rr_ij_vec = atom_distances(positions, neighbor_indices, cell=cell, cell_offsets=cell_offset, return_vecs = True)
    R_ij, R_ik, R_jk = triple_distances(positions, nbh_idx_j, nbh_idx_k, offset_idx_j, offset_idx_k, cell=cell, cell_offsets=cell_offset, return_vecs=True)
    
    r_ij = torch.norm(R_ij, dim=3)
    r_ik = torch.norm(R_ik, dim=3)
    costheta = (torch.sum(R_ij*R_ik, dim=3)/(r_ij*r_ik + 1e-12)*neighbor_mask_ij*neighbor_mask_ik).detach().cpu().numpy()
    costheta[costheta > 1.0 ] = 1.0
    costheta[costheta < -1.0] = -1.0
    theta = np.arccos(costheta)
    rr_ij = rr_ij.detach().cpu().numpy()
    neighbor_mask = neighbor_mask.detach().cpu().numpy()
    neighbor_mask_ij = neighbor_mask_ij.detach().cpu().numpy()
    neighbor_mask_ik = neighbor_mask_ik.detach().cpu().numpy()
    
    '''
    print('Radial matrix : ',rr_ij.shape)
    print('Theta matrix : ',theta.shape)
    print('Neighbor mask matrix : ',neighbor_mask.shape)
    print('Neighbor mask ij matrix : ', neighbor_mask_ij.shape)
    print('Neighbor mask ik matrix : ', neighbor_mask_ik.shape)
    '''
    
    dev_distance_array = np.sqrt((rr_ij - bond_eq)**2*neighbor_mask).ravel()   # RMS deviation of bond length
    dev_angle_array = (np.sqrt((theta - np.arccos(-1/3))**2)*180/np.pi*neighbor_mask_ij*neighbor_mask_ik).ravel()    # RMS deviation in degrees
    
    return dev_distance_array/bond_eq*100, dev_angle_array
  
def create_environment(atoms, rc):
    r'''
    Returns the neighbors of all atoms within a given cutoff
    
    Args:
    
        atoms (Ase atoms object): input atomic strcuture containing atomic positions
        rc (float): the cutoff within which the neighbors are calculated
        
    Returns:
    
        neighborhood_idx (np.array of float of shape natoms X nmax_neighbors) : Returns the neighbor indices of all atoms.
                                                                       Even the non-existent neighbors are indexed as 0 
                                                                       
        mask (np.array of bool of shape natoms X nmax_neighbors): True if truly a neighbor. False if the neighbor is non-existent
        cell (np.array of float of shape 3 X 3): The basis of the unit cell
        offset (np.array of float of shape natoms X nmax_neighbors X 3): The offset values for each neighbor to calculate atomic distance vectors 
         
    '''
    natoms = len(atoms)
    idx_i, idx_j, idx_S = neighbor_list(
        "ijS", atoms, rc, self_interaction=False
    )
    if idx_i.shape[0] > 0:
        uidx, n_nbh = np.unique(idx_i, return_counts=True)
        n_max_nbh = np.max(n_nbh)
    
        n_nbh = np.tile(n_nbh[:, np.newaxis], (1, n_max_nbh))
        nbh_range = np.tile(
            np.arange(n_max_nbh, dtype=np.int64)[np.newaxis], (n_nbh.shape[0], 1)
        )
    
        mask = np.zeros((natoms, np.max(n_max_nbh)), dtype=np.bool)
        mask[uidx, :] = nbh_range < n_nbh
        neighborhood_idx = np.zeros((natoms, np.max(n_max_nbh)), dtype=np.int64)
        neighborhood_idx[mask] = idx_j
        
        offset = np.zeros((natoms, np.max(n_max_nbh), 3), dtype=np.float32)
        offset[mask] = idx_S
        
    else:
        neighborhood_idx = np.zeros((natoms, 1), dtype=np.int64)
        mask = np.zeros((natoms, 1), dtype=np.int64)
        offset = np.zeros((natoms, 1, 3), dtype=np.float32)
    
    cell = np.array(atoms.get_cell())
    
    return neighborhood_idx, mask, cell, offset