import numpy as np

from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from ase.neighborlist import NewPrimitiveNeighborList, neighbor_list


class AseKeating(Calculator):

    implemented_properties = ['energy', 'energies', 'free_energy', 'forces']
    default_parameters = {
        'alpha': 2.965,
        'beta': 0.285*2.965,
        'a0': 5.43,
        'd': 2.35,
        'rc': 1.0,
        'neighbor_indices': None,
        'neighbor_mask': None,
        'cell_offset': None
    }
    nolabel = True

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        alpha: float
          The constant for the radial dependence of Keating potential
        beta: float
          The constant for the angular dependence of Keating potential
        d: float
          The equlibirum distance length of the nearest neighbors
        rc: float, None
          Cut-off for the NeighborList is set to 3 * sigma if None.

        """

        Calculator.__init__(self, **kwargs)

        self.nl = None

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)

        natoms = len(self.atoms)

        alpha = self.parameters.alpha
        beta = self.parameters.beta
        d = self.parameters.d
        rc = self.parameters.rc
        a0 = self.parameters.a0
        neighbor_indices = self.parameters.neighbor_indices
        neighbor_mask = self.parameters.neighbor_mask
        offset = self.parameters.offset
        positions = self.atoms.positions
        
        if neighbor_indices is None:
            neighbor_indices, neighbor_mask, cell_offset = self.create_environment(rc, natoms)
            self.parameters.neighbor_indices = neighbor_indices
            self.parameters.neighbor_mask = neighbor_mask
            self.parameters.cell_offset = cell_offset
                    
        energies = np.zeros(natoms)
        forces = np.zeros((natoms, 3))
        
        ''' 
        Generating the R, T and L matrix. 
        R: Neighbor distance vectors. 
        T: T matrix for angular dependence. Contains the dot product with distance vectors of all neighbors
        L: L matrix for angular depende. Contains the 
        '''
        
        R, T, L, neighbor_mask_T = self.generate_RTL_matrix(natoms, positions, neighbor_indices, neighbor_mask, offset)  
        
        energies = self.calc_energy(R, T, alpha, beta, d, neighbor_mask, neighbor_mask_T)      
        energy = energies.sum()
        forces = self.calc_forces(R, T, L, neighbor_indices, neighbor_mask, alpha, beta, d, natoms)

        self.results['energy'] = energy
        self.results['energies'] = energies
        self.results['free_energy'] = energy
        self.results['forces'] = forces
     
     
    '''
    Creates the neighbor environment for all atoms.
    Returns neighbor indices and neighbor mask
    '''   
        
    def create_environment(self, rc, natoms):
        
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
            neighborhood_idx = -np.ones((natoms, np.max(n_max_nbh)), dtype=np.int_)
            neighborhood_idx[mask] = idx_j
            
            offset = np.zeros((n_atoms, np.max(n_max_nbh), 3), dtype=np.float32)
            offset[mask] = idx_S

        else:
            neighborhood_idx = -np.ones((natoms, 1), dtype=np.int_)
            mask = np.ones((natoms, np.max(n_max_nbh)), dtype=np.bool)
            offset = np.zeros((n_atoms, 1, 3), dtype=np.float32)
            
        return neighborhood_idx, mask, offset
        

    '''
    Generates the R,T and L matrix for the calculation of energy and atomic forces
    
    R (N X n X 3): Distance matrix to all neighbors for all the atoms
    T (n - 1 X N X n): Dot product of the distance vector to all its neighbors
    L (n - 1 X N X n X 3): Dot product of the distance vector to all its neighbors
    '''
    def generate_RTL_matrix(self, natoms, positions, neighbor_indices, neighbor_mask, L_array):  
        
        n_max_nbh = neighbor_indices.shape[1]
     
        R = np.zeros((natoms, n_max_nbh, 3))    # The distance matrix for the neighbors inside rc for all the atoms
        T = np.zeros((n_max_nbh - 1, natoms, n_max_nbh))    # Dot product with all neighbor vectors
        L = np.zeros((num_bonds - 1, natoms, num_bonds, 3))    # Sum of all neighbor vectors
        neighbor_mask_T = np.zeros((num_bonds - 1, natoms, num_bonds))    # neigbor_mask for T matrix
        
        B, A, N, D = cell_offsets.size()
        cell_offsets = cell_offsets.view(B, A * N, D)
        offsets = cell_offsets.bmm(cell)
        offsets = offsets.view(B, A, N, D)
        dist_vec += offsets
        
        R = R*neighbor_mask[:, :, None]
        
        for kk in range(1, num_bonds):    
            neighbor_mask_T[kk - 1, :, :] = neighbor_mask*np.roll(neighbor_mask, -kk, axis = 1)
            shiftedR = np.roll(R, -kk, axis = 1)
            T[kk - 1, :, :] = np.sum(R*shiftedR, axis = 2)*neighbor_mask_T[kk -1, :, :]
            L[kk - 1, :, :, :] = (R + shiftedR)*neighbor_mask_T[kk -1, :, :, None]
            
        
        return R, T, L, neighbor_mask_T
     
    '''
    Calculates the Keating energy with the R and T Matrix
    '''   
    def calc_energy(self, R, T, alpha, beta, d, neighbor_mask, neighbor_mask_T):
        E_radial = 3*alpha/(16*d**2)*np.sum(((np.sum(R**2, axis = 2) - d**2)**2)*neighbor_mask, axis = 1)
        E_angular = 3*beta/(16*d**2)*np.sum(((T + 1/3*d**2)**2)*neighbor_mask_T, axis = (0, 2))
        
        return E_radial + E_angular
        
    '''
    Calculates the Keating force on all atoms with the R, T and L matrix
    '''
    def calc_forces(self, R, T, L, neighbor_indices, neighbor_mask, alpha, beta, d, natoms):
        Falpha = 3*alpha/(2*d**2)*np.sum((np.sum(R**2, axis = 2) - d**2)[:,:,None]*R, axis = 1)    # Force due to to radial energy
        Fbeta1 = 3*beta/(8*d**2)*np.sum((T + 1/3*d**2)[:, :,:,None]*L, axis = (0, 2))    # Part 1 Force due to angular energy
        Fbeta2 = np.zeros((natoms, 3))
        
        for aa in range(natoms):
            Rdash = R[neighbor_indices[aa, :], :, :]*neighbor_mask[aa][:, None, None]
      
            if Rdash[~np.all(Rdash ==0, axis = (1, 2))].shape[0] < 2 :    # If the number of neighbors is less than 2 then no force
                Rdash = np.zeros_like(Rdash)
            
            mask = neighbor_indices[neighbor_indices[aa, :], :] == aa    # This mask is so that the neighbor of the neighbor of aa is not aa
            Rdash[mask,:] = 0
            Fbeta2[aa, :] = 3*beta/(4*d**2)*np.sum((R[[aa], :, None ,:]*Rdash[None, :, :, :] - 1/3*d**2)*Rdash[None, :, :, :], axis = (0, 1, 2))
        
        return (Falpha + Fbeta1 + Fbeta2)
            
            
      
                

def cutoff_function(r, rc, ro):
    """Smooth cutoff function.

    Goes from 1 to 0 between ro and rc, ensuring
    that u(r) = lj(r) * cutoff_function(r) is C^1.

    Defined as 1 below ro, 0 above rc.

    Note that r, rc, ro are all expected to be squared,
    i.e. `r = r_ij^2`, etc.

    Taken from https://github.com/google/jax-md.

    """

    return np.where(
        r < ro,
        1.0,
        np.where(r < rc, (rc - r) ** 2 * (rc + 2 * r - 3 * ro) / (rc - ro) ** 3, 0.0),
    )


def d_cutoff_function(r, rc, ro):
    """Derivative of smooth cutoff function wrt r.

    Note that `r = r_ij^2`, so for the derivative wrt to `r_ij`,
    we need to multiply `2*r_ij`. This gives rise to the factor 2
    above, the `r_ij` is cancelled out by the remaining derivative
    `d r_ij / d d_ij`, i.e. going from scalar distance to distance vector.
    """

    return np.where(
        r < ro,
        0.0,
        np.where(r < rc, 6 * (rc - r) * (ro - r) / (rc - ro) ** 3, 0.0),
    )
