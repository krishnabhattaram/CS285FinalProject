import numpy as np
import os
import sys
from ase import Atoms


__all__ = ["generate_Si_atom",
           "generate_Si_blocks",
           "generate_SiO_blocks",
           "generate_HfO_blocks",
           "create_atom_from_positions",
           "create_atom_from_positions_with_supercell",
           "combine_structures",
           ]

def generate_Si_atom(a0, structure, Na, Nb, Nc):
    r'''
    Generate the structures of crystalline Si atom with the given inputs
    
      Args:
          
          a0 (float) :  Lattice constant of the unit cell
          structure (str) : Diamond/simple cubic/ fcc/ bcc
          Na (float) : The number of unit cells along the first basis axis
          Nb (float) : The number of unit cells along the second basis axis
          Nc (float) : The number of unit cells along the third basis axis
        
      Returns:
        
          Si (Atoms ase object) : atoms object containing the atomic positions
    
    '''
  
    if(structure == "diamond"):
        unit_cell_positions = a0*np.array([[0, 0, 0], [1/4, 1/4, 1/4]])
        basis = a0/2*np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
        
        II, JJ, KK = np.meshgrid(np.arange(Na), np.arange(Nb), np.arange(Nc), indexing = 'ij')
        positions = II[None, :, :, :]*basis[0, :, None, None ,None] + JJ[None, :, :, :]*basis[1, :, None, None ,None] + KK[None, :, :, :]*basis[2, :, None, None ,None]
        positions = positions.reshape(3,-1).T
        positions0 = unit_cell_positions[[0],:] + positions
        positions1 = unit_cell_positions[[1],:] + positions
        positions = np.hstack([positions0, positions1]).reshape((-1,3))
        Si = Atoms('Si'*(2*Na*Nb*Nc), positions = positions, cell = a0/2*np.array([[Na, Na, 0], [0, Nb, Nb], [Nc, 0, Nc]]), pbc = True)
    
    
    elif( structure == "sc"):
        unit_cell_positions = np.array([0, 0, 0])
        basis = a0*np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        II, JJ, KK = np.meshgrid(np.arange(Na), np.arange(Nb), np.arange(Nc), indexing = 'ij')
        positions = II[None, :, :, :]*basis[0, :, None, None ,None] + JJ[None, :, :, :]*basis[1, :, None, None ,None] + KK[None, :, :, :]*basis[2, :, None, None ,None]
        positions = positions.reshape(3,-1).T
        positions = unit_cell_positions + positions
        Si = Atoms('Si'*(Na*Nb*Nc), positions = positions, cell = a0*np.array([[Na, 0, 0], [0, Nb, 0], [0, 0, Nc]]), pbc = True)
     
        
    elif(structure == "fcc"):
        unit_cell_positions = np.array([0, 0, 0])
        basis = a0/2*np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
        
        II, JJ, KK = np.meshgrid(np.arange(Na), np.arange(Nb), np.arange(Nc), indexing = 'ij')
        positions = II[None, :, :, :]*basis[0, :, None, None ,None] + JJ[None, :, :, :]*basis[1, :, None, None ,None] + KK[None, :, :, :]*basis[2, :, None, None ,None]
        positions = positions.reshape(3,-1).T
        positions = unit_cell_positions + positions
        Si = Atoms('Si'*(Na*Nb*Nc), positions = positions, cell = a0/2*np.array([[Na, Na, 0], [0, Nb, Nb], [Nc, 0, Nc]]), pbc = True)
    
    
    elif(structure == "bcc"):
        unit_cell_positions = np.array([0, 0, 0])
        basis = a0/2*np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
        
        II, JJ, KK = np.meshgrid(np.arange(Na), np.arange(Nb), np.arange(Nc), indexing = 'ij')
        positions = II[None, :, :, :]*basis[0, :, None, None ,None] + JJ[None, :, :, :]*basis[1, :, None, None ,None] + KK[None, :, :, :]*basis[2, :, None, None ,None]
        positions = positions.reshape(3,-1).T
        positions = unit_cell_positions + positions
        Si = Atoms('Si'*(Na*Nb*Nc), positions = positions, cell = a0/2*np.array([[-Na, Na, Na], [Nb, -Nb, Nb], [Nc, Nc, -Nc]]), pbc = True)
      
    
    else:
      NotImplemented     
  
    return Si
    
    
def generate_Si_blocks(a0, Na, Nb, Nc, return_dictionary = False):
    r'''
    Generate the structures of crystalline Si atom where the basis is simple cubic
    Structure is always diamond
    
      Args:
          
          a0 (float) :  Lattice constant of the unit cellc
          Na (float) : The number of unit cells along the x axis
          Nb (float) : The number of unit cells along the y axis
          Nc (float) : The number of unit cells along the z axis
          return_dictionary (Optional, False) : If True then returns the dictionary which helps to map the index of atom to
                                                position in the unit cell
        
      Returns:
        
          Si (Atoms ase object) : atoms object containing the atomic positions
    
    '''
    natoms = 8*Na*Nb*Nc
    
    unit_cell_positions = a0*np.array([[0, 0, 0], [0, 1/2, 1/2], [1/2, 1/2, 0], [1/2, 0, 1/2], [1/4, 1/4, 1/4], [3/4, 3/4, 1/4], [3/4, 1/4, 3/4], [1/4, 3/4, 3/4]])
    dictionary_pos = np.zeros((natoms, 4))
    dictionary_pos[0:8, 0] = np.arange(8)
    basis = np.eye(3)
    
    
    positions = unit_cell_positions
    for kk in range(Nc):
      for jj in range(Nb):
        for ii in range(Na):             
          if(not (ii == 0 and jj == 0 and kk == 0)):
            positions = np.append(positions, unit_cell_positions + a0*(ii*basis[0] + jj*basis[1] + kk*basis[2]), axis = 0)
            dictionary_pos[8*(ii + 3*jj + 9*kk):8*(ii + 3*jj + 9*kk + 1) , 0] = np.arange(8)
            dictionary_pos[8*(ii + 3*jj + 9*kk):8*(ii + 3*jj + 9*kk + 1), 1] = ii
            dictionary_pos[8*(ii + 3*jj + 9*kk):8*(ii + 3*jj + 9*kk + 1), 2] = jj
            dictionary_pos[8*(ii + 3*jj + 9*kk):8*(ii + 3*jj + 9*kk + 1), 3] = kk
    Si = Atoms("Si" + str(natoms), positions = positions, cell = [Na*a0, Nb*a0, Nc*a0], pbc = True)
    
    if return_dictionary:
        return Si, dictionary_pos
    
    else:
        return Si
        
def generate_HfO_blocks(Na, Nb, Nc, return_dictionary = False):
    r'''
    Generate the structures of crystalline HfO2 atom with cubic symmetry where the basis is simple cubic
    
      Args:
          
          a0 (float) :  Lattice constant of the unit cell
          Na (float) : The number of unit cells along the x axis
          Nb (float) : The number of unit cells along the y axis
          Nc (float) : The number of unit cells along the z axis
          return_dictionary (Optional, False) : If True then returns the dictionary which helps to map the index of atom to
                                                position in the unit cell
        
      Returns:
        
          Si (Atoms ase object) : atoms object containing the atomic positions
    
    '''
    a0 = 5.076
    natoms = 12*Na*Nb*Nc
    
    unit_cell_positions = a0*np.array([[0, 0, 0], [0, 1/2, 1/2], [1/2, 1/2, 0], [1/2, 0, 1/2], [1/4, 1/4, 1/4], [3/4, 1/4, 1/4], [1/4, 3/4, 1/4], [1/4, 1/4, 3/4], [1/4, 3/4, 3/4], [3/4, 1/4, 3/4], [3/4, 3/4, 1/4], [3/4, 3/4, 3/4]])
    dictionary_pos = np.zeros((natoms, 4))
    dictionary_pos[0:12, 0] = np.arange(12)
    basis = np.eye(3)
    
    
    positions = unit_cell_positions
    sym = "Hf"*4 + "O"*8
    
    for kk in range(Nc):
      for jj in range(Nb):
        for ii in range(Na):             
          if(not (ii == 0 and jj == 0 and kk == 0)):
            positions = np.append(positions, unit_cell_positions + a0*(ii*basis[0] + jj*basis[1] + kk*basis[2]), axis = 0)
            dictionary_pos[12*(ii + 3*jj + 9*kk):12*(ii + 3*jj + 9*kk + 1) , 0] = np.arange(12)
            dictionary_pos[12*(ii + 3*jj + 9*kk):12*(ii + 3*jj + 9*kk + 1), 1] = ii
            dictionary_pos[12*(ii + 3*jj + 9*kk):12*(ii + 3*jj + 9*kk + 1), 2] = jj
            dictionary_pos[12*(ii + 3*jj + 9*kk):12*(ii + 3*jj + 9*kk + 1), 3] = kk
            sym += "Hf"*4 + "O"*8
            
    HfO = Atoms(sym, positions = positions, cell = [Na*a0, Nb*a0, Nc*a0], pbc = True)
    
    if return_dictionary:
        return HfO, dictionary_pos
    
    else:
        return HfO
        
        
def generate_SiO_blocks(Na, Nb, Nc, return_dictionary = False):
    r'''
    Generate the structures of alpha-cristobalite SiO atom where the basis is tetragonal
    Structure is always diamond
    
      Args:
          
          Na (float) : The number of unit cells along the x axis
          Nb (float) : The number of unit cells along the y axis
          Nc (float) : The number of unit cells along the z axis
          return_dictionary (Optional, False) : If True then returns the dictionary which helps to map the index of atom to
                                                position in the unit cell
        
      Returns:
        
          SiO (Atoms ase object) : atoms object containing the atomic positions
    
    '''
    natoms = 12*Na*Nb*Nc
    la = 5.08
    lb = 5.08
    lc = 7.10
    basis = np.array([[la, 0, 0], [0, lb, 0], [0, 0, lc]])
    
    unit_cell_positions = np.zeros((12, 3))
    unit_cell_positions[0, :] = np.array([1.497, 1.497, 0])    # Si1 atom
    unit_cell_positions[1, :] = np.array([1.045, 4.040, 1.775])    # Si2 atom
    unit_cell_positions[2, :] = np.array([3.588, 3.588, 3.549])    # Si3 atom
    unit_cell_positions[3, :] = np.array([4.040, 1.045, 5.324])    # Si4 atom
    
    unit_cell_positions[4, :] = np.array([1.226, 0.478, 1.238])    # O1 atom
    unit_cell_positions[5, :] = np.array([1.317, 3.021, 0.537])    # O2 atom
    unit_cell_positions[6, :] = np.array([2.064, 3.768, 3.012])    # O3 atom
    unit_cell_positions[7, :] = np.array([4.606, 3.859, 2.312])    # O4 atom
    unit_cell_positions[8, :] = np.array([3.859, 4.606, 4.787])    # O5 atom
    unit_cell_positions[9, :] = np.array([3.768, 2.064, 4.086])    # O6 atom
    unit_cell_positions[10, :] = np.array([0.478, 1.226, 5.861])    # O7 atom
    unit_cell_positions[11, :] = np.array([3.021, 1.317, 6.562])    # O8 atom
    
    dictionary_pos = np.zeros((natoms, 12))
    dictionary_pos[0:12, 0] = np.arange(12)
    
    
    positions = unit_cell_positions
    sym = "Si"*4 + "O"*8
    
    for kk in range(Nc):
      for jj in range(Nb):
        for ii in range(Na):             
          if(not (ii == 0 and jj == 0 and kk == 0)):
            positions = np.append(positions, unit_cell_positions + (ii*basis[0] + jj*basis[1] + kk*basis[2]), axis = 0)
            dictionary_pos[12*(ii + Na*jj + Na*Nb*kk):12*(ii + Na*jj + Na*Nb*kk + 1) , 0] = np.arange(12)
            dictionary_pos[12*(ii + Na*jj + Na*Nb*kk):12*(ii + Na*jj + Na*Nb*kk + 1), 1] = ii
            dictionary_pos[12*(ii + Na*jj + Na*Nb*kk):12*(ii + Na*jj + Na*Nb*kk + 1), 2] = jj
            dictionary_pos[12*(ii + Na*jj + Na*Nb*kk):12*(ii + Na*jj + Na*Nb*kk + 1), 3] = kk
            sym += "Si"*4 + "O"*8
    
    SiO = Atoms(sym, positions = positions, cell = [Na*la, Nb*lb, Nc*lc], pbc = True)
    
    if return_dictionary:
        return SiO, dictionary_pos
    
    else:
        return SiO
        
        
def create_atom_from_positions(positions, Na, Nb, Nc, type_atom, given_length = False):
    r'''
    Generate the structures of SiO atom or Si atom depending. If type_atom is "Si" then the basis is cubic.
    If type_atom is "SiO" then the basis is tetragonal
    
      Args:
          
          positions(np.array) : The positions of all the atoms
          Na (float) : The number of unit cells along the x axis
          Nb (float) : The number of unit cells along the y axis
          Nc (float) : The number of unit cells along the z axis
          type_atom (str) : 'Si' or 'SiO'
          given_length (bool) : False then Na, Nb, Nc represent the number of unit cells
                                True then Na, Nb, Nc represent the length of the supercell 
       
      Returns:
        
          Atoms ase object : atoms object containing the atomic positions    
    '''
        
    if(type_atom == 'Si'):
        a0 = 5.43
        natoms = Na*Nb*Nc*8    
        sym = "Si"*natoms  
        atoms = Atoms(sym, positions = positions, cell = [Na*a0, Nb*a0, Nc*a0], pbc = True)
        
    elif(type_atom == 'SiO'):
        
        if(given_length == False):
        
            la = 5.08
            lb = 5.08
            lc = 7.10
        
            sym = ('Si'*4 + 'O'*8)*(Na*Nb*Nc)
            atoms = Atoms(sym, positions = positions, cell = [Na*la, Nb*lb, Nc*lc], pbc = True)
            
        else:
        
            sym = ('Si'*18 + 'O'*36)    # First 18 atoms are silicon. Next 36 atoms are oxygen. Confirmed by Jack.
            atoms = Atoms(sym, positions = positions, cell = [Na, Nb, Nc], pbc = True)
        
    elif(type_atom == 'HfO'):
        a0 = 5.076
        sym = ('Hf'*4 + 'O'*8)*(Na*Nb*Nc)
        atoms = Atoms(sym, positions = positions, cell = [Na*a0, Nb*a0, Nc*a0], pbc = True)
        
    
    return atoms
    

def create_atom_from_positions_with_supercell(positions, supercell, type_atom):
    r'''
    Generate the structures of SiO atom or Si atom depending. Supercell basis is also given
    
      Args:
          
          positions(np.array) : The positions of all the atoms
          supercell (np.array) : The supercell basis of the atoms
          type_atom (str) : 'Si' or 'SiO'

      Returns:
        
          Atoms ase object : atoms object containing the atomic positions    
    '''
        
    if(type_atom == 'Si'):
        raise NotImplemented
        
    elif(type_atom == 'SiO'):
        
        sym = ('Si'*18 + 'O'*36)    # First 18 atoms are silicon. Next 36 atoms are oxygen. Confirmed by Jack.
        atoms = Atoms(sym, positions = positions, cell = supercell, pbc = True)
        
    elif(type_atom == 'HfO'):
        raise NotImplemented
        
    
    return atoms
    
    
def combine_structures(sym, atom1, atom2, distance = 1.0):   
    r'''
    Combine the two atomic structures
    
      Args:
          
          sym (String): Total string after combining the two atom structures
          atom1 (Atoms ase object): The first atomic structure 
          atom2 (Atoms ase object): The second atomic structure which is kept on top of the first one along z-axis
          distance (float): Value of distance between the two atomic structures. Default value is 1 A
       
      Returns:
        
          result (Atoms ase object) : Combined atomic structure  
    '''
    
    position1 = atom1.get_positions()
    position2 = atom2.get_positions()
    
    cell1 = np.sum(atom1.get_cell(), axis = 0)
    cell2 = np.sum(atom2.get_cell(), axis = 0)
    
    cellx = max(cell1[0], cell2[0])
    celly = max(cell1[1], cell2[1])
    cellz = cell1[2] + cell2[2] + distance
    
    position2 = position2 + np.array([[abs(cell1[0] - cell2[0])/2, abs(cell1[1] - cell2[1])/2, cell1[2] + distance]])
    positions = np.append(position1, position2, axis = 0)
    print(positions.shape)
    
    result = Atoms(sym, positions = positions, cell = [cellx, celly, cellz], pbc = [True, True, False])
        
    return result
    
        