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