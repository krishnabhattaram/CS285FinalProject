import numpy as np

from ase.optimize import BFGS, FIRE
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.md.andersen import Andersen
from ase.md.nptberendsen import NPTBerendsen
from ase.md.npt import NPT
from ase import units

from schnetpack.utils import Stillinger_Weber, Heterogenous_Stillinger_Weber, SiO2_Stillinger_Weber

__all__ = ["reset_positions",
           "print_statements",
           "return_scheduled_temperature",
           "simulate_andersen_dynamics",
           "simulate_const_energy_dynamics",
           "equilibriate_structure",
           ]

def reset_positions(Si):
    r'''
    Resets the positions of all the atoms in Si according to PBC. Assuming the cell vectors are along x, y and z axes.
    
    Args:
        Si (atoms Ase object) : The atomic structure and be periodic
        
    Returns
        Si (Atoms Ase object) : The update positions according to PBC
    
    '''
    
    cell = np.sum(np.array(Si.get_cell()), axis = 0)
    
    for ii, truth_val in enumerate(Si.get_pbc()):
        if(truth_val ==  True):
           Si.positions[:, ii] = np.remainder(Si.positions[:, ii], cell[ii])
    return Si
    
def print_statements(Si, T = None, divide_by_atoms = True):
    r'''
    Prints some statements relating to energies and forces 
    
    Args:
        
        Si (Ase atoms object) : The atomic structure. Should have a calculator
        T (Optional) : Current Temperature
    '''
    
    if T is None:
        
        epot = Si.get_potential_energy()
        ekin = Si.get_kinetic_energy() / len(Si)
        fmax = np.max(Si.get_forces())
        
        if divide_by_atoms == True:
            epot /= len(Si)
            
            
        print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
              'Etot = %.3feV  fmax = %.3f' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin, fmax))
    
    else:

        epot = Si.get_potential_energy()
        ekin = Si.get_kinetic_energy() / len(Si)
        fmax = np.max(Si.get_forces())
        
        if divide_by_atoms == True:
            epot /= len(Si)
            
        print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
              'Etot = %.3feV  Tset = %3.0fK  fmax = %.3f' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin, T, fmax))
    
    
    

def return_scheduled_temperature(Tinitial, rate, ii, step):
    '''
    Returns the sceduled temperature according to linear decrease rate
    
    Args:
    
        rate (float): Rate of temepratrue decrease in K/fm
        ii (int): Current time step
        Tinitial (float): Initial temperature set
        step (int): Step size for constant temperature for some time
    
    Returns:

        T (float) : The value of scheduled temperature    
    '''
    return Tinitial - rate*(ii//step)*step


def simulate_andersen_dynamics(atoms,
                               Tinitial,
                               Tfinal,
                               rate, 
                               initial_time,
                               step,
                               total_steps,
                               andersen_prob,
                               temp_step = 1.0,
                               initialise_velocity = True,
                               write_interval = 10,
                               print_observables = False,
                               divide_by_atoms = True,
                               calculate_analytical = False):
                               
    r'''
    Simulates the andersen dynamics according to a temperature schedule
    
    Args:
        atoms (Ase atoms object) : Perform md simulation on this atoms object. Has to have a calculator
        Tinitial (float) : The initial temperature to start the dynamics
        Tfinal (float) : The final temperature to end the dynamics
        rate (float) : The rate at which temperature is decreased (if rate = 0, then the temperature is never set for andersen)
        initial_time (float): The initial time for logging
        step (float) : The step size relative to Ase units.fs
        total_steps (int) : Totat number of iterations to perform md
        andersen_prob (float) : The value of andersen probability 
        temp_step (Optional= 1.0) : The step size at which temperature remains constant
        initialise_velocity (Optional : True) : If true then initialise the velocity according to Maxwell Boltzmann Set at Tinitial
        write_interval (Optional : 10) : The interval at which observables are saved
        print_observables (Optional : False) : Whether to print the energies and maximum forces during each time step
        divide_by_atoms (Optional: True): If True then divide the PE of atoms by the number of atoms
        calculate_analytical (Optional : False) : If True then analytically calculate atoms energy using Stillinger Weber
        
    Returns:
    
        temp_array (np.array) : The whole temperature schedule 
        time_array (np.array) : The real time of molecular dynamics
        ekin_array (np.array) : Contains the kinetic energy of the atom
        epot_array (np.array) : Constains the potential energy of the atom
        opt_bool_array (np.array) : Whether the optimizer was on or off
        positions_array (np.array) : positions of the atoms when performing MD
    '''
    
    time_step = step*units.fs    # Converting to fs
    T = Tinitial
    analytical_calculator = SiO2_Stillinger_Weber()
    
    temp_array = np.array([])
    time_array = np.array([])
    ekin_array = np.array([])
    epot_array = np.array([])
    analytical_epot_array = np.array([])
    opt_bool_array = np.array([])
    positions_array = np.empty((0, len(atoms), 3))
        
    
    if initialise_velocity == True:
        MaxwellBoltzmannDistribution(atoms, temperature_K = Tinitial)
        Stationary(atoms)    # Sets the center-of-mass momentum to zero while keeping the temperature constant
    dyn = Andersen(atoms, time_step, temperature_K = Tinitial, andersen_prob = andersen_prob)  # 5 fs time step
    
    
    time = initial_time
    positions_array = np.append(positions_array, atoms.positions[None,:,:], axis = 0)
    
    epot = atoms.get_potential_energy()    # total potential energy 
    ekin = atoms.get_kinetic_energy() / len(atoms)    # kinetic energy per atom
    
    if divide_by_atoms == True:
        epot = epot / len(atoms)
    
    if calculate_analytical == True:
        analytical_calculator.calculate(atoms)
        analytical_epot_array = np.append(analytical_epot_array, analytical_calculator.results["energy"]/len(atoms))    # per atom
    
    temp_array = np.append(temp_array, Tinitial)
    time_array = np.append(time_array, time)
    ekin_array = np.append(ekin_array, ekin)
    epot_array = np.append(epot_array, epot)
    opt_bool_array = np.append(opt_bool_array, False)
    
    
    for kk, _ in enumerate(dyn.irun(total_steps)):
        
        atoms = reset_positions(atoms)
        time += step
        
        if((kk + 1)%write_interval == 0):
            
            positions_array = np.append(positions_array, atoms.positions[None,:,:], axis = 0)
            
            
            epot = atoms.get_potential_energy()
            ekin = atoms.get_kinetic_energy() / len(atoms)
            
            if divide_by_atoms == True:
                epot = epot / len(atoms)
            
            if calculate_analytical == True :             
                analytical_calculator.calculate(atoms)
                analytical_epot_array = np.append(analytical_epot_array, analytical_calculator.results["energy"]/len(atoms))    # per atom

            temp_array = np.append(temp_array, T)
            time_array = np.append(time_array, time)
            ekin_array = np.append(ekin_array, ekin)
            epot_array = np.append(epot_array, epot)
            opt_bool_array = np.append(opt_bool_array, False)
            
           
        
        if print_observables == True:             
            print_statements(atoms, T, divide_by_atoms = divide_by_atoms)
        
        if(rate != 0) :    # There is some change in temperature           
            T = return_scheduled_temperature(Tinitial, rate, kk*step, temp_step)
            
            if(T < Tfinal):
                T = Tfinal    # Saturate the temperature unitil total steps of MD
                
            dyn.set_temperature(T)
                                        
    return [temp_array, time_array, ekin_array, epot_array, opt_bool_array, positions_array, analytical_epot_array]
    
    

def simulate_const_energy_dynamics(atoms,
                                   temperature,
                                   time_stamp,
                                   step,
                                   total_steps,
                                   write_interval = 10,
                                   divide_by_atoms = True,
                                   print_observables = False,
                                   calculate_analytical = False):
                           
    r'''
    Does constant energy dynamics for relaxation of the atomic stucture
    
    Args:
    
        atoms (Ase atoms object): The input atomic structure
        temperature (float): The current temperature of the system
        time_stamp (float): The current time stamp of the md simulation
        step (float) : The step size relative to Ase units.fs
        total_steps (int) : Totat number of iterations to perform md
        write_interval (int): The interval at which observables are saved
        divide_by_atoms (Optional: True): If True then divide the PE of atoms by the number of atoms
        print_observables (Optional : False) : Whether to print the energies and maximum forces during each time step
        calculate_analytical (Optional : False) : If True then analytically calculate atoms energy using Stillinger Weber
        
    Returns:
        
        temp_array (np.array) : The whole temperature schedule (only the constant temperature)
        time_array (np.array) : The real time of molecular dynamics (only time stamps)
        ekin_array (np.array) : Contains the kinetic energy of the atom
        epot_array (np.array) : Constains the potential energy of the atom
        opt_bool_array (np.array) : Whether the optimizer was on or off
        positions_array (np.array) : positions of the atoms when performing equilibriation
        
    '''
    
    time_step = step*units.fs    # Converting to fs
    T = temperature
    analytical_calculator = SiO2_Stillinger_Weber()
    
    temp_array = np.array([])
    time_array = np.array([])
    ekin_array = np.array([])
    epot_array = np.array([])
    analytical_epot_array = np.array([])
    opt_bool_array = np.array([])
    positions_array = np.empty((0, len(atoms), 3))
        

    dyn = VelocityVerlet(atoms, time_step)  # 5 fs time step
    
    
    time = time_stamp
    positions_array = np.append(positions_array, atoms.positions[None,:,:], axis = 0)
    
    epot = atoms.get_potential_energy()    # total potential energy 
    ekin = atoms.get_kinetic_energy() / len(atoms)    # kinetic energy per atom
    
    if divide_by_atoms == True:
        epot = epot / len(atoms)
    
    if calculate_analytical == True:
        analytical_calculator.calculate(atoms)
        analytical_epot_array = np.append(analytical_epot_array, analytical_calculator.results["energy"]/len(atoms))    # per atom
    
    temp_array = np.append(temp_array, T)
    time_array = np.append(time_array, time)
    ekin_array = np.append(ekin_array, ekin)
    epot_array = np.append(epot_array, epot)
    opt_bool_array = np.append(opt_bool_array, False)
    
    
    for kk, _ in enumerate(dyn.irun(total_steps)):
        
        atoms = reset_positions(atoms)
        time += step
        
        if((kk + 1)%write_interval == 0):
            
            positions_array = np.append(positions_array, atoms.positions[None,:,:], axis = 0)
            
            
            epot = atoms.get_potential_energy()
            ekin = atoms.get_kinetic_energy() / len(atoms)
            
            if divide_by_atoms == True:
                epot = epot / len(atoms)
            
            if calculate_analytical == True :             
                analytical_calculator.calculate(atoms)
                analytical_epot_array = np.append(analytical_epot_array, analytical_calculator.results["energy"]/len(atoms))    # per atom

            temp_array = np.append(temp_array, T)
            time_array = np.append(time_array, time)
            ekin_array = np.append(ekin_array, ekin)
            epot_array = np.append(epot_array, epot)
            opt_bool_array = np.append(opt_bool_array, False)
                    
        if print_observables == True:             
            print_statements(atoms, T, divide_by_atoms = divide_by_atoms)
        
    
    return [temp_array, time_array, ekin_array, epot_array, opt_bool_array, positions_array, analytical_epot_array]



def equilibriate_structure(atoms,
                           temperature,
                           time_stamp,
                           fmax = 0.05,
                           write_interval = 10,
                           divide_by_atoms = True,
                           calculate_analytical = False):
                           
    r'''
    Equilibriates the given structure with the FIRE optimizer
    
    Args:
    
        atoms (Ase atoms object): The input atomic structure
        temperature (float): The current temperature of the system
        time_stamp (float): The current time stamp of the md simulation
        fmax (float): The maximum force to optimize to 
        write_interval (int): The interval at which observables are saved
        divide_by_atoms (Optional: True): If True then divide the PE of atoms by the number of atoms
        calculate_analytical (Optional : False) : If True then analytically calculate atoms energy using Stillinger Weber
        
    Returns:
        
        temp_array (np.array) : The whole temperature schedule (only the constant temperature)
        time_array (np.array) : The real time of molecular dynamics (only time stamps)
        ekin_array (np.array) : Contains the kinetic energy of the atom
        epot_array (np.array) : Constains the potential energy of the atom
        opt_bool_array (np.array) : Whether the optimizer was on or off
        positions_array (np.array) : positions of the atoms when performing equilibriation
        
    '''
    analytical_calculator = SiO2_Stillinger_Weber()
    
    temp_array = np.array([])
    time_array = np.array([])
    ekin_array = np.array([])
    epot_array = np.array([])
    analytical_epot_array = np.array([])
    opt_bool_array = np.array([])
    positions_array = np.empty((0, len(atoms), 3))
    
    opt = FIRE(atoms)    # Initialising the FIRE optimizer

    for kk, _ in enumerate(opt.irun(fmax = fmax, steps = 2000)):

        if((kk + 1)%write_interval == 0):
            
            positions_array = np.append(positions_array, atoms.positions[None,:,:], axis = 0)
            
            epot = atoms.get_potential_energy()   
            ekin = atoms.get_kinetic_energy() / len(atoms)    # KE per atom
            
            if divide_by_atoms == True:
                epot = epot / len(atoms)
            
            if calculate_analytical == True :             
                analytical_calculator.calculate(atoms)
                analytical_epot_array = np.append(analytical_epot_array, analytical_calculator.results["energy"]/len(atoms))    # per atom
            
            temp_array = np.append(temp_array, temperature)
            time_array = np.append(time_array, time_stamp)
            ekin_array = np.append(ekin_array, ekin)
            epot_array = np.append(epot_array, epot)
            opt_bool_array = np.append(opt_bool_array, True)
    
    atoms = reset_positions(atoms)
    
    return [temp_array, time_array, ekin_array, epot_array, opt_bool_array, positions_array, analytical_epot_array]
