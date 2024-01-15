import os
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print(f'Process {rank} running!')

def get_crystalline_si(n_layers, a=5.431 * Angstrom, h=1, k=0, l=0,
                       vacuum_gap=20 * Angstrom, should_passivate=True):

    lattice = FaceCenteredCubic(a)
    elements = [Silicon, Silicon]

    fractional_coordinates = [[0, 0, 0],
                              [0.25, 0.25, 0.25]]
    bulk_si_structure = BulkConfiguration(
        bravais_lattice=lattice,
        elements=elements,
        fractional_coordinates=fractional_coordinates
    )

    # Layer count divided by two because two atoms in unit cell
    cleave_parameters = {'h': h, 'k': k, 'l': l, 'layers': n_layers / 2,
                         'top_vacuum': vacuum_gap, 'bottom_vacuum': 0.0 * Angstrom}

    slab = bulk_si_structure.cleave(**cleave_parameters).center()

    if should_passivate:
        slab = passivate(slab, All)

    return slab

def add_calculator(geometry, calculator_type='SE', k_points=10, density_mesh_cutoff=10.0 * Hartree,
                   **calculator_parameters):

    k_point_sampling = MonkhorstPackGrid(
        na=k_points,
        nb=k_points,
    )
    numerical_accuracy_parameters = NumericalAccuracyParameters(
        k_point_sampling=k_point_sampling,
        density_mesh_cutoff=density_mesh_cutoff,
    )

    if calculator_type == 'SE':
        if calculator_parameters.get('SCC_on', True):
            iteration_control_parameters = IterationControlParameters()
        else:
            iteration_control_parameters = None

        hamiltonian_parametrization = SlaterKosterHamiltonianParametrization(
            basis_set=Bassani.SiH_Basis)
        calculator = SemiEmpiricalCalculator(
            hamiltonian_parametrization=hamiltonian_parametrization,
            numerical_accuracy_parameters=numerical_accuracy_parameters,
            iteration_control_parameters=iteration_control_parameters,
        )
    elif calculator_type == 'LCAO':
        calculator = LCAOCalculator(
            numerical_accuracy_parameters=numerical_accuracy_parameters,
        )
    else:
        raise ValueError(f'Calculator type {calculator_type} is unknown!')

    geometry.setCalculator(calculator)
    geometry.update()

    return

def generate_dos(geometry, k_points=150, method='TETRAHEDRON', **calc_params):
    broadening = calc_params.get('broadening', 0.01 * eV)

    k_point_grid = MonkhorstPackGrid(
        na=k_points,
        nb=k_points,
    )

    density_of_states = DensityOfStates(
        configuration=geometry,
        kpoints=k_point_grid,
        energy_zero_parameter=FermiLevel,
        bands_above_fermi_level=All,
        method=Full,
    )

    if method == "TETRAHEDRON":
        if k_points * k_points > 10:
            dos_values = density_of_states.evaluate()
        else:
            dos_values = density_of_states.tetrahedronSpectrum()
    elif method == 'GAUSSIAN':
        dos_values = density_of_states.gaussianSpectrum(broadening=broadening)
    else:
        raise ValueError(f'Method {method} is unknown!')

    dos_energies = density_of_states.energies()
    dos_energy_fermi = density_of_states.fermiLevel()

    return dos_energies, dos_values, dos_energy_fermi

def generate_bandstructure(geometry, k_points=150):
    k_point_grid = MonkhorstPackGrid(
        na=k_points,
        nb=k_points,
    )

    k_points_vals = k_point_grid.allKpoints()
    band_structure = Bandstructure(configuration=geometry,
                                   kpoints=k_points_vals)
    band_structure_vals = band_structure.evaluate()

    return k_points_vals, band_structure_vals


FORCE_OVERWRITE = False

h, k, l = 1, 0, 0
k_points = 150

a = 5.431  * Angstrom
a_scaling_range = [0.9, 1.035]
a_delta = 0.002
a_values = np.linspace(a_scaling_range[0], a_scaling_range[1],
                       int((a_scaling_range[1] - a_scaling_range[0])/a_delta)) * a

n_layer_range = [5, 19]
n_layer_delta = 1
n_layer_values = np.array(range(n_layer_range[0], n_layer_range[1] + 1, n_layer_delta))

i = 1

for n_layers in n_layer_values:
    for a_val in a_values:

        #######
        comm.barrier()
        skip = False
        save_file = f"/home/krishna_bhattaram/.quantumatk/projects/CrystallineSiliconSlab/" \
                    f"RunData/a={a_val.inUnitsOf(Angstrom):.4f}_Ang_{n_layers}_layers_{h}x{k}x{l}.npz"

        if rank == 0:
            print(f'On structure {i} of {len(n_layer_values) * len(a_values)}.....')
            print(f'a = {a_val} n_layers = {n_layers}')

            if os.path.isfile(save_file) and not FORCE_OVERWRITE:
                print('Skipping existing structure!')
                skip = True

        skip = comm.bcast(skip, root=0)
        if skip:
            print('Rank {} skipping iteration {}'.format(rank, i))
            i += 1
            continue

        #######
        comm.barrier()
        structure = get_crystalline_si(n_layers=n_layers, a=a_val, h=h, k=k, l=l)
        #######
        comm.barrier()
        add_calculator(geometry=structure, SCC_on=True)
        #######
        comm.barrier()
        dos_energies, dos_values, dos_energy_fermi = generate_dos(geometry=structure, k_points=k_points, method='TETRAHEDRON')
        #######
        comm.barrier()
        k_points_vals, band_structure_vals = generate_bandstructure(geometry=structure, k_points=k_points)
        #######
        comm.barrier()

        if rank == 0:
            structure_positions = structure.cartesianCoordinates().inUnitsOf(Angstrom)
            structure_atomic_numbers = np.array(structure.atomicNumbers())
            primitive_vectors = structure.primitiveVectors().inUnitsOf(Angstrom)

            dos_energies_eV = dos_energies.inUnitsOf(eV)
            dos_values_inv_eV = dos_values.inUnitsOf(eV**-1)
            dos_energy_fermi_eV = dos_energy_fermi.inUnitsOf(eV)

            band_structure_vals_eV = band_structure_vals.inUnitsOf(eV)

            np.savez(save_file,
                     positions_angstroms = structure_positions,
                     atomic_numbers = structure_atomic_numbers,
                     primitive_vectors_angstroms = primitive_vectors,
                     dos_energies_eV = dos_energies_eV,
                     dos_values_inv_eV = dos_values_inv_eV,
                     dos_fermi_energy_eV = dos_energy_fermi_eV,
                     k_points_fractional = k_points_vals,
                     band_structure_energies_eV = band_structure_vals_eV)

        i += 1
