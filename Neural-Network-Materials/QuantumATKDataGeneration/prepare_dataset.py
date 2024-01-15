import re

from ase import Atoms
import os
import numpy as np
import sys
import matplotlib.pyplot as plt

from scipy import constants
from scipy.stats import norm
from scipy import integrate
from scipy import interpolate

current_dir = os.getcwd()
root_dir = os.path.dirname(current_dir)
struct_dir = os.path.join(current_dir, "nanosheet_silicon/utils")
sys.path.append(root_dir)
sys.path.append(struct_dir)

from schnetpack import AtomsData
from schnetpack import Properties
from schnetpack.environment import AseEnvironmentProvider

from tqdm import tqdm
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

number_to_symbol = {1: 'H', 2: 'He', 14: 'Si'}


def get_symbol_from_atomic_number(atomic_number): return number_to_symbol.get(atomic_number, -1)


def broaden_e(k_points, E_eV, val, energies, broadening_eV=0.01, sign=0, verbose=False):
    smooth_vals = np.zeros_like(energies)

    perrank = len(k_points) // size
    lower_bound = rank * perrank
    upper_bound = (rank + 1) * perrank

    comm.barrier()

    if rank == size - 1:
        upper_bound = len(k_points)

    if rank == size - 1 and verbose:
        iteration = tqdm(range(lower_bound, upper_bound))
    else:
        iteration = range(lower_bound, upper_bound)

    for k_ind in iteration:
        for band, E in enumerate(E_eV(k_ind)):
            if E < energies[0] - 2 * broadening_eV or E > energies[-1] + 2 * broadening_eV:
                continue

            value = val(k_ind, band)
            use_val = ((value <= 0) if sign < 0 else ((value >= 0) if sign > 0 else True))

            if not use_val:
                continue

            smooth_vals += norm(E, broadening_eV).pdf(energies) * value

    comm.barrier()
    smooth_vals = comm.reduce(smooth_vals, op=MPI.SUM, root=0)

    if rank == 0:
        return smooth_vals / len(k_points)
    return None


def get_k_space_velocities(k_points_fractional, band_structure, primitive_vectors):
    """
    Assumes constant spacing in k_points_fractional in both x- and y-dimension!
    Works for FCC or SC lattice
    """
    k_pts = int(np.sqrt(k_points_fractional.shape[0]))

    dkx = 2 * np.pi / (primitive_vectors[0, 0] * 1e-10)  # 1/m
    dky = 2 * np.pi / (primitive_vectors[1, 1] * 1e-10)  # 1/m

    all_eval_bs_2d = band_structure.reshape((k_pts, k_pts, -1))
    k_points_2d = k_points_fractional.reshape((k_pts, k_pts, -1))

    band_structure_band_vels = []

    pad_width = 2
    for band in range(all_eval_bs_2d.shape[-1]):
        band_structure_vel = np.transpose(np.array(np.gradient(np.pad(all_eval_bs_2d[:, :, band], mode='wrap',
                                                                      pad_width=pad_width),
                                                               (k_points_2d[1, 0, 0] - k_points_2d[0, 0, 0]) * dkx,
                                                               (k_points_2d[0, 1, 1] - k_points_2d[0, 0, 1]) * dky)),
                                          axes=(1, 2, 0))[pad_width:-pad_width,
                             pad_width:-pad_width] * constants.e / constants.hbar

        band_structure_band_vels += [band_structure_vel]

    return np.array(band_structure_band_vels) * 1e2 * 1e-7  # m/s to cm/s, then scaled to reasonable range


def plot_vinj():
    def integrate_over_E(f, energies_eV, dos_eV, fermi_level_eV, T_K=300):
        kb = 8.617e-5  # eV/K
        fermi_dirac_distribution = 1 / (1 + np.exp((energies_eV - fermi_level_eV) / (kb * T_K)))

        return integrate.trapezoid(f * dos_eV * fermi_dirac_distribution, energies_eV)

    def integrate_over_E_wo_DOS(f, energies_eV, fermi_level_eV, T_K=300):
        kb = 8.617e-5  # eV/K
        fermi_dirac_distribution = 1 / (1 + np.exp((energies_eV - fermi_level_eV) / (kb * T_K)))

        return integrate.trapezoid(f * fermi_dirac_distribution, energies_eV)

    fermi_levels = np.linspace(0.5, 1.0, 100)

    Ns = []
    v_injs = []

    if rank == 0:
        plt.figure()
        plt.plot(energies_zeroed, smooth_positive_Jx)
        plt.ylabel('Jy (cm/s (1e7))')
        plt.xlabel('Energies (eV)')
        plt.show()

        plt.figure()
        plt.plot(energies_zeroed, interp_dos/num_Si)
        plt.ylabel('DOS (cm/s (1e7))')
        plt.xlabel('Energies (eV)')
        plt.show()

    for fermi_level in fermi_levels:
        A = primitive_vectors[0, 0] * primitive_vectors[1, 1] * 1e-16  # Angstroms sq to cm sq
        N = 0.5 * integrate_over_E(f=np.ones(energies.shape), energies_eV=energies, dos_eV=interp_dos,
                                   fermi_level_eV=fermi_level)

        I1 = integrate_over_E_wo_DOS(f=smooth_positive_Jx, energies_eV=energies, fermi_level_eV=fermi_level)

        Ns += [N / A]
        v_injs += [I1 / N]

        if rank == 0:
            print(f'Fermi Level: {fermi_level} eV')
            print(f'N: {N}')
            print(f'N per Area: {N / A} cm^-2')
            print(f'I1: {I1} 1/s')
            print(f'v_inj: {I1 / N} m/s')

    if rank == 0:
        plt.figure()
        plt.title(f'Energies Range {(np.min(energies), np.max(energies))} eV')
        plt.plot(fermi_levels, Ns, label='Gaussian')
        plt.yscale('log')
        plt.ylabel('N/A ($cm^{-2}$)')
        plt.xlabel('Fermi Level (eV)')
        plt.show()

        plt.figure()
        plt.title(f'Energies Range {(np.min(energies), np.max(energies))} eV')
        plt.xlabel('N ($cm^{-2}$)')
        plt.ylabel('$v_{inj}$ (m/s)')
        plt.plot(Ns, v_injs, label='Gaussian')
        plt.xscale('log')
        plt.show()


# Angstroms environment cutoff
cutoff = 5.0

# Conduction band minimum expected to lie in this range (absolute energy units)
Ec_min = 0.0
Ec_max = 1.5
dos_threshold = 1e-12  # Minimum value below which DOS is assumed to be zero

# Energy range relative to E_c
energy_below_Ec = 0.3
energy_above_Ec = 0.3
n_points = 200

environment_provider = AseEnvironmentProvider(cutoff)
property_list = []
atoms_list = []

root_dir = '/home/krishna_bhattaram/Desktop/Neural-Network-Materials-main'
read_dir = '/home/krishna_bhattaram/.quantumatk/projects/CrystallineSiliconSlab/RunData'

dataset_dir = os.path.join(root_dir, 'Data/QuantumATK/InjectionVelocity/')
dataset_filename = os.path.join(dataset_dir, 'Jx_DOS_with_n_layers_nanosheet_data.db')

iterate_vals = list(reversed(os.listdir(read_dir)))
if rank == 0:
    iterator = tqdm(iterate_vals)
else:
    iterator = iterate_vals

num_total = 0
for file in iterator:
    comm.barrier()
    n_layers = int(re.search('Ang_([0-9]+)_layers', file).group(1))

    # Load data
    data = np.load(os.path.join(read_dir, file))
    atom_positions = data['positions_angstroms']
    atomic_numbers = data['atomic_numbers']
    primitive_vectors = data['primitive_vectors_angstroms']
    dos_energies = data['dos_energies_eV']
    dos_values = data['dos_values_inv_eV']
    fermi_level = data['dos_fermi_energy_eV']
    k_points_fractional = data['k_points_fractional']
    band_structure_energies = data['band_structure_energies_eV']

    # Remove redundant spin-dependent DOS information from ATK if present
    if dos_values.shape[0] == 2:
        np.testing.assert_allclose(dos_values[0], dos_values[1])
        dos_values = dos_values[0]


    # Calculate E_c
    truncated_energy_indices = np.where((dos_energies >= Ec_min) & (dos_energies <= Ec_max))[0]
    dos_Ec_range = dos_values[truncated_energy_indices]
    Ec = dos_energies[truncated_energy_indices][np.where(dos_Ec_range < dos_threshold)[0][-1]]

    # Energy spectrum being used
    energies = np.linspace(Ec - energy_below_Ec, Ec + energy_above_Ec, n_points)
    interp_dos = interpolate.CubicSpline(dos_energies, dos_values)(energies)

    # Calculate J
    velocities_2d = get_k_space_velocities(k_points_fractional=k_points_fractional,
                                           band_structure=band_structure_energies,
                                           primitive_vectors=primitive_vectors)
    velocities_flattened = velocities_2d.reshape((velocities_2d.shape[0], -1, 2))

    # Multiprocessed function
    comm.barrier()
    smooth_positive_Jx = broaden_e(k_points_fractional, lambda k_ind: band_structure_energies[k_ind],
                                   lambda k_ind, band: velocities_flattened[band, k_ind, 0],
                                   energies=energies, broadening_eV=0.01, sign=1)
    comm.barrier()

    # Reduce memory burden by skipping unnecessary calculation and storage for most threads
    if rank == 0:
        # ASE
        atoms = Atoms(symbols=[get_symbol_from_atomic_number(n) for n in atomic_numbers],
                      positions=atom_positions,
                      cell=primitive_vectors,
                      pbc=[True, True, False])
        neighborhood_idx, offset = environment_provider.get_environment(
            atoms)  # Calculating neighbors

        # Lattice area
        lat_area = primitive_vectors[0, 0] * primitive_vectors[1, 1] * 1e-16  # For converting Angstrom^2 to cm^2

        # Shift energies to zero at Ec
        energies_zeroed = energies - Ec
        num_Si = np.sum(np.array(atoms.get_chemical_symbols()) == 'Si')

        property_dict = {Properties.neighbors: neighborhood_idx, "neighbors": neighborhood_idx, "cell_offset": offset,
                         "energies": energies_zeroed, "dosperatom": interp_dos / num_Si, "Jx": smooth_positive_Jx,
                         "lat_vol": lat_area, "Ec_abs_energy": Ec, "Ef_abs_energy": fermi_level, 'n_layers': n_layers,
                         'lat_const': primitive_vectors[0, 0]}

        atoms_list.append(atoms)
        property_list.append(property_dict)

        num_total += 1

    comm.barrier()

print(f'Process {rank} exited loop!')

if rank == 0:
    if os.path.exists(dataset_filename):
        os.system('rm ' + dataset_filename)

    silicon_full_dataset = AtomsData(dataset_filename, available_properties=property_list[0].keys())
    silicon_full_dataset.add_systems(atoms_list, property_list)

print(f'Process {rank} Done!')

