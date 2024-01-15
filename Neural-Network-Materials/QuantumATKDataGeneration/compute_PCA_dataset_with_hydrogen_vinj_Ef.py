import os

os.environ["OMP_NUM_THREADS"] = "6"
import sys
import numpy as np

import copy
import matplotlib.pyplot as plt
from scipy import integrate

current_dir = os.getcwd()
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from schnetpack import AtomsData


def integrate_over_E(f, energies_eV, dos_eV, fermi_level_eV, T_K=300):
    kb = 8.617e-5  # eV/K
    fermi_dirac_distribution = 1 / (1 + np.exp((energies_eV - fermi_level_eV) / (kb * T_K)))

    return integrate.trapezoid(f * dos_eV * fermi_dirac_distribution, energies_eV)


def integrate_over_E_wo_DOS(f, energies_eV, fermi_level_eV, T_K=300):
    kb = 8.617e-5  # eV/K
    fermi_dirac_distribution = 1 / (1 + np.exp((energies_eV - fermi_level_eV) / (kb * T_K)))

    return integrate.trapezoid(f * fermi_dirac_distribution, energies_eV)


def calc_injection_velocity(vel, dos, energies, Ef, A):
    N = 0.5 * integrate_over_E(f=np.ones(energies.shape), energies_eV=energies, dos_eV=dos,
                               fermi_level_eV=Ef)

    I1 = integrate_over_E_wo_DOS(f=vel, energies_eV=energies, fermi_level_eV=Ef)

    return N / A, I1 / N


def compute_PCA(array):
    S = np.transpose(array) @ array
    w, v = np.linalg.eig(S)
    w = np.real(w)
    v = np.real(v)

    return w, v


threshold = 0.05
cutoff = 5.0
num_Ef_points = 20

root_dir = '/home/krishna_bhattaram/Desktop/Neural-Network-Materials-main'
read_dir = '/home/krishna_bhattaram/.quantumatk/projects/CrystallineSiliconSlab/RunData'

dataset_dir = os.path.join(root_dir, 'Data/QuantumATK/InjectionVelocity/')
dataset_filename = os.path.join(dataset_dir, 'Jx_DOS_with_n_layers_nanosheet_data.db')
hyperparameter_split_filename = os.path.join(dataset_dir, "Jx_DOS_with_n_layers_nanosheet_holdout_layers_[13, 19]_hyperparameter_split.npz")

pca_filename = os.path.join(dataset_dir, "Jx_DOS_with_n_layers_nanosheet_vinj_pca_info.npz")
means_std_filename = os.path.join(dataset_dir, "Jx_DOS_with_n_layers_nanosheet_vinj_means_std.npz")

# dataset_dir = os.path.join(root_dir, 'Data/QuantumATK/InjectionVelocity/')
# dataset_filename = os.path.join(dataset_dir, 'injection_velocity_only_conduction_band_cutoff_' + str(
#     cutoff) + '_with_hydrogen_nanosheet.db')
# hyperparameter_split_filename = os.path.join(dataset_dir, "injection_velocity_only_conduction_band_cutoff_" + str(
#     cutoff) + "_with_hydrogen_nanosheet_hyperparameter_split.npz")
# pca_filename = os.path.join(dataset_dir, "injection_velocity_only_conduction_band_cutoff_" + str(
#     cutoff) + "_with_hydrogen_nanosheet_vinj_pca_info.npz")
# means_std_filename = os.path.join(dataset_dir, "injection_velocity_only_conduction_band_cutoff_" + str(
#     cutoff) + "_with_hydrogen_nanosheet_vinj_means_std.npz")

energy_lo = 0.3
energy_hi = 0.3

split_data = np.load(hyperparameter_split_filename)
train_idx = split_data["train_idx"]
val_idx = split_data["val_idx"]
test_idx = split_data["test_idx"]

silicon_dataset = AtomsData(dataset_filename)
dos_array = []
Jx_array = []

for ii in train_idx:
    atoms, properties = silicon_dataset.get_properties(int(ii))

    dos_array.append(properties['dosperatom'])
    Jx_array.append(properties['Jx'])

######### PCA for DOS ##############

dos_array = np.array(dos_array, dtype=float)
Jx_array = np.array(Jx_array, dtype=float)

mean_dos_array = np.mean(dos_array, axis=0)
std_dos_array = 1.0
standard_dos_array = (dos_array - mean_dos_array) / std_dos_array
dos_PCA_eigens, dos_PCA_vectors = compute_PCA(standard_dos_array)

# Sorting the PCA eigen values
sorted_index = np.flip(np.argsort(dos_PCA_eigens))
dos_PCA_eigens = dos_PCA_eigens[sorted_index]
dos_PCA_vectors = dos_PCA_vectors[:, sorted_index]

############## PCA for J #############

mean_Jx_array = np.mean(Jx_array, axis=0)
std_Jx_array = 1.0  # np.std(vel_array, axis = 0)
standard_Jx_array = (Jx_array - mean_Jx_array) / std_Jx_array
Jx_PCA_eigens, Jx_PCA_vectors = compute_PCA(standard_Jx_array)

# Sorting the PCA eigen values
sorted_index = np.flip(np.argsort(Jx_PCA_eigens))
Jx_PCA_eigens = Jx_PCA_eigens[sorted_index]
Jx_PCA_vectors = Jx_PCA_vectors[:, sorted_index]

np.savez(pca_filename,
         dos_PCA_vectors=dos_PCA_vectors,
         dos_PCA_eigens=dos_PCA_eigens,
         mean_dos_array=mean_dos_array,
         std_dos_array=std_dos_array,
         Jx_PCA_vectors=Jx_PCA_vectors,
         Jx_PCA_eigens=Jx_PCA_eigens,
         mean_Jx_array=mean_Jx_array,
         std_Jx_array=std_Jx_array)

# Calculating the reconstruction error
PCA_dos_array = standard_dos_array @ dos_PCA_vectors
error = np.array([])
for ii in range(dos_PCA_vectors.shape[1]):
    reconstruct_dos_array = std_dos_array * (PCA_dos_array[:, 0:ii] @ dos_PCA_vectors[:, 0:ii].T) + mean_dos_array
    error = np.append(error, np.mean(abs(reconstruct_dos_array - dos_array)))

dos_threshold_index = np.where(error <= 1e-3)[0][0]
print("The index at threshold 1e-3: ", np.where(error <= 1e-3)[0][0])
fig, ax = plt.subplots()
ax.plot(error)
ax.set_ylabel("Reconstruction Error of DOS", fontsize=15)
ax.set_xlabel("Number of PCA vectors", fontsize=15)
plt.show()

PCA_Jx_array = standard_Jx_array @ Jx_PCA_vectors
error = np.array([])
for ii in range(Jx_PCA_vectors.shape[1]):
    reconstruct_Jx_array = std_Jx_array * (PCA_Jx_array[:, 0:ii] @ Jx_PCA_vectors[:, 0:ii].T) + mean_Jx_array
    error = np.append(error, np.mean(abs(reconstruct_Jx_array - Jx_array)))

Jx_threshold_index = np.where(error <= 1e-3)[0][0]
print("The index at threshold 1e-3: ", np.where(error <= 1e-3)[0][0])
fig, ax = plt.subplots()
ax.plot(error)
ax.set_ylabel("Reconstruction Error of Jx", fontsize=15)
ax.set_xlabel("Number of PCA vectors", fontsize=15)
plt.show()

############### Creating a new dataset to replace the new one and calculate pca means #########

atoms_list = []
properties_list = []

threshold_pca_dos_array = []
threshold_pca_Jx_array = []

smooth_dos_array = []
smooth_Jx_array = []

del_idx = []  # The indices which need to be deleted after new dataset is formed
# Like the nanoslab silicon [1, 1, 1] surface axis

for ii in range(len(silicon_dataset)):
    atoms, properties = silicon_dataset.get_properties(int(ii))
    num_Si = np.sum(np.array(atoms.get_chemical_symbols()) == 'Si')

    area = properties['lat_vol']
    smooth_energies = properties['energies']
    smooth_dos = properties['dosperatom']
    smooth_Jx = properties['Jx']

    pca_dos = ((smooth_dos - mean_dos_array) / std_dos_array)[None, :] @ dos_PCA_vectors
    pca_Jx = ((smooth_Jx - mean_Jx_array) / std_Jx_array)[None, :] @ Jx_PCA_vectors

    threshold_pca_dos = pca_dos[0, 0:dos_threshold_index]
    threshold_pca_Jx = pca_Jx[0, 0:Jx_threshold_index]

    properties["smooth_energies"] = smooth_energies
    properties["smooth_dos"] = smooth_dos
    properties["smooth_Jx"] = smooth_Jx
    properties["pca_dos"] = pca_dos
    properties["pca_Jx"] = pca_Jx
    properties["threshold_pca_dos"] = threshold_pca_dos
    properties["threshold_pca_Jx"] = threshold_pca_Jx

    ### Finding out injection velocity with changing Ef

    Ef_array = np.linspace(-0.2, 0.4, num_Ef_points)
    for ff in range(Ef_array.size):
        Ef = Ef_array[ff]
        N1, vinj = calc_injection_velocity(vel=smooth_Jx, dos=smooth_dos * num_Si,
                                           energies=smooth_energies, Ef=Ef, A=area)
        add_properties = copy.deepcopy(properties)
        add_properties["vinj"] = vinj
        add_properties["Ef"] = Ef
        add_properties["N1"] = N1

        atoms_list.append(atoms)
        properties_list.append(add_properties)
        properties = add_properties

    if ((ii in train_idx) and (ii not in del_idx)):  # Calculating the mean and std for training data points
        threshold_pca_dos_array.append(threshold_pca_dos)
        threshold_pca_Jx_array.append(threshold_pca_Jx)

        smooth_dos_array.append(smooth_dos)
        smooth_Jx_array.append(smooth_Jx)

threshold_pca_dos_array = np.array(threshold_pca_dos_array)
threshold_pca_Jx_array = np.array(threshold_pca_Jx_array)

smooth_dos_array = np.array(smooth_dos_array)
smooth_Jx_array = np.array(smooth_Jx_array)

means_threshold_pca_dos = np.mean(threshold_pca_dos_array, axis=0)
std_threshold_pca_dos = np.std(threshold_pca_dos_array, axis=0)

means_smooth_dos = np.mean(smooth_dos_array, axis=0)
std_smooth_dos = np.std(smooth_dos_array, axis=0)

means_threshold_pca_Jx = np.mean(threshold_pca_Jx_array, axis=0)
std_threshold_pca_Jx = np.std(threshold_pca_Jx_array, axis=0)

means_smooth_Jx = np.mean(smooth_Jx_array, axis=0)
std_smooth_Jx = np.std(smooth_Jx_array, axis=0)

np.savez(means_std_filename,
         means_threshold_pca_dos=means_threshold_pca_dos,
         std_threshold_pca_dos=std_threshold_pca_dos,
         means_threshold_pca_Jx=means_threshold_pca_Jx,
         std_threshold_pca_Jx=std_threshold_pca_Jx,
         means_smooth_dos=means_smooth_dos,
         std_smooth_dos=std_smooth_dos,
         means_smooth_Jx=means_smooth_Jx,
         std_smooth_Jx=std_smooth_Jx)

##### Creating the new dataset #########
dataset_dir = os.path.join(root_dir, 'Data/QuantumATK/InjectionVelocity/')
dataset_filename = os.path.join(dataset_dir, 'Jx_DOS_with_n_layers_nanosheet_vinj.db')
hyperparameter_split_filename = os.path.join(dataset_dir, "Jx_DOS_with_n_layers_nanosheet_vinj_hyperparameter_split.npz")

if (os.path.exists(dataset_filename)):
    os.system('rm ' + dataset_filename)

silicon_dataset = AtomsData(dataset_filename, available_properties=properties.keys())
silicon_dataset.add_systems(atoms_list, properties_list)
num_total = len(silicon_dataset)

#### Changing the hyperparameter split file with inclusion of different Ef levels
print(del_idx)
del_train_idx = np.setdiff1d(train_idx, del_idx)  # Removing the elements that need to be deleted
new_train_idx = del_train_idx * num_Ef_points  # Since we have num_Ef copies for each structure
new_train_idx = np.repeat(new_train_idx, num_Ef_points)
idx_add = np.arange(num_Ef_points)
idx_add = np.tile(idx_add, len(del_train_idx))
new_train_idx += idx_add

del_val_idx = np.setdiff1d(val_idx, del_idx)  # Removing the elements that need to be deleted
new_val_idx = del_val_idx * num_Ef_points  # Since we have num_Ef copies for each structure
new_val_idx = np.repeat(new_val_idx, num_Ef_points)
idx_add = np.arange(num_Ef_points)
idx_add = np.tile(idx_add, len(del_val_idx))
new_val_idx += idx_add

del_test_idx = np.setdiff1d(test_idx, del_idx)  # Removing the elements that need to be deleted
new_test_idx = del_test_idx * num_Ef_points  # Since we have num_Ef copies for each structure
new_test_idx = np.repeat(new_test_idx, num_Ef_points)
idx_add = np.arange(num_Ef_points)
idx_add = np.tile(idx_add, len(del_test_idx))
new_test_idx += idx_add

np.savez(hyperparameter_split_filename,
         train_idx=new_train_idx,
         val_idx=new_val_idx,
         test_idx=new_test_idx)

example = silicon_dataset.get_atoms(0)
print(example.info)

print("Number of reference calculations:", len(silicon_dataset))
print("Available parameters:")

for p in silicon_dataset.available_properties:
    print("-", p)

example = silicon_dataset.get_properties(0)[1]
print("Properties of molecule with id 0:")

for k, v in example.items():
    print(v)
    print("-", k, ":", v.shape)
