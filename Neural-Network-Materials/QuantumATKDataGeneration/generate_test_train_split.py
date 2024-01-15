import os
import sys

current_dir = os.getcwd()
root_dir = os.path.dirname(current_dir)
struct_dir = os.path.join(current_dir, "nanosheet_silicon/utils")
sys.path.append(root_dir)
sys.path.append(struct_dir)

import numpy as np
from schnetpack import AtomsData

root_dir = '/home/krishna_bhattaram/Desktop/Neural-Network-Materials-main'

holdout_layers = [13, 19]

dataset_dir = os.path.join(root_dir, 'Data/QuantumATK/InjectionVelocity/')
dataset_filename = os.path.join(dataset_dir, 'Jx_DOS_with_n_layers_nanosheet_data.db')
hyperparameter_split_filename = os.path.join(dataset_dir,
                                             f"Jx_DOS_with_n_layers_nanosheet_holdout_layers_{holdout_layers}_hyperparameter_split.npz")

silicon_full_dataset = AtomsData(dataset_filename)
num_total = len(silicon_full_dataset)

test_indices = np.array([i for i in range(num_total) if silicon_full_dataset[i]['n_layers'] in holdout_layers])
keep_indices = np.array([i for i in range(num_total) if silicon_full_dataset[i]['n_layers'] not in holdout_layers])

''' Generating the training, validation and test data '''

num_train = len(keep_indices) - 1

keep_idx = np.random.permutation(keep_indices)
train_idx = keep_idx[0:num_train]
val_idx = keep_idx[num_train:]

test_idx = np.random.permutation(test_indices)
np.savez(hyperparameter_split_filename, train_idx=train_idx, test_idx=val_idx, val_idx=test_idx)
