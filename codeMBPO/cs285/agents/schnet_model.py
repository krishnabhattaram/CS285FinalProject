import os
import sys
import pickle

sys.path.append('/Users/krishnabhattaram/Desktop/CS285/Project/Neural-Network-Materials')

import numpy as np
import torch
from ase import Atoms
import matplotlib.pyplot as plt
import schnetpack as spk
from schnetpack.datasets import QM9
from torch.optim import Adam
import schnetpack.train as trn
from schnetpack import AtomsData
from schnetpack.environment import AseEnvironmentProvider, BondEnvironmentProvider, SimpleEnvironmentProvider
from schnetpack.data import AtomsDataError, AtomsDataSubset
from schnetpack.nn import *
from schnetpack.utils import convert_dict_to_class,  calc_gradient_clip_norm
from schnetpack import Properties
from scipy.interpolate import CubicSpline
from ase.visualize.plot import plot_atoms

import numpy as np
import schnetpack as spk
from schnetpack.datasets import QM9
from torch.optim import Adam
import schnetpack.train as trn
from schnetpack import AtomsData
from schnetpack.environment import AseEnvironmentProvider, BondEnvironmentProvider, SimpleEnvironmentProvider
from schnetpack.data import AtomsDataError, AtomsDataSubset
from schnetpack.nn import *
from schnetpack.utils import convert_dict_to_class, calc_gradient_clip_norm
import wandb
import cProfile, pstats

torch.set_num_threads(3)


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


''' setting up the profiler and stats module '''
# profiler = cProfile.Profile()
# profiler.enable()

''' WandB indicator '''
wandb_indicator = True

''' Fixing the seed if required '''
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)

''' Whether to include spherical harmonics '''
spherical_harmonics = True
both_p_and_d = True
projection_indicator = True

'''Default parameters for NN structure '''

''' Embedding table parameters '''
n_atom_basis = 8
trainable_embedding = True

''' Atomic neighbor environment '''
environment_provider = "BondEnvironmentProvider"
environment_cutoff = 5.0
cutoff = 5.0

''' Distance expansion '''
n_gaussians = 8
n_basis_functions = 8
trainable_gaussians = True

''' Filter Generators '''
n_filters = 8
cutoff_network = "GaussianCutoff"

''' SchNet interaction blocks '''
coupled_interactions = True
n_interactions = 4
normalize_filter = True

''' Default parameters for training algorithm '''
optimizer_name = "Adam"
main_objective = "smooth_dos"
main_rotated_objective = "smooth_dos"
lr = 0.01
weight_decay = 0
lr_schedule = "reducelronplateau"
lr_factor = 0.8
gradient_clipping = True
gradient_clip_norm = 5.0

device = "cpu"
n_epochs = 500
batch_size = 32
checkpoint_interval = 100
keep_n_checkpoints = 0

''' Creating the default dict for wandb '''
default_dict = {}

default_dict["spherical_harmonics"] = spherical_harmonics
default_dict["both_p_and_d"] = both_p_and_d
default_dict["projection_indicator"] = projection_indicator

default_dict["optimizer_name"] = optimizer_name
default_dict["lr"] = lr
default_dict["weight_decay"] = weight_decay
default_dict["lr_schedule"] = lr_schedule
default_dict["lr_factor"] = lr_factor
default_dict["batch_size"] = batch_size
default_dict["n_epochs"] = n_epochs

default_dict["n_atom_basis"] = n_atom_basis
default_dict["trainable_embedding"] = trainable_embedding

default_dict["environment_provider"] = environment_provider
default_dict["environment_cutoff"] = environment_cutoff

default_dict["n_gaussians"] = n_gaussians
default_dict["trainable_gaussians"] = trainable_gaussians
default_dict["n_basis_functions"] = n_basis_functions
default_dict["n_filters"] = n_filters
default_dict["cutoff_network"] = cutoff_network
default_dict["cutoff"] = cutoff

default_dict["coupled_interactions"] = coupled_interactions
default_dict["n_interactions"] = n_interactions
default_dict["normalize_filter"] = normalize_filter
default_dict["gradient_clipping"] = gradient_clipping
default_dict["gradient_clip_norm"] = gradient_clip_norm

config = convert_dict_to_class(default_dict)


''' Preparing the loss function for training '''
loss = trn.build_mse_loss([main_objective])

def get_schnet_model(n_out):
    schnet = spk.representation.AngularSchNet(n_atom_basis=config.n_atom_basis,
                                              n_filters=config.n_filters,
                                              n_basis_functions=config.n_basis_functions,
                                              n_interactions=config.n_interactions, cutoff=config.cutoff,
                                              cutoff_network=str_to_class(config.cutoff_network),
                                              trainable_embedding=config.trainable_embedding,
                                              coupled_interactions=config.coupled_interactions,
                                              environment_cutoff=config.environment_cutoff,
                                              normalize_filter=config.normalize_filter,
                                              dblock_indicator=config.both_p_and_d,
                                              projection_indicator=config.projection_indicator)

    output_U0 = spk.atomistic.Rotate_Vectorized(n_in=config.n_atom_basis,
                                                n_out=n_out,
                                                atomref=None,
                                                property=main_objective,
                                                mean=None,
                                                stddev=None,
                                                aggregation_mode='avg',
                                                rotate_indicator=False,
                                                rotate_property=main_rotated_objective,
                                                rotate_basis=None,
                                                rotate_std=None,
                                                rotate_mean=None,
                                                remove_elements_list=[1])

    return spk.AtomisticModel(representation=schnet, output_modules=output_U0)
