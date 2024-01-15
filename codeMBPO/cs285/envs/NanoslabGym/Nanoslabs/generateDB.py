import sys
sys.path.append('/Users/krishnabhattaram/Desktop/CS285/Project/codeDQN')

import os.path

from cs285.envs.NanoslabGym.Nanoslabs.SiliconNanoslab import SiliconNanoslab
import numpy as np

from tqdm import tqdm

save_dir = '/Users/krishnabhattaram/Desktop/CS285/Project/SiliconDatabase'

n_range = [38, 80]
a_range = [1, 12]

rc = 12.5

d_n = 1
d_a = 0.1

n_n = int((n_range[1] - n_range[0]) / d_n) + 1
n_a = int((a_range[1] - a_range[0]) / d_a) + 1

for n_layers in np.arange(n_range[0], n_range[1]):
    for a_val in np.linspace(a_range[0], a_range[1], n_a):
        savename = f'Si_a={round(a_val, 4)}_n={n_layers}.npz'
        if os.path.isfile(os.path.join(save_dir, savename)):
            print('Skipping', savename, '!')
            continue

        print(n_layers, a_val)

        instance = SiliconNanoslab(n_layers=n_layers, a=a_val)
        energies = instance.get2DPeriodicEnergies_k(N=10).flatten()

        np.savez(os.path.join(save_dir, savename),
                 energies=energies,
                 n_layers=n_layers,
                 a_val=a_val,
                 rc=rc)

