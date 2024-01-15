import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag


def plot_dos(energies, title='DoS', bins=50):
    energies = np.reshape(energies, (-1, 1))
    plt.figure()
    histvals = plt.hist(energies, bins=bins, density=True)
    plt.title(title)
    plt.xlabel('E')
    plt.ylabel('count')
    plt.savefig(title.replace('.', '') + '.png')
    plt.show()

    return histvals


def get_dos(energies, n_bins=10):
    counts, bins = np.histogram(energies, bins=n_bins, density=True, range=(-3, 4))

    return counts


def hist_mse(b1, b2, err=1e-2):
    err = 1e-2

    bins1, counts1, bars1 = b1
    bins2, counts2, bars2 = b2

    if (any(np.abs(bins1 - bins2) > err)):
        print('Bin match beyond error! Max diff: ', np.max(np.abs(bins1 - bins2)))

    return np.mean(np.square(counts1 - counts2))


def murnagham_eq(a0, P, B0, B0_prime):
    return a0 * np.power(1 + B0_prime * P / B0, -1 / (3 * B0_prime))


def build_diagonal_frame(N):
    return np.diag(N * [1])


def build_upper_interaction_frame(N):
    return np.diag((N - 1) * [1], 1) + np.diag([1], 1 - N)


def build_lower_interaction_frame(N):
    return build_upper_interaction_frame(N).T


def build_periodic_H(eps, t, N):
    if N <= 2:
        return eps * build_diagonal_frame(N) + t * build_upper_interaction_frame(N)
    return eps * build_diagonal_frame(N) + t * (build_upper_interaction_frame(N) + build_lower_interaction_frame(N))


def build_block_periodic_H(A, B, N):
    return np.array(block_diag(*[A for _ in range(N)])) + np.kron(build_upper_interaction_frame(N), B) \
           + np.kron(build_lower_interaction_frame(N), B.T)
