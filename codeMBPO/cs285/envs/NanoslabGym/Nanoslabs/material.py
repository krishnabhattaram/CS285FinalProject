from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import block_diag

from codeMBPO.cs285.envs.NanoslabGym.Nanoslabs.helpers import build_block_periodic_H
from codeMBPO.cs285.envs.NanoslabGym.Nanoslabs.parameters import HarrisonScaledParameter


class Material(ABC):
    def __init__(self, ax0, ay0, az0, ax, ay, az):
        # Lattice Shape
        self.ax0 = ax0
        self.ax = ax
        self.ay0 = ay0
        self.ay = ay
        self.az0 = az0
        self.az = az

    @abstractmethod
    def getOnsiteTerm(self):
        pass

    @abstractmethod
    def getInteractionTerm(self, r0, r, index=0):
        pass

    def get1DPeriodicH(self, N):
        alpha1 = self.getOnsiteTerm()
        betaX = self.getInteractionTerm(self.ax0, self.ax, 0)
        return build_block_periodic_H(alpha1, betaX, N)

    def get2DPeriodicH(self, N, M):
        alpha2 = self.get1DPeriodicH(N)
        betaY = np.array(block_diag(*[self.getInteractionTerm(self.ay0, self.ay, 1) for _ in range(N)]))
        return build_block_periodic_H(alpha2, betaY, M)

    def get3DPeriodicH(self, N, M, P):
        alpha3 = self.get2DPeriodicH(N, M)
        betaZ = np.array(block_diag(*[self.getInteractionTerm(self.az0, self.az, 2) for _ in range(N * M)]))
        return build_block_periodic_H(alpha3, betaZ, P)

    def get1DPeriodicH_k(self, kxax):
        return self.getOnsiteTerm() + np.exp(1j * kxax) * self.getInteractionTerm(self.ax0, self.ax, 0) \
               + np.exp(-1j * kxax) * self.getInteractionTerm(self.ax0, self.ax, 0).T

    def get2DPeriodicH_k(self, kxax, kyay):
        return self.get1DPeriodicH_k(kxax) + np.exp(1j * kyay) * self.getInteractionTerm(self.ay0, self.ay, 1) \
               + np.exp(-1j * kyay) * self.getInteractionTerm(self.ay0, self.ay, 1).T

    def get3DPeriodicH_k(self, kxax, kyay, kzaz):
        return self.get2DPeriodicH_k(kxax, kyay) + np.exp(1j * kzaz) * self.getInteractionTerm(self.az0, self.az, 2) \
               + np.exp(-1j * kzaz) * self.getInteractionTerm(self.az0, self.az, 2).T

    # Note that ax/ay/az makes no difference for the NN SC
    # structure (multiplies out), but could be useful
    # to include in the signature when overriding
    def get1DPeriodicEnergies_k(self, N=1000, ax=None, ay=None, az=None):
        if not ax: ax = self.ax

        bx = 2 * np.pi / ax
        kx = np.linspace(-bx / 2, bx / 2, N)

        def eH(kx):
            H = self.get1DPeriodicH_k(kx * ax)
            if H.ndim == 1:
                H = H.reshape((1, -1))
            return np.sort(np.array([evl.real for evl in np.linalg.eigvalsh(H)]))

        return np.array([eH(k) for k in kx])

    def get2DPeriodicEnergies_k(self, N=50, ax=None, ay=None, az=None):
        if not ax: ax = self.ax
        if not ay: ay = self.ay

        bx = 2 * np.pi / self.ax
        by = 2 * np.pi / self.ay

        kx = np.linspace(-bx / 2, bx / 2, N)
        ky = np.linspace(-by / 2, by / 2, N)
        X, Y = np.meshgrid(kx, ky)

        def eH(kx, ky):
            H = self.get2DPeriodicH_k(kx * ax, ky * ay)
            if H.ndim == 1:
                H = H.reshape((1, -1))
            return np.sort(np.array([evl.real for evl in np.linalg.eigvalsh(H)]))

        return np.array([[eH(X[i, j], Y[i, j]) for i in range(X.shape[0])] for j in range(X.shape[1])])

    def get3DPeriodicEnergies_k(self, N=10, ax=None, ay=None, az=None):
        if not ax: ax = self.ax
        if not ay: ay = self.ay
        if not az: az = self.az

        bx = 2 * np.pi / self.ax
        by = 2 * np.pi / self.ay
        bz = 2 * np.pi / self.az

        kx = np.linspace(-bx / 2, bx / 2, N)
        ky = np.linspace(-by / 2, by / 2, N)
        kz = np.linspace(-bz / 2, bz / 2, N)
        X, Y, Z = np.meshgrid(kx, ky, kz)

        def eH(kx, ky, kz):
            H = self.get3DPeriodicH_k(kx * ax, ky * ay, kz * az)
            if H.ndim == 1:
                H = H.reshape((1, -1))
            return np.sort(np.array([evl.real for evl in np.linalg.eigvalsh(H)]))

        return np.array(
            [[[eH(X[i, j, k], Y[i, j, k], Z[i, j, k]) for i in range(X.shape[0])] for j in range(X.shape[1])] for k in
             range(X.shape[2])])

    def get1DPeriodicEnergies(self, N):
        return np.sort(np.linalg.eigvalsh(self.get1DPeriodicH(N)).real)

    def get2DPeriodicEnergies(self, N, M):
        return np.sort(np.linalg.eigvalsh(self.get2DPeriodicH(N, M)).real)

    def get3DPeriodicEnergies(self, N, M, P):
        return np.sort(np.linalg.eigvalsh(self.get3DPeriodicH(N, M, P)).real)


class SimpleMaterial(Material):
    def __init__(self, eps, t, ax0=1, ay0=1, az0=1, ax=1, ay=1, az=1, rc=5):
        super().__init__(ax0, ay0, az0, ax, ay, az)
        self.eps = eps
        self.t = t

    def getOnsiteTerm(self):
        return np.array([self.eps])

    def getInteractionTerm(self, r0, r, index=0):
        return -np.array([self.t])


class SimpleScaledMaterial(Material):
    def __init__(self, eps, t, ax0=1, ay0=1, az0=1, ax=1, ay=1, az=1, rc=5):
        super().__init__(ax0, ay0, az0, ax, ay, az)
        self.eps = eps
        self.t = HarrisonScaledParameter(t, rc)

    def getOnsiteTerm(self):
        return np.array([self.eps])

    def getInteractionTerm(self, r0, r, index=0):
        return -np.array([self.t.getScaled(r0, r)])


class TwoDSupercell(Material):
    def __init__(self, eps1, eps2, t11, t22, ax0=1, ay0=1, az0=1, ax=1, ay=1, az=1, rc=5):
        super().__init__(ax0, ay0, az0, ax, ay, az)
        self.eps = (eps1, eps2)
        self.t = (t11, t22, np.mean([t11, t22]))

    def getOnsiteTerm(self):
        eps1, eps2 = self.eps
        t11, t22, t12 = self.t

        return np.diag([eps1, eps1, eps2, eps2]) - np.diag([t11, t12, t22], -1) - np.diag([t11, t12, t22], 1)

    def getInteractionTerm(self, r0, r, index):
        t11, t22, t12 = self.t

        if index == 0 or index == 2:
            return -np.diag([t11, t11, t22, t22])
        elif index == 1:
            return -np.diag([t12], -3)

    def get2DPeriodicEnergies_k(self, N=10, ax=None, ay=None, az=None):
        return super().get2DPeriodicEnergies_k(N=N, ay=3 * self.ay)

    def get3DPeriodicEnergies_k(self, N=10, ax=None, ay=None, az=None):
        return super().get3DPeriodicEnergies_k(N=N, ay=3 * self.ay)
