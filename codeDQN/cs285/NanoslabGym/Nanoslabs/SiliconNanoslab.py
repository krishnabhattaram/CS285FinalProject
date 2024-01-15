import numpy as np
from scipy.linalg import block_diag

from cs285.NanoslabGym.Nanoslabs.GroupIVSemiconductors import SiliconHarrison
from cs285.NanoslabGym.Nanoslabs.material import Material


class SiliconNanoslab(Material):
    def __init__(self, n_layers, a=5.430, rc=12.5):
        ax0 = ay0 = az0 = 5.430
        super().__init__(ax0, ay0, az0, a, a, a)
        self.si = SiliconHarrison(a, a, a, rc)
        self.n_layers = n_layers

    def getOnsiteTerm(self):
        si_onsite = self.si.getOnsiteTerm()
        si_interaction_z = self.si.getInteractionTerm(self.az0, self.az, index=2)

        dg = block_diag(*(self.n_layers * [si_onsite]))

        od_upper = np.kron(np.diag((self.n_layers - 1) * [1], 1), si_interaction_z)
        od = od_upper + od_upper.T

        return dg + od

    def getInteractionTerm(self, r0, r, index):
        si_interaction = self.si.getInteractionTerm(r0, r, index=index)

        if index == 0 or index == 1:
            return block_diag(*(self.n_layers * [si_interaction]))
        elif index == 2:
            raise RuntimeError('Attempting to access interaction term in z-direction')
