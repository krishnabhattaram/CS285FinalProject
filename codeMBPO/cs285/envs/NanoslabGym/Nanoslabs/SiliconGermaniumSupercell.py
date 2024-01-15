import numpy as np
from scipy.linalg import block_diag

from codeMBPO.cs285.envs.NanoslabGym.Nanoslabs.GroupIVSemiconductors import SiliconHarrison, GermaniumHarrison
from codeMBPO.cs285.envs.NanoslabGym.Nanoslabs.material import Material


class SiliconGermaniumSupercell(Material):
    def __init__(self, ax=5.54315, ay=5.54315, az=5.54315, rc=12.75):
        ax0 = ay0 = az0 = 5.54315
        super().__init__(ax0, ay0, az0, ax, ay, az)
        self.si = SiliconHarrison(ax, ay, az, rc)
        self.ge = GermaniumHarrison(ax, ay, az, rc)

    def getOnsiteTerm(self):
        si_onsite = self.si.getOnsiteTerm()
        ge_onsite = self.ge.getOnsiteTerm()

        si_interaction_y = self.si.getInteractionTerm(self.ay0, self.ay, index=1)
        ge_interaction_y = self.ge.getInteractionTerm(self.ay0, self.ay, index=1)
        ge_si_avg_y = (si_interaction_y + ge_interaction_y) / 2

        dg = block_diag(*[ge_onsite, ge_onsite, si_onsite, si_onsite])
        od_upper = np.kron(np.diag([1, 0, 0], 1), ge_interaction_y) \
                   + np.kron(np.diag([0, 1, 0], 1), ge_si_avg_y) \
                   + np.kron(np.diag([0, 0, 1], 1), si_interaction_y)
        od = od_upper + od_upper.T

        return dg + od

    def getInteractionTerm(self, r0, r, index):
        si_interaction = self.si.getInteractionTerm(r0, r, index=index)
        ge_interaction = self.ge.getInteractionTerm(r0, r, index=index)
        ge_si_avg = (si_interaction + ge_interaction) / 2

        if index == 0 or index == 2:
            return block_diag(*[ge_interaction, ge_interaction, si_interaction, si_interaction])
        elif index == 1:
            return np.kron(np.diag([1], -3), ge_si_avg)
