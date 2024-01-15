import numpy as np

from codeMBPO.cs285.envs.NanoslabGym.Nanoslabs.material import Material
from codeMBPO.cs285.envs.NanoslabGym.Nanoslabs.parameters import UnscaledParameter, HarrisonScaledParameter, \
    NRLScaledParameterOnsite, NRLScaledParameterInteraction


class GroupIVSemiconductor(Material):
    def __init__(self, ax0, ay0, az0, ax, ay, az, onsp_s, onsp_p,
                 hp_ss_sigma, eta_ss_sigma, hp_sp_sigma, eta_sp_sigma,
                 hp_ppsigma, eta_pp_sigma, hp_pppi, eta_pp_pi, rc):
        super().__init__(ax0, ay0, az0, ax, ay, az)

        # Onsite Parameters
        self.onsp_s = UnscaledParameter(onsp_s)  # NRLScaledParameter(onsp_s, rc)
        self.onsp_p = UnscaledParameter(onsp_p)  # NRLScaledParameter(onsp_p, rc)

        # Hopping Parameters
        self.hp_ss_sigma = HarrisonScaledParameter(hp_ss_sigma, rc, eta=eta_ss_sigma)
        self.hp_sp_sigma = HarrisonScaledParameter(hp_sp_sigma, rc, eta=eta_sp_sigma)
        self.hp_ppsigma = HarrisonScaledParameter(hp_ppsigma, rc, eta=eta_pp_sigma)
        self.hp_pppi = HarrisonScaledParameter(hp_pppi, rc, eta=eta_pp_pi)

    def getOnsiteTerm(self):
        onsp_ss = self.onsp_s.getScaled()
        onsp_ps = self.onsp_p.getScaled()

        return np.diag([onsp_ss, onsp_ss, onsp_ps, onsp_ps, onsp_ps, onsp_ps, onsp_ps, onsp_ps])

    def getInteractionTerm(self, r0, r, index=0):
        sssig = self.hp_ss_sigma.getScaled(r0, r)
        spsig = self.hp_sp_sigma.getScaled(r0, r)
        ppsig = self.hp_ppsigma.getScaled(r0, r)
        pppi = self.hp_pppi.getScaled(r0, r)

        # Bond along the axis is sigma, other two are pi
        ppx = ppsig if index == 0 else pppi
        ppy = ppsig if index == 1 else pppi
        ppz = ppsig if index == 2 else pppi

        dg = np.diag([sssig, sssig, ppx, ppx, ppy, ppy, ppz, ppz])
        od2 = np.diag([spsig, spsig, 0, 0, 0, 0], 2) + np.diag([spsig, spsig, 0, 0, 0, 0], -2)
        od4 = np.diag([spsig, spsig, 0, 0], 4) + np.diag([spsig, spsig, 0, 0], -4)
        od6 = np.diag([spsig, spsig], 6) + np.diag([spsig, spsig], -6)

        return dg + od2 + od4 + od6


class Silicon3DNRL(GroupIVSemiconductor):
    def __init__(self, ax=5.430, ay=5.430, az=5.430, rc=12.5):
        # Geometrical structure
        ax0 = ay0 = az0 = 5.430

        # Onsite Parameters
        ons_lambda = 1.1035662
        onsp_s = {"lambda": ons_lambda, "alpha": -0.0532, "beta": -0.907642, "gamma": -8.30849, "chi": 56.56613}
        onsp_p = {"lambda": ons_lambda, "alpha": 0.357859, "beta": 0.303647, "gamma": 7.092229, "chi": -77.47855}

        neighbors = [ax, ax, ay, ay, az, az]

        # Onsite Parameters
        self.onsp_s = NRLScaledParameterOnsite(onsp_s, rc, neighbors)
        self.onsp_p = NRLScaledParameterOnsite(onsp_p, rc, neighbors)

        # Hopping parameters
        hp_sssigma = {"a": 219.5608, "b": -16.2132, "c": -15.5048, "d": 1.264399}
        hp_spsigma = {"a": 10.127, "b": -4.40368, "c": 0.22667, "d": 0.922671}
        hp_ppsigma = {"a": -22.959, "b": 1.72, "c": 1.41913, "d": 1.0313}
        hp_pppi = {"a": 10.2654, "b": 4.6718, "c": -2.2161, "d": 1.1113}

        # Hopping Parameters
        self.hp_ss_sigma = NRLScaledParameterInteraction(hp_sssigma, rc)
        self.hp_sp_sigma = NRLScaledParameterInteraction(hp_spsigma, rc)
        self.hp_ppsigma = NRLScaledParameterInteraction(hp_ppsigma, rc)
        self.hp_pppi = NRLScaledParameterInteraction(hp_pppi, rc)

        Material.__init__(self, ax0, ay0, az0, ax, ay, az)


class SiliconHarrison(GroupIVSemiconductor):
    def __init__(self, ax=5.430, ay=5.430, az=5.430, rc=12.5):
        # Geometrical structure
        ax0 = ay0 = az0 = 5.430

        onsp_s = -2.0196
        onsp_p = 4.5448

        # Hopping Parameters
        hp_ss_sigma = -1.9413
        eta_ss_sigma = 3.672

        hp_sp_sigma = 2.7836
        eta_sp_sigma = 2.488

        hp_ppsigma = 4.1068
        eta_pp_sigma = 2.187

        hp_pppi = -1.5934
        eta_pp_pi = 3.711

        super().__init__(ax0, ay0, az0, ax, ay, az, onsp_s, onsp_p,
                         hp_ss_sigma, eta_ss_sigma, hp_sp_sigma, eta_sp_sigma,
                         hp_ppsigma, eta_pp_sigma, hp_pppi, eta_pp_pi, rc)


class GermaniumHarrison(GroupIVSemiconductor):
    def __init__(self, ax=5.6563, ay=5.6563, az=5.6563, rc=13):
        # Geometrical structure
        ax0 = ay0 = az0 = 5.6563

        # Onsite Parameters
        onsp_s = -3.2967
        onsp_p = 4.6560

        # Hopping Parameters
        hp_ss_sigma = -1.5003
        eta_ss_sigma = 3.631

        hp_sp_sigma = 2.7986
        eta_sp_sigma = 3.713

        hp_ppsigma = 4.2541
        eta_pp_sigma = 2.030

        hp_pppi = -1.6510
        eta_pp_pi = 4.025

        super().__init__(ax0, ay0, az0, ax, ay, az, onsp_s, onsp_p,
                         hp_ss_sigma, eta_ss_sigma, hp_sp_sigma, eta_sp_sigma,
                         hp_ppsigma, eta_pp_sigma, hp_pppi, eta_pp_pi, rc)