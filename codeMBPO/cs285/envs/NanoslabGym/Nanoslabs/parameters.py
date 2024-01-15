from abc import ABC, abstractmethod
import numpy as np


class ScaledParameter(ABC):
    def cutoff(r, rc):
        l = 0.5
        return (1 - np.heaviside(r - rc, 1)) / (1 + np.exp((r - rc) / l))

    @abstractmethod
    def getScaled(self):
        pass


class UnscaledParameter(ScaledParameter):
    def __init__(self, value):
        self.value = value

    def getScaled(self):
        return self.value


class HarrisonScaledParameter(ScaledParameter):
    def __init__(self, value, rc, eta=2):
        self.value = value
        self.rc = rc
        self.eta = eta

    def getScaled(self, r0, r):
        return self.value * np.power(r0 / r, self.eta) * ScaledParameter.cutoff(r, self.rc)


class NRLScaledParameterOnsite(ScaledParameter):
    def __init__(self, ons, rc, neighbors):
        self.ons = ons
        self.rc = rc
        self.neighbors = neighbors

    def getScaled(self):
        rho = np.sum(
            [np.exp(-self.ons['lambda'] ** 2 * r) * ScaledParameter.cutoff(r, self.rc) for r in self.neighbors])
        return self.ons["alpha"] + self.ons["beta"] * rho ** (2 / 3) + self.ons["gamma"] * rho ** (4 / 3) + self.ons[
            "chi"] * rho ** 2


class NRLScaledParameterInteraction(ScaledParameter):
    def __init__(self, hp, rc):
        self.hp = hp
        self.rc = rc

    def getScaled(self, r0, r):
        return (self.hp["a"] + self.hp["b"] * r + self.hp["c"] * r ** 2) * np.exp(
            -self.hp["d"] ** 2 * r) * ScaledParameter.cutoff(r, self.rc)
