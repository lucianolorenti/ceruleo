from temporis.iterators.iterators import AbstractSampleWeights
import numpy as np

class RULInverseWeighted(AbstractSampleWeights):
    def __init__(self, c:float):
        super().__init__()
        self.c = c
    
    def __call__(self, y, i: int, metadata):
        return self.c / (y[i, 0] + 1)


class InverseToLengthWeighted(AbstractSampleWeights):
    def __call__(self, y, i: int, metadata):
        return 1 / y[0]


class ExponentialDecay(AbstractSampleWeights):
    def __init__(self, *, near_0_at:float):
        super().__init__()
        self.alpha = -(near_0_at)**2/np.log(0.000001)

    def __call__(self, y, i: int, metadata):
        return (1 + np.exp(-(y[i, 0]**2) / self.alpha))**2