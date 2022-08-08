from abc import abstractmethod

from signal import signal
from typing import Any, Callable, Union

import numpy as np



class AbstractSampleWeights:
    """The base class for the sample weight provider
    """
    def __call__(self, y, i: int, metadata):
        raise NotImplementedError


class NotWeighted(AbstractSampleWeights):
    """Simplest sample weight provvider

    Provide 1 as a sample weight for every sample
    """
    def __call__(self, y, i: int, metadata):
        return 1

"""The Sample Weight type is a callable with the following signature

    fun(y, i:int, metadata)

Given the target and the sample index `i` it returns the sample weight for sample `i`
"""
SampleWeight = Union[AbstractSampleWeights, Callable[[np.ndarray, int, Any], float]]


class RULInverseWeighted(AbstractSampleWeights):
    r"""Weight each sample inverse to the RUL

    ```math
    w_i = \frac{1}{RUL_{i} + 1} 
    ```
    """
    def __call__(self, y, i: int, metadata):
        return 1 / (y[i, 0] + 1)


class InverseToLengthWeighted(AbstractSampleWeights):
    """Weights samples according to the duration of the run-to-failure cycle they bolng

    All points in the run-to-cycle are weighted equally inverse to the cycle duration
    """
    def __call__(self, y, i: int, metadata):
        return 1 / y[0]


class ExponentialDecay(AbstractSampleWeights):
    def __init__(self, *, near_0_at:float):
        super().__init__()
        self.alpha = -(near_0_at)**2/np.log(0.000001)

    def __call__(self, y, i: int, metadata):
        return (1 + np.exp(-(y[i, 0]**2) / self.alpha))**2
        
