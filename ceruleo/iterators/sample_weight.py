
from typing import Any, Callable, Union

import numpy as np
import pandas as pd


class AbstractSampleWeights:
    """
    The base class for the sample weight provider
    """

    def __call__(self, y: Union[np.ndarray, pd.DataFrame], i: int, metadata):
        raise NotImplementedError



def get_value(y: Union[np.ndarray, pd.DataFrame], i:int) -> float:
    if isinstance(y, np.ndarray):
        if len(y.shape) > 1:
            return y[i, 0]
        else:
            return  y[i]
    elif isinstance(y, pd.DataFrame):
        return y.iloc[i, 0] 
    elif isinstance(y, pd.Series):
        return y.iloc[i]
    else:
        raise ValueError(f"Unsupported type {type(y)}")
   


class NotWeighted(AbstractSampleWeights):
    """
    Simplest sample weight provider

    Provide 1 as a sample weight for every sample
    """

    def __call__(self, y: Union[np.ndarray, pd.DataFrame], i: int, metadata):
        return 1


"""
The Sample Weight type is a callable with the following signature

    fun(y, i:int, metadata)

Given the target and the sample index `i` it returns the sample weight for sample `i`
"""
SampleWeight = Union[AbstractSampleWeights, Callable[[np.ndarray, int, Any], float]]


class RULInverseWeighted(AbstractSampleWeights):
    """
    Weight each sample by the inverse of the RUL
    """

    def __call__(self, y : Union[np.ndarray, pd.DataFrame], i: int, metadata):
        return 1 / (get_value(y, i) + 1)

        


class InverseToLengthWeighted(AbstractSampleWeights):
    """
    Weights samples according to the duration of the run-to-failure cycle they belong to.

    All points in the run-to-cycle are weighted equally inverse to the cycle duration

    """

    def __call__(self, y:Union[np.ndarray, pd.DataFrame], i: int, metadata):
        return 1 / get_value(y, 0)


class ExponentialDecay(AbstractSampleWeights):
    """
    Weight samples with an exponential decay function based on the RUL.

    """

    def __init__(self, *, near_0_at: float):
        
        super().__init__()
        self.alpha = -((near_0_at) ** 2) / np.log(0.000001)

    def __call__(self, y:Union[np.ndarray, pd.DataFrame], i: int, metadata):
        return ( np.exp(-(get_value(y,i) ** 2) / self.alpha))
