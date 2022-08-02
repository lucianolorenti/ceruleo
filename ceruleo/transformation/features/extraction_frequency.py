
from math import sqrt

import numba
import numpy as np
import numpy.fft as fft
from numba import float32, jit, objmode
from numba.experimental import jitclass


@jit(nopython=True, error_model='numpy')
def get_moment(y, moment) -> np.float32:
    """
    Returns the (non centered) moment of the distribution y:
    E[y**moment] = \\sum_i[index(y_i)^moment * y_i] / \\sum_i[y_i]
    :param y: the discrete distribution from which one wants to calculate the moment
    :type y: pandas.Series or np.array
    :param moment: the moment one wants to calcalate (choose 1,2,3, ... )
    :type moment: int
    :return: the moment requested
    :return type: float
    """
    return y.dot(np.arange(len(y), dtype=np.float64)**moment) / y.sum()


@jit(nopython=True, error_model='numpy')
def get_centroid(y) -> np.float32:
    """
    :param y: the discrete distribution from which one wants to calculate the centroid
    :type y: pandas.Series or np.array
    :return: the centroid of distribution y (aka distribution mean, first moment)
    :return type: float
    """
    return get_moment(y, 1)


@jit(nopython=True, error_model='numpy')
def get_variance(y) -> np.float32:
    """
    :param y: the discrete distribution from which one wants to calculate the variance
    :type y: pandas.Series or np.array
    :return: the variance of distribution y
    :return type: float
    """
    return get_moment(y, 2) - get_centroid(y) ** 2


@jit(nopython=True, error_model='numpy')
def get_skew(y) -> np.float32:
    """
    Calculates the skew as the third standardized moment.
    Ref: https://en.wikipedia.org/wiki/Skewness#Definition
    :param y: the discrete distribution from which one wants to calculate the skew
    :type y: pandas.Series or np.array
    :return: the skew of distribution y
    :return type: float
    """

    variance = get_variance(y)
    # In the limit of a dirac delta, skew should be 0 and variance 0.  However, in the discrete limit,
    # the skew blows up as variance --> 0, hence return nan when variance is smaller than a resolution of 0.5:
    if variance < 0.5:
        return np.nan
    else:
        return (
            get_moment(y, 3) - 3 * get_centroid(y) *
            variance - get_centroid(y)**3
        ) / get_variance(y)**(1.5)


@jit(nopython=True, error_model='numpy')
def get_kurtosis(y) -> np.float32:
    """
    Calculates the kurtosis as the fourth standardized moment.
    Ref: https://en.wikipedia.org/wiki/Kurtosis#Pearson_moments
    :param y: the discrete distribution from which one wants to calculate the kurtosis
    :type y: pandas.Series or np.array
    :return: the kurtosis of distribution y
    :return type: float
    """

    variance = get_variance(y)
    # In the limit of a dirac delta, kurtosis should be 3 and variance 0.  However, in the discrete limit,
    # the kurtosis blows up as variance --> 0, hence return nan when variance is smaller than a resolution of 0.5:
    if variance < 0.5:
        return np.nan
    else:
        return (
            get_moment(y, 4) - 4 * get_centroid(y) * get_moment(y, 3)
            + 6 * get_moment(y, 2) * get_centroid(y)**2 -
            3 * get_centroid(y)
        ) / get_variance(y)**2


@jit(debug=True, nopython=True, error_model='numpy')
def compute_frequency_features(x):
    """
    Returns the spectral centroid (mean), variance, skew, and kurtosis of the absolute fourier transform spectrum.
    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"aggtype": s} where s str and in ["centroid", "variance",
        "skew", "kurtosis"]
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """

    with objmode(fft_abs='float64[:]'):
        fft_abs = np.abs(fft.rfft(x))

    ps = fft_abs**2

    data = np.zeros(8, dtype=np.float32)
    data[0] = get_centroid(fft_abs)
    data[1] = get_variance(fft_abs)
    data[2] = get_skew(fft_abs)
    data[3] = get_kurtosis(fft_abs)
    data[4] = get_centroid(ps)
    data[5] = get_variance(ps)
    data[6] = get_skew(ps)
    data[7] = get_kurtosis(ps)
    return data
