
from math import sqrt

import numpy as np
import numpy.fft as fft
from numba import float64, jit, objmode, prange
from numba.experimental import jitclass
from numba.typed import List
from numba.types import ListType, unicode_type
from rul_pm.transformation.features.extraction_frequency import \
    compute_frequency_features

spec = [
    ('n', float64),
    ('M1', float64),
    ('M2', float64),
    ('M3', float64),
    ('M4', float64),

]


@jitclass(spec)
class RunningStats:
    """
    https://www.johndcook.com/blog/skewness_kurtosis/
    """

    def __init__(self):
        self.n = 0
        self.M1 = 0
        self.M2 = 0
        self.M3 = 0
        self.M4 = 0.0

    def add(self, x):

        n1 = self.n
        self.n += 1
        delta = x - self.M1
        delta_n = delta / self.n
        delta_n2 = delta_n * delta_n
        term1 = delta * delta_n * n1
        self.M1 += delta_n
        self.M4 += (term1 * delta_n2 * (self.n*self.n - 3*self.n + 3) + 6 *
                    delta_n2 * self.M2 - 4 * delta_n * self.M3)
        self.M3 += term1 * delta_n * (self.n - 2) - 3 * delta_n * self.M2
        self.M2 += term1

    @property
    def mean(self):
        return self.M1

    @property
    def var(self):
        return self.M2/(self.n-1.0)

    @property
    def std(self):
        return sqrt(self.var)

    @property
    def skewness(self):
        return sqrt(self.n) * self.M3 / (self.M2**1.5)

    @property
    def kurtosis(self):
        return (self.n)*self.M4 / (self.M2*self.M2) - 3.0


@jit(nopython=True, error_model='numpy')
def stats_order(time=True, frequency=True):
    features = []

    if time:
        features.extend(['max', 'min', 'peak', 'std', 'impulse', 'rms',
                                'crest', 'shape', 'clearance', 'kurtosis', 'skewness'])
    if frequency:
        features.extend(['fft_centroid', 'fft_variance',
                         'fft_skew', 'fft_kurtosis',
                         'ps_centroid', 'ps_variance',
                         'ps_skew', 'ps_kurtosis'])
    return features


@jit(nopython=True, error_model='numpy')
def running_stat_init():
    return [0.0, 0.0, 0.0, 0.0, 0.0]


@jit(nopython=True, error_model='numpy')
def add(stats, x):
    n1 = stats[0]
    stats[0] += 1
    delta = x - stats[1]
    delta_n = delta / stats[0]
    delta_n2 = delta_n * delta_n
    term1 = delta * delta_n * n1
    stats[1] += delta_n
    stats[4] += (term1 * delta_n2 * (stats[0]*stats[0] - 3*stats[0] + 3) + 6 *
                 delta_n2 * stats[2] - 4 * delta_n * stats[3])
    stats[3] += term1 * delta_n * (stats[0] - 2) - 3 * delta_n * stats[2]
    stats[2] += term1


@jit(nopython=True, error_model='numpy')
def mean(self):
    return self[0]


@jit(nopython=True, error_model='numpy')
def var(self):
    return self[2]/(self[0]-1.0)


@jit(nopython=True, error_model='numpy')
def std(self):
    return sqrt(var(self))


@jit(nopython=True, error_model='numpy')
def skewness(self):
    return sqrt(self[0]) * self[3] / (self[2]**1.5)


@jit(nopython=True, error_model='numpy')
def kurtosis(self):
    return (self[0])*self[4] / (self[2]*self[2]) - 3.0


@jit(nopython=True, error_model='numpy')
def compute_time_features(window):
    """
        'max': val_max,
        'min': val_min,
        'peak': peak,
        'std': std,
        'impulse': peak / mean_abs,
        'rms': rms,
        'crest': peak / rms,
        'shape': rms / mean_abs,
        'clearance': peak / (maen_abs_sqrt)**2,
        'kurtosis': stats.kurtosis,
        'skewness': stats.skewness
    """

    val_max = -np.inf
    val_min = np.inf
    sum_abs = 0.0
    sum_vals = 0.0
    sum_squared = 0.0
    sum_abs_sqrt = 0.0
    n = 0
    stats = running_stat_init()
    for i in range(len(window)):
        if np.isnan(window[i]):
            continue
        val_max = max(val_max, window[i])
        val_min = min(val_min, window[i])
        sum_vals += window[i]
        sum_abs += abs(window[i])
        sum_squared += window[i]**2
        sum_abs_sqrt += sqrt(abs(window[i]))
        add(stats, window[i])
        n += 1

    mean_val = mean(stats)
    mean_abs = sum_abs / n
    mean_squared = sum_squared / n
    maen_abs_sqrt = sum_abs_sqrt / n

    std_computed = std(stats)

    rms = sqrt(mean_squared)
    peak = abs(val_max - val_min)

    data = [
        val_max,
        val_min,
        peak,
        std_computed,
        peak / (mean_abs),
        rms,
        peak / (rms),
        rms / (mean_abs),
        peak / ((maen_abs_sqrt)**2),
        kurtosis(stats),
        skewness(stats)
    ]
    return data


@jit(nopython=True, error_model='numpy')
def compute(window, time=True, frequency=True):
    features = []
    if time:
        features.extend(compute_time_features(window))
    if frequency:
        freq_data = compute_frequency_features(window)
        features.extend(freq_data)
    return features


@jit(nopython=True, error_model='numpy')
def roll_matrix(input, window: int, min_samples: int, output, time=True, frequency=True):

    nrows, ncols = input.shape

    stats = stats_order(time=time, frequency=frequency)
    number_of_columns = len(stats)

    for i in range(nrows):
        if i < min_samples:
            output[i, :] = np.nan
            continue

        for c in range(ncols):
            result = compute(input[max(i-window, 0):i, c],
                             time=time, frequency=frequency)
            for j, k in zip(range(c*number_of_columns, (c+1)*number_of_columns), range(len(result))):
                output[i, j] = result[k]
