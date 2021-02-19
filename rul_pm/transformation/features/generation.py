
import logging
from typing import Optional

import emd
import numpy as np
import pandas as pd
from rul_pm.transformation.features.extraction import (compute, roll_matrix,
                                                       stats_order)
from rul_pm.transformation.transformerstep import TransformerStep
from sklearn.feature_extraction.text import CountVectorizer
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class SampleNumber(TransformerStep):

    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(index=X.index)
        df['sample_number'] = list(range(X.shape[0]))
        return df


class ExpandingCentering(TransformerStep):

    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
        return self

    def transform(self, X):
        return X - X.expanding().mean()


class ExpandingNormalization(TransformerStep):

    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
        return self

    def transform(self, X):
        return (X - X.expanding().mean()) / (X.expanding().std())


class MeanCentering(TransformerStep):
    def __init__(self):
        super().__init__()
        self.N = 0
        self.sum = None

    def fit(self, X, y=None):
        self.mean = X.mean()

    def partial_fit(self, X, y=None):
        if self.sum is None:
            self.sum = X.sum()
        else:
            self.sum += X.sum()

        self.N += X.shape[0]
        self.mean = self.sum / self.N

    def transform(self, X):
        return X - self.mean


class Accumulate(TransformerStep):

    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.cumsum()


class Diff(TransformerStep):

    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.diff()


class AccumulateEWMAOutOfRange(TransformerStep):
    """
    Compute the EWMA limits and accumulate the number of points
    outsite UCL and LCL

    Parameters
    ----------

    """

    def __init__(self, lambda_=0.5, scale: bool = False, name: Optional[str] = None):
        super().__init__(name)
        self.lambda_ = lambda_
        self.UCL = None
        self.LCL = None
        self.columns = None
        self.scale = scale

    def partial_fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.columns.values
        else:
            self.columns = [c for c in self.columns if c in X.columns]
        if self.LCL is not None:
            self.LCL = self.LCL.loc[self.columns].copy()
            self.UCL = self.UCL.loc[self.columns].copy()
        LCL, UCL = self._compute_limits(X[self.columns].copy())
        self.LCL = (np.minimum(LCL, self.LCL) if self.LCL is not None
                    else LCL)
        self.UCL = (np.maximum(UCL, self.UCL) if self.UCL is not None
                    else UCL)
        return self

    def _compute_limits(self, X):

        mean = np.nanmean(X, axis=0)
        s = np.sqrt(self.lambda_ / (2-self.lambda_)) * \
            np.nanstd(X, axis=0)
        UCL = mean + 3*s
        LCL = mean - 3*s
        return (pd.Series(LCL, index=self.columns),
                pd.Series(UCL, index=self.columns))

    def fit(self, X, y=None):
        self.columns = X.columns
        LCL, UCL = self._compute_limits(X)
        self.LCL = LCL
        self.UCL = UCL
        return self

    def transform(self, X):
        mask = (
            (X[self.columns] < (self.LCL)) |
            (X[self.columns] > (self.UCL))
        )
        return mask.astype('int').cumsum()


class OneHotCategoricalPandas(TransformerStep):
    def __init__(self, feature, name: Optional[str] = None):
        super().__init__(name)
        self.feature = feature
        self.categories = set()
        self.encoder = None

    def partial_fit(self, X, y=None):
        self.categories.update(set(X[self.feature].unique()))
        return self

    def fit(self, X, y=None):
        self.categories.update(set(X[self.feature].unique()))
        return self

    def transform(self, X, y=None):
        categories = sorted(
            list([c for c in self.categories if c is not None]))
        d = pd.Categorical(X[self.feature], categories=categories)

        df = pd.get_dummies(d)
        df.index = X.index
        return df


class SimpleEncodingCategorical(TransformerStep):
    def __init__(self, feature, name: Optional[str] = None):
        super().__init__(name)
        self.feature = feature
        self.categories = set()
        self.encoder = None

    def partial_fit(self, X, y=None):
        self.categories.update(set(X[self.feature].unique()))
        return self

    def fit(self, X, y=None):
        self.categories.update(set(X[self.feature].unique()))
        return self

    def transform(self, X, y=None):
        categories = sorted(
            list([c for c in self.categories if c is not None]))
        d = pd.Categorical(X[self.feature], categories=categories)

        return pd.DataFrame(d.codes, index=X.index)


class LowFrequencies(TransformerStep):
    def __init__(self, window, name: Optional[str] = None):
        super().__init__(name)
        self.window = window

    def fit(self, X, y=None):
        return self

    def _low(self, signal, t):
        a = firwin(self.window+1, cutoff=0.01,
                   window="hann", pass_zero='lowpass')
        return lfilter(a, 1, signal)

    def transform(self, X, y=None):
        cnames = ([f'{c}_low' for c in X.columns])
        new_X = pd.DataFrame(np.zeros((len(X.index), len(cnames)),
                                      dtype=np.float32), columns=cnames, index=X.index)
        for c in X.columns:
            new_X.loc[:, f'{c}_low'] = self._low(X[c], 0)
        return new_X


class HighFrequencies(TransformerStep):
    def __init__(self, window, name: Optional[str] = None):
        super().__init__(name)
        self.window = window

    def fit(self, X, y=None):
        return self

    def _high(self, signal, t):
        a = firwin(self.window+1, cutoff=0.2,
                   window="hann", pass_zero='highpass')
        return lfilter(a, 1, signal)

    def transform(self, X, y=None):
        cnames = [f'{c}_high' for c in X.columns]
        new_X = pd.DataFrame(np.zeros((len(X.index), len(cnames)),
                                      dtype=np.float32), columns=cnames, index=X.index)
        for c in X.columns:
            new_X.loc[:, f'{c}_high'] = self._high(X[c], 0)
        return new_X


class MedianFilter(TransformerStep):
    def __init__(self, window: int, min_periods: int = 15, name: Optional[str] = None):
        super().__init__(name)
        self.window = window
        self.min_periods = min_periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.rolling(self.window, min_periods=self.min_periods).median()


class MeanFilter(TransformerStep):
    def __init__(self, window: int, min_periods: int = 15, name: Optional[str] = None):
        super().__init__(name)
        self.window = window
        self.min_periods = min_periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.rolling(self.window, min_periods=self.min_periods).mean(skip_na=True)


def rolling_kurtosis(s: pd.Series, window, min_periods):
    return s.rolling(window, min_periods=min_periods).kurt(skipna=True)


class RollingStatistics(TransformerStep):
    def __init__(self, window, step=1, min_periods: int = 15, time: bool = True, frequency: bool = True, name: Optional[str] = None):
        super().__init__(name)
        self.window = window
        self.min_periods = min_periods
        self.step = step
        self.fs = 1
        self.time = time
        self.frequency = frequency

    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
        return self

    def transform(self, X):

        columns = []
        stat_columns_dict = {}
        for column in X.columns:
            for stat in stats_order(time=self.time, frequency=self.frequency):
                new_cname = f'{column}_{stat}'
                stat_columns_dict.setdefault(column, []).append(len(columns))
                columns.append(new_cname)

        X_new = pd.DataFrame(
            np.zeros((len(X.index), len(columns))),
            index=X.index,
            columns=columns,
            dtype=np.float32)
        X_values = X.values
        X_new_values = X_new.values
        roll_matrix(X_values, self.window, self.min_periods,
                    X_new_values, time=self.time, frequency=self.frequency)

        return X_new


class ExpandingStatistics(TransformerStep):
    def __init__(self, min_points=2,  to_compute=None, name: Optional[str] = None):
        super().__init__(name)
        self.min_points = min_points
        valid_stats = ['kurtosis', 'skewness', 'max',
                       'min', 'std', 'peak', 'impulse', 'clearance',
                       'rms', 'shape', 'crest']
        if to_compute is None:
            self.to_compute = ['kurtosis', 'skewness', 'max',
                               'min', 'std', 'peak', 'impulse', 'clearance',
                               'rms', 'shape', 'crest']
        else:
            for f in to_compute:
                if f not in valid_stats:
                    raise ValueError(
                        f'Invalid feature to compute {f}. Valids are {valid_stats}')
            self.to_compute = to_compute

    def partial_fit(self, X, y=None):
        return self

    def fit(self, X, y=None):
        return self

    def _kurtosis(self, s: pd.Series):
        return s.expanding(self.min_points).kurtosis(skipna=True)

    def _skewness(self, s: pd.Series):
        return s.expanding(self.min_points).skewness(skipna=True)

    def _max(self, s: pd.Series):
        return s.expanding(self.min_points).max(skipna=True)

    def _min(self, s: pd.Series):
        return s.expanding(self.min_points).min(skipna=True)

    def _std(self, s: pd.Series):
        return s.expanding(self.min_points).std(skipna=True)

    def _peak(self, s: pd.Series):
        return (s.expanding(self.min_points).max(skipna=True) -
                s.expanding(self.min_points).min(skipna=True))

    def _impulse(self, s: pd.Series):
        return self._peak(s) / s.abs().expanding(self.min_points).mean()

    def _clearance(self, s: pd.Series):
        return self._peak(s) / s.abs().pow(1./2).expanding(self.min_points).mean().pow(2)

    def _rms(self, s: pd.Series):
        return (s.pow(2)
                 .expanding(self.min_points)
                 .mean(skipna=True)
                 .pow(1/2.))

    def _shape(self, s: pd.Series):
        return self._rms(s) / s.abs().expanding(self.min_points).mean(skipna=True)

    def _crest(self, s: pd.Series):
        return self._peak(s) / self._rms(s)

    def transform(self, X):

        X_new = pd.DataFrame(index=X.index)

        for c in X.columns:
            for stats in self.to_compute:
                X_new[f'{c}_{stats}'] = getattr(self, f'_{stats}')(X[c])

        return X_new


class Difference(TransformerStep):
    def __init__(self, feature_set1: list, feature_set2: list, name: Optional[str] = None):
        super().__init__(name)
        if len(feature_set1) != len(feature_set2):
            raise ValueError(
                'Feature set 1 and feature set 2 must have the same length')
        self.feature_set1 = feature_set1
        self.feature_set2 = feature_set2

    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
        return self

    def transform(self, X):
        new_X = X[self.feature_set1].copy()
        new_X = new_X - X[self.feature_set2].values
        return new_X


class EMD(TransformerStep):
    def __init__(self,  name: Optional[str] = 'EMD'):
        super().__init__(name)

    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
        return self

    def transform(self, X):
        new_X = pd.DataFrame(index=X.index)
        n = 4
        for c in X.columns:
            try:
                imf = emd.sift.sift(X[c].values, max_imfs=n)
                for j in range(n):
                    if j < imf.shape[1]:
                        new_X[f'{c}_{j}'] = imf[:, j]
                    else:
                        new_X[f'{c}_{j}'] = np.nan
            except Exception as e:
                print(e)
                for j in range(n):
                    new_X[f'{c}_{j}'] = np.nan

        return new_X


class EMDFilter(TransformerStep):
    def __init__(self,  name: Optional[str] = 'EMD'):
        super().__init__(name)

    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
        return self

    def transform(self, X):
        new_X = pd.DataFrame(index=X.index)
        n = 8
        for c in X.columns:
            try:
                imf = emd.sift.sift(X[c].values, max_imfs=n)
                new_X[c] = np.sum(imf[:, 1:], axis=1)
            except Exception as e:
                new_X[c] = X[c]

        return new_X


class ChangesCounter(TransformerStep):
    def __init__(self, feature_name: str,  name: Optional[str] = 'ChangesCounter'):
        super().__init__(name)
        self.feature_name = feature_name

    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
        return self

    def transform(self, X):
        return ((X[[self.feature_name]] != X[[self.feature_name]]
                 .shift(axis=0))
                .cumsum())
