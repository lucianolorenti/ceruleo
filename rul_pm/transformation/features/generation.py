import itertools
import logging
from rul_pm.transformation.features.hurst import hurst_exponent
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


class TimeToPreviousBinaryValue(TransformerStep):
    """Return a column with increasing number
    """
    def time_to_previous_event(self, X: pd.DataFrame, c: str):
        def min_idex(group):
            if group.iloc[0, 0] == 0:
                return np.nan
            else:
                return np.min(group.index)

        X_c_cumsum = X[[c]].cumsum()
        min_index = X_c_cumsum.groupby(c).apply(min_idex)
        X_merged = X_c_cumsum.merge(pd.DataFrame(min_index, columns=['start']),
                                    left_on=c,
                                    right_index=True)
        return X_merged.index - X_merged['start']

    def transform(self, X: pd.DataFrame):
        new_X = pd.DataFrame(index=X.index)
        for c in X.columns:
            new_X[f'ttp_{c}'] = self.time_to_previous_event(X, c)
        return new_X


class Sum(TransformerStep):
    """
    Sum each column
    """
    def __init__(self, column_name: str, name: Optional[str] = None):
        super().__init__(name)
        self.column_name = column_name

    def transform(self, X: pd.DataFrame):
        return pd.DataFrame(X.sum(axis=1), columns=[self.column_name])


class Scale(TransformerStep):
    """
    Scale the dataframe
    """
    def __init__(self, scale_factor: float, name: Optional[str] = None):
        super().__init__(name)
        self.scale_factor = scale_factor

    def transform(self, X: pd.DataFrame):
        return X * self.scale_factor


class SampleNumber(TransformerStep):
    """Return a column with increasing number
    """
    def transform(self, X):
        df = pd.DataFrame(index=X.index)
        df['sample_number'] = list(range(X.shape[0]))
        return df


class ExpandingCentering(TransformerStep):
    def transform(self, X):
        return X - X.expanding().mean()


class ExpandingNormalization(TransformerStep):
    def transform(self, X):
        return (X - X.expanding().mean()) / (X.expanding().std())


class Accumulate(TransformerStep):
    def transform(self, X):
        return X.cumsum()


class Diff(TransformerStep):
    def transform(self, X):
        return X.diff()





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
        categories = sorted(list([c for c in self.categories
                                  if c is not None]))
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
        categories = sorted(list([c for c in self.categories
                                  if c is not None]))
        d = pd.Categorical(X[self.feature], categories=categories)

        return pd.DataFrame(d.codes, index=X.index)


class LowFrequencies(TransformerStep):
    def __init__(self, window, name: Optional[str] = None):
        super().__init__(name)
        self.window = window

    def fit(self, X, y=None):
        return self

    def _low(self, signal, t):
        a = firwin(self.window + 1,
                   cutoff=0.01,
                   window="hann",
                   pass_zero='lowpass')
        return lfilter(a, 1, signal)

    def transform(self, X, y=None):
        cnames = ([f'{c}_low' for c in X.columns])
        new_X = pd.DataFrame(np.zeros((len(X.index), len(cnames)),
                                      dtype=np.float32),
                             columns=cnames,
                             index=X.index)
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
        a = firwin(self.window + 1,
                   cutoff=0.2,
                   window="hann",
                   pass_zero='highpass')
        return lfilter(a, 1, signal)

    def transform(self, X, y=None):
        cnames = [f'{c}_high' for c in X.columns]
        new_X = pd.DataFrame(np.zeros((len(X.index), len(cnames)),
                                      dtype=np.float32),
                             columns=cnames,
                             index=X.index)
        for c in X.columns:
            new_X.loc[:, f'{c}_high'] = self._high(X[c], 0)
        return new_X


class MedianFilter(TransformerStep):
    def __init__(self,
                 window: int,
                 min_periods: int = 15,
                 name: Optional[str] = None):
        super().__init__(name)
        self.window = window
        self.min_periods = min_periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.rolling(self.window, min_periods=self.min_periods).median()





def rolling_kurtosis(s: pd.Series, window, min_periods):
    return s.rolling(window, min_periods=min_periods).kurt(skipna=True)


class RollingStatistics(TransformerStep):
    def __init__(self,
                 window,
                 step=1,
                 min_periods: int = 15,
                 time: bool = True,
                 frequency: bool = True,
                 select_features: Optional[list] = None,
                 name: Optional[str] = None):
        super().__init__(name)
        self.window = window
        self.min_periods = min_periods
        self.step = step
        self.fs = 1
        self.time = time
        self.frequency = frequency
        self.select_features = select_features

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

        X_new = pd.DataFrame(np.zeros((len(X.index), len(columns))),
                             index=X.index,
                             columns=columns,
                             dtype=np.float32)
        X_values = X.values
        X_new_values = X_new.values
        roll_matrix(X_values,
                    self.window,
                    self.min_periods,
                    X_new_values,
                    time=self.time,
                    frequency=self.frequency)
        if self.select_features:
            selected_columns = []
            for c in X_new.columns:
                for f in self.select_features:
                    if f in c:
                        selected_columns.append(c)
                        break


            X_new = X_new.loc[:, selected_columns]

        return X_new



class RollingStatistics1(TransformerStep):
    def __init__(self,
                 window:int=15,
                 min_points=2,
                 to_compute=None,
                 name: Optional[str] = None):
        super().__init__(name)
        self.window = window
        self.min_points = min_points
        valid_stats = [
            'kurtosis', 'skewness', 'max', 'min', 'std', 'peak', 'impulse',
            'clearance', 'rms', 'shape', 'crest'
        ]
        if to_compute is None:
            self.to_compute = [
                'kurtosis', 'skewness', 'max', 'min', 'std', 'peak', 'impulse',
                'clearance', 'rms', 'shape', 'crest'
            ]
        else:
            for f in to_compute:
                if f not in valid_stats:
                    raise ValueError(
                        f'Invalid feature to compute {f}. Valids are {valid_stats}'
                    )
            self.to_compute = to_compute

    def partial_fit(self, X, y=None):
        return self

    def fit(self, X, y=None):
        return self

    def _kurtosis(self, s: pd.Series):
        return s.rolling(self.window, self.min_points).kurt(skipna=True)

    def _skewness(self, s: pd.Series):
        return s.rolling(self.window,self.min_points).skew(skipna=True)

    def _max(self, s: pd.Series):
        return s.rolling(self.window,self.min_points).max(skipna=True)

    def _min(self, s: pd.Series):
        return s.rolling(self.window,self.min_points).min(skipna=True)

    def _std(self, s: pd.Series):
        return s.rolling(self.window,self.min_points).std(skipna=True)

    def _peak(self, s: pd.Series):
        return (s.rolling(self.window,self.min_points).max(skipna=True) -
                s.rolling(self.window,self.min_points).min(skipna=True))

    def _impulse(self, s: pd.Series):
        return self._peak(s) / s.abs().rolling(self.window,self.min_points).mean()

    def _clearance(self, s: pd.Series):
        return self._peak(s) / s.abs().pow(1. / 2).rolling(self.window,
            self.min_points).mean().pow(2)

    def _rms(self, s: pd.Series):
        return (s.pow(2).rolling(self.window,self.min_points).mean(skipna=True).pow(1 /
                                                                          2.))

    def _shape(self, s: pd.Series):
        return self._rms(s) / s.abs().rolling(self.window,
            self.min_points).mean(skipna=True)

    def _crest(self, s: pd.Series):
        return self._peak(s) / self._rms(s)

    def transform(self, X):
        X_new = pd.DataFrame(index=X.index)
        for c in X.columns:
            for stats in self.to_compute:
                X_new[f'{c}_{stats}'] = getattr(self, f'_{stats}')(X[c])

        return X_new


class ExpandingStatistics(TransformerStep):
    def __init__(self,
                 min_points=2,
                 to_compute=None,
                 name: Optional[str] = None):
        super().__init__(name)
        self.min_points = min_points
        valid_stats = [
            'kurtosis', 'skewness', 'max', 'min', 'std', 'peak', 'impulse',
            'clearance', 'rms', 'shape', 'crest', 'hurst'
        ]
        if to_compute is None:
            self.to_compute = [
                'kurtosis', 'skewness', 'max', 'min', 'std', 'peak', 'impulse',
                'clearance', 'rms', 'shape', 'crest', 'hurst'
            ]
        else:
            for f in to_compute:
                if f not in valid_stats:
                    raise ValueError(
                        f'Invalid feature to compute {f}. Valids are {valid_stats}'
                    )
            self.to_compute = to_compute

    def partial_fit(self, X, y=None):
        return self

    def fit(self, X, y=None):
        return self

    def _kurtosis(self, s: pd.Series):
        return s.expanding(self.min_points).kurt(skipna=True)

    def _skewness(self, s: pd.Series):
        return s.expanding(self.min_points).skew(skipna=True)

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
        return self._peak(s) / s.abs().pow(1. / 2).expanding(
            self.min_points).mean().pow(2)

    def _hurst(self, s: pd.Series):
        return s.expanding(min_periods=max(self.min_points, 50)).apply(lambda s: hurst_exponent(s, method='RS'))

    def _rms(self, s: pd.Series):
        return (s.pow(2).expanding(self.min_points).mean(skipna=True).pow(1 /
                                                                          2.))

    def _shape(self, s: pd.Series):
        return self._rms(s) / s.abs().expanding(
            self.min_points).mean(skipna=True)

    def _crest(self, s: pd.Series):
        return self._peak(s) / self._rms(s)

    def transform(self, X):
        X_new = pd.DataFrame(index=X.index)
        for c in X.columns:            
            for stats in self.to_compute:
                X_new[f'{c}_{stats}'] = getattr(self, f'_{stats}')(X[c])

        return X_new


class Difference(TransformerStep):
    def __init__(self,
                 feature_set1: list,
                 feature_set2: list,
                 name: Optional[str] = None):
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
        new_X = (new_X - X[self.feature_set2].values)
        return new_X


class EMD(TransformerStep):
    def __init__(self, n: int, name: Optional[str] = 'EMD'):
        super().__init__(name)
        self.n = n

    def transform(self, X):
        new_X = pd.DataFrame(index=X.index)
        for c in X.columns:
            try:
                imf = emd.sift.sift(X[c].values, max_imfs=self.n)
                for j in range(self.n):
                    if j < imf.shape[1]:
                        new_X[f'{c}_{j}'] = imf[:, j]
                    else:
                        new_X[f'{c}_{j}'] = np.nan
            except Exception as e:
                for j in range(self.n):
                    new_X[f'{c}_{j}'] = np.nan

        return new_X


class EMDFilter(TransformerStep):
    """Filter the signals using Empirical Mode decomposition

    Parameters
    ----------
    n: int
       Number of
    """
    def __init__(self, n: int, name: Optional[str] = 'EMD'):
        super().__init__(name)
        self.n = n

    def transform(self, X):
        new_X = pd.DataFrame(index=X.index)

        for c in X.columns:
            try:
                imf = emd.sift.sift(X[c].values, max_imfs=self.n)
                new_X[c] = np.sum(imf[:, 1:], axis=1)
            except Exception as e:
                new_X[c] = X[c]

        return new_X


class ChangesDetector(TransformerStep):
    """
    Compute how many changes there are in a categorical variable
    ['a', 'a', 'b', 'c] -> [0, 0, 1, 1]


    """
    def transform(self, X):
        return (X != X.shift(axis=0))

class Interactions(TransformerStep):
    def transform(self, X):
        X_new = pd.DataFrame(index=X.index)
        for c1, c2 in itertools.combinations(X.columns, 2):
            X_new[f'{c1}_{c2}'] = X[c1]*X[c2]
        return X_new
