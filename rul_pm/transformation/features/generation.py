
import logging

import numpy as np
import pandas as pd
import scipy
from scipy.signal import firwin, lfilter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


class LifeCumSum(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, life_id_col='life'):
        self.life_id_col = life_id_col
        self.columns = columns

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.columns
        self.columns = [f for f in self.columns if f != self.life_id_col]
        return self

    def transform(self, X):
        X_new = pd.DataFrame(
            np.zeros((len(X.index), len(self.columns))),
            index=X.index,
            columns=self.columns
        )
        X_new.columns = [f'{c}_cumsum' for c in self.columns]
        for life in X[self.life_id_col].unique():
            data = X[X[self.life_id_col] == life]
            data_columns = (data[self.columns]
                            .cumsum())
            for c in self.columns:
                X_new.loc[X[self.life_id_col] == life,
                          f'{c}_cumsum'] = data_columns[c]
        return X_new


class LifeExceededCumSum(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None,  life_id_col='life', lambda_=0.5):
        self.lambda_ = lambda_
        self.UCL = None
        self.LCL = None
        self.life_id_col = life_id_col
        self.columns = columns

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = set(X.columns.values) - set(['life'])
        mean = np.nanmean(X[self.columns], axis=0)
        s = np.sqrt(self.lambda_ / (2-self.lambda_)) * \
            np.nanstd(X[self.columns], axis=0)
        self.UCL = mean + 3*s
        self.LCL = mean - 3*s
        return self

    def transform(self, X):

        cnames = [f'{c}_cumsum' for c in self.columns]
        new_X = pd.DataFrame({}, columns=cnames, index=X.index)
        for life in X[self.life_id_col].unique():
            data = X[X[self.life_id_col] == life]
            data_columns = data[self.columns]
            mask = (
                (data_columns < (self.LCL)) |
                (data_columns > (self.UCL))
            )
            df_cumsum = mask.astype('int').cumsum()
            for c in self.columns:
                new_X.loc[
                    X[self.life_id_col] == life,
                    f'{c}_cumsum'] = df_cumsum[c]

        return new_X


class OneHotCategoricalPandas(BaseEstimator, TransformerMixin):
    def __init__(self, feature):
        self.feature = feature
        self.encoder = None

    def fit(self, X, y=None):

        self.encoder = (
            OneHotEncoder(handle_unknown='ignore', sparse=False)
            .fit(X[self.feature].values.reshape(-1, 1)))
        return self

    def transform(self, X, y=None):
        categories = [
            f'{self.feature}_{feature_name}'
            for feature_name in self.encoder.categories_[0]]
        d = self.encoder.transform(X[self.feature].values.reshape(-1, 1))
        return pd.DataFrame(d,
                            index=X.index,
                            columns=categories)


class LowFrequencies(BaseEstimator, TransformerMixin):
    def __init__(self, window):
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


class HighFrequencies(BaseEstimator, TransformerMixin):
    def __init__(self, window):
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


class MedianFilter(BaseEstimator, TransformerMixin):
    def __init__(self, window: int, min_periods: int = 15):
        self.window = window
        self.min_periods = min_periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.rolling(self.window, min_periods=self.min_periods).median()


class MeanFilter(BaseEstimator, TransformerMixin):
    def __init__(self, window: int, min_periods: int = 15):
        self.window = window
        self.min_periods = min_periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.rolling(self.window, min_periods=self.min_periods).mean()


def rolling_kurtosis(s: pd.Series, window, min_periods):
    return s.rolling(window, min_periods=min_periods).kurt(skipna=True)


class RollingStatistics(BaseEstimator, TransformerMixin):
    def __init__(self, window, min_periods: int = 15, to_compute=None):
        self.window = window
        self.min_periods = min_periods

        if to_compute is None:
            self.to_compute = ['kurtosis', 'skeweness', 'max',
                               'min', 'std', 'peak', 'impulse', 'clearance',
                               'rms', 'shape', 'crest', 'spectral_kurtosis']
        else:
            valid_stats = ['kurtosis', 'skeweness', 'mean', 'max',
                           'min', 'std', 'peak', 'impulse', 'clearance',
                           'rms', 'shape', 'crest', 'spectral_kurtosis']
            for f in to_compute:
                if f not in valid_stats:
                    raise ValueError(
                        f'Invalid feature to compute {f}. Valids are {valid_stats}')
            self.to_compute = to_compute

    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
        return self

    def _mean(self, s: pd.Series):
        return s.rolling(self.window, min_periods=self.min_periods).mean(skipna=True)

    def _kurtosis(self, s: pd.Series):
        return s.rolling(self.window, min_periods=self.min_periods).kurt(skipna=True)

    def _skeweness(self, s: pd.Series):
        return s.rolling(self.window, min_periods=self.min_periods).skew(skipna=True)

    def _max(self, s: pd.Series):
        return s.rolling(self.window, min_periods=self.min_periods).max(skipna=True)

    def _min(self, s: pd.Series):
        return s.rolling(self.window, min_periods=self.min_periods).min(skipna=True)

    def _std(self, s: pd.Series):
        return s.rolling(self.window, min_periods=self.min_periods).std(skipna=True)

    def _peak(self, s: pd.Series):
        return (s.rolling(self.window, min_periods=self.min_periods).max(skipna=True) -
                s.rolling(self.window, min_periods=self.min_periods).min(skipna=True))

    def _spectral_kurtosis(self, s: pd.Series):
        def spectral_kurtosis(x):
            return scipy.stats.kurtosis(np.abs(np.fft.rfft(x)))
        return s.rolling(self.window).apply(spectral_kurtosis)

    def _impulse(self, s: pd.Series):
        return self._peak(s) / s.abs().rolling(self.window, min_periods=self.min_periods).mean()

    def _clearance(self, s: pd.Series):
        return self._peak(s) / s.abs().pow(1./2).rolling(self.window, min_periods=self.min_periods).mean().pow(2)

    def _rms(self, s: pd.Series):
        return (s.pow(2)
                 .rolling(self.window, min_periods=self.min_periods)
                 .mean(skipna=True)
                 .pow(1/2.))

    def _shape(self, s: pd.Series):
        return self._rms(s) / s.abs().rolling(self.window, min_periods=self.min_periods).mean(skipna=True)

    def _crest(self, s: pd.Series):
        return self._peak(s) / self._rms(s)

    def transform(self, X):

        X_new = pd.DataFrame(index=X.index)

        for c in X.columns:
            for stats in self.to_compute:
                X_new[f'{c}_{stats}'] = getattr(self, f'_{stats}')(X[c])

        return X_new


class ExpandingStatistics(BaseEstimator, TransformerMixin):
    def __init__(self, min_points=2,  to_compute=None):
        self.min_points = min_points
        valid_stats = ['kurtosis', 'skeweness', 'max',
                       'min', 'std', 'peak', 'impulse', 'clearance',
                       'rms', 'shape', 'crest']
        if to_compute is None:
            self.to_compute = ['kurtosis', 'skeweness', 'max',
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
        return s.expanding(self.min_points).kurt(skipna=True)

    def _skeweness(self, s: pd.Series):
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
