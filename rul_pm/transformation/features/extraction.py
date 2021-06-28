import itertools
import logging
from typing import List, Optional

import emd
import mmh3
import numpy as np
from numpy.lib.arraysetops import isin
import pandas as pd
from rul_pm.transformation.features.extraction_numba import (compute,
                                                             roll_matrix,
                                                             stats_order)
from rul_pm.transformation.features.hurst import hurst_exponent
from rul_pm.transformation.transformerstep import TransformerStep
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

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
    Compute the column-wise sum each column
    """
    def __init__(self, column_name: str, name: Optional[str] = None):
        super().__init__(name)
        self.column_name = column_name

    def transform(self, X: pd.DataFrame):
        return pd.DataFrame(X.sum(axis=1), columns=[self.column_name])





class SampleNumber(TransformerStep):
    """Return a column with increasing number
    """
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """Construct a new DataFrame with a single column named called sample_number

        Parameters
        ----------
        X : pd.DataFrame
            Input life

        Returns
        -------
        pd.DataFrame
            A new DataFrame with the same index as the input with one column
            named sample_number that contains and increasing number
        """
        df = pd.DataFrame(index=X.index)
        df['sample_number'] = list(range(X.shape[0]))
        return df






class OneHotCategoricalPandas(TransformerStep):
    """Compute a one-hot encoding for a given feature

    Parameters
    ----------
    feature : str
        Feature name from which compute the one-hot encoding
    name : Optional[str], optional
        Step name, by default None
    """

    def __init__(self, feature:Optional[str]=None, categories: Optional[List[any]]=None, name: Optional[str] = None):
        super().__init__(name)
        self.feature = feature
        self.categories = categories
        self.fixed_categories = True
        if self.categories is None:
            self.categories = set()
            self.fixed_categories = False
        self.encoder = None
        

    def partial_fit(self, X:pd.DataFrame, y=None):
        """Compute incrementally the set of possible categories

        Parameters
        ----------
        X : pd.DataFrame
            The input life

        Returns
        -------
        OneHotCategoricalPandas
            self
        """
        if self.fixed_categories:
            return self
        if self.feature is None:
            self.feature = X.columns[0]
        self.categories.update(set(X[self.feature].unique()))
        return self

    def fit(self, X:pd.DataFrame, y=None):
        """Compute the set of possible categories

        Parameters
        ----------
        X : pd.DataFrame
            The input life


        Returns
        -------
        OneHotCategoricalPandas
            self
        """
        if self.fixed_categories:
            return self
        if self.feature is None:
            self.feature = X.columns[0]
        self.categories.update(set(X[self.feature].unique()))
        return self

    def transform(self, X:pd.DataFrame, y=None) ->pd.DataFrame:
        """Return a new DataFrame with the feature one-hot encoded

        Parameters
        ----------
        X : pd.DataFrame
            The input life
        y : [type], optional
            

        Returns
        -------
        pd.DataFrame
            A new dataframe with the same index as the input 
            with n columns, where n is the number of possible categories
            of the input column
        """
        categories = sorted(list([c for c in self.categories
                                  if c is not None]))
        d = pd.Categorical(X[self.feature], categories=categories)

        df = pd.get_dummies(d)
        df.index = X.index
        return df


class HashingEncodingCategorical(TransformerStep):
    """Compute a simple numerical encoding for a given feature

    Parameters
    ----------
    nbins: int
        Number of bins after the hash
    feature : str
        Feature name from which compute the simple encoding
    name : Optional[str], optional
        Step name, by default None
    """
    def __init__(self, nbins:int, feature:Optional[str]=None, name: Optional[str] = None):
        super().__init__(name)
        self.nbins = nbins
        self.feature = feature
        self.categories = set()
        self.encoder = None


    def transform(self, X, y=None):
        """Return a new DataFrame with the feature  encoded with integer numbers

        Parameters
        ----------
        X : pd.DataFrame
            The input life
        y : [type], optional
            

        Returns
        -------
        pd.DataFrame
            A new dataframe with the same index as the input 
            with 1 column
        """
        def hash(x):
            if isinstance(x, int):
                x = x.to_bytes((x.bit_length() + 7) // 8, 'little') 
            return (mmh3.hash(x) & 0xffffffff) % self.nbins

        if self.feature is None:
            self.feature = X.columns[0]
        X_new = pd.DataFrame(index=X.index)
        X_new['encoding'] =  X[self.feature].map(hash)
        return X_new

class SimpleEncodingCategorical(TransformerStep):
    """Compute a simple numerical encoding for a given feature

    Parameters
    ----------
    feature : str
        Feature name from which compute the simple encoding
    name : Optional[str], optional
        Step name, by default None
    """
    def __init__(self, feature:Optional[str]=None, name: Optional[str] = None):
        super().__init__(name)
        self.feature = feature
        self.categories = set()
        self.encoder = None

    def partial_fit(self, X:pd.DataFrame, y=None):
        """Compute incrementally the set of possible categories

        Parameters
        ----------
        X : pd.DataFrame
            The input life

        Returns
        -------
        SimpleEncodingCategorical
            self
        """
        if self.feature is None:
            self.feature = X.columns[0]
            print(self.feature)

        
        self.categories.update(set(X[self.feature].unique()))
        return self

    def fit(self, X:pd.DataFrame, y=None):
        """Compute the set of possible categories

        Parameters
        ----------
        X : pd.DataFrame
            The input life


        Returns
        -------
        OneHotCategoricalPandas
            self
        """
        if self.feature is None:
            self.feature = X.columns[0]

        self.categories.update(set(X[self.feature].unique()))
        return self

    def transform(self, X, y=None):
        """Return a new DataFrame with the feature  encoded with integer numbers

        Parameters
        ----------
        X : pd.DataFrame
            The input life
        y : [type], optional
            

        Returns
        -------
        pd.DataFrame
            A new dataframe with the same index as the input 
            with 1 column
        """
        categories = sorted(list([c for c in self.categories
                                  if c is not None]))
        d = pd.Categorical(X[self.feature], categories=categories)
        pd.DataFrame({'encoding': d.codes}, index=X.index)
        return pd.DataFrame({'encoding': d.codes}, index=X.index)


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





def rolling_kurtosis(s: pd.Series, window, min_periods):
    return s.rolling(window, min_periods=min_periods).kurt(skipna=True)


class LifeStatistics(TransformerStep):
    """Compute diverse number of features for each life.

    Returns a 1 row with the statistics computed for every feature


    The possible features are:

    - Kurtosis 
    - Skewness
    - Max
    - Min
    - Std
    - Peak
    - Impulse
    - Clearance
    - RMS
    - Shape
    - Crest
    - Hurst


    Parameters
    ----------
    to_compute : List[str], optional
        List of the features to compute, by default None
        Valid values are:
        'kurtosis', 'skewness', 'max', 'min', 'std', 'peak', 'impulse',
        'clearance', 'rms', 'shape', 'crest', 'hurst'
    name : Optional[str], optional
        Name of the step, by default None

    """
    def __init__(self, to_compute: Optional[List[str]]=None, name: Optional[str] = None):

        super().__init__(name)
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
        return s.kurt(skipna=True)

    def _skewness(self, s: pd.Series):
        return s.skew(skipna=True)

    def _max(self, s: pd.Series):
        return s.max(skipna=True)

    def _min(self, s: pd.Series):
        return s.min(skipna=True)

    def _std(self, s: pd.Series):
        return s.std(skipna=True)

    def _peak(self, s: pd.Series):
        return (s.max(skipna=True) - s.min(skipna=True))

    def _impulse(self, s: pd.Series):
        m = s.abs().mean()
        if m > 0:
            return self._peak(s) / m
        else:
            return 0

    def _clearance(self, s: pd.Series):
        m = s.abs().pow(1. / 2).mean()
        if m > 0:
            return (self._peak(s) / m)**2
        else:
            return 0

    def _rms(self, s: pd.Series):
        return (np.sqrt(s.pow(2).mean(skipna=True)))

    def _shape(self, s: pd.Series):
        m = s.abs().mean(skipna=True)
        if m > 0:
            return self._rms(s) / m
        else:
            return 0

    def _crest(self, s: pd.Series):
        m = self._rms(s)
        if m > 0:
            return self._peak(s) / m
        else:
            return 0

    def transform(self, X:pd.DataFrame)->pd.DataFrame:
        """Compute features from the given life

        Parameters
        ----------
        X : pd.DataFrame
            Input life

        Returns
        -------
        pd.DataFrame
            A new DataFrame with one row with n columns.
            Let m be the number of features of the life and 
            f the len(to_compute) ten  where n = m x f, 
        """
        X_new = pd.DataFrame(index=[0])
        for c in X.columns:
            for stats in self.to_compute:
                X_new[f'{c}_{stats}'] = getattr(self, f'_{stats}')(X[c])
        return X_new


class RollingStatisticsNumba(TransformerStep):
    """Compute diverse number of features using an rolling window. Numba implementation

    For each feature present in the life a number of feature will be computed for each time stamp
    Features from time and frequency domain can be computed

    The possible features are:

    Time domain:

    - Kurtosis 
    - Skewness
    - Max
    - Min
    - Std
    - Peak
    - Impulse
    - Clearance
    - RMS
    - Shape
    - Crest
    
    Frequency Domain:
    
    - fft_centroid: Centroid of the abscolute FT 
    - fft_variance: Variance of the abscolute FT 
    - fft_skew: Skewness of the abscolute FT 
    - fft_kurtosis: Kurtosis of the abscolute FT 
    - ps_centroid: Power spectrum centroid
    - ps_variance: Power spectrum variance
    - ps_skew: Power spectrum skewness
    - ps_kurtosis: Power spectrum skewness

    Parameters
    ----------
    window:int
        Size of the rolling window
    min_periods : int, optional
        The minimun number of points of the expanding window, by default 15
    time: bool 
        Wether to compute time domain features, by default True.
    frequency: bool = True,
        Wether to compute frequency domain features, by default True.
    select_features : Optional[List[Str]], optional
        Name of features to keep
    name: Optiona[str]
        Name of the step, by default None

    """
    def __init__(self,
                 window:int,
                 min_periods: int = 15,
                 time: bool = True,
                 frequency: bool = True,
                 select_features: Optional[list] = None,
                 name: Optional[str] = None):
        super().__init__(name)
        self.window = window
        self.min_periods = min_periods
        self.fs = 1
        self.time = time
        self.frequency = frequency
        self.select_features = select_features

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:

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


class RollingStatisticsPandas(TransformerStep):
    """Compute diverse number of features using an rolling window. Pandas implementation

    For each feature present in the life a number of feature will be computed for each time stamp

    The possible features are:

    Time domain:

    - Kurtosis 
    - Skewness
    - Max
    - Min
    - Std
    - Peak
    - Impulse
    - Clearance
    - RMS
    - Shape
    - Crest
    
    Parameters
    ----------
    window:int
        Size of the rolling window
    min_points : int, optional
        The minimun number of points of the expanding window, by default 15
    to_compute : Optional[List[Str]], optional
        Name of features to compute
    Possible values are:  
    'kurtosis', 'skewness', 'max', 'min', 'std', 'peak', 'impulse',
    'clearance', 'rms', 'shape', 'crest'
    name: Optiona[str]
        Name of the step, by default None

    """
    def __init__(self,
                 window: int = 15,
                 min_points=2,
                 to_compute: Optional[List[str]]=None,
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
        return s.rolling(self.window, self.min_points).skew(skipna=True)

    def _max(self, s: pd.Series):
        return s.rolling(self.window, self.min_points).max(skipna=True)

    def _min(self, s: pd.Series):
        return s.rolling(self.window, self.min_points).min(skipna=True)

    def _std(self, s: pd.Series):
        return s.rolling(self.window, self.min_points).std(skipna=True)

    def _peak(self, s: pd.Series):
        return (s.rolling(self.window, self.min_points).max(skipna=True) -
                s.rolling(self.window, self.min_points).min(skipna=True))

    def _impulse(self, s: pd.Series):
        return self._peak(s) / s.abs().rolling(self.window,
                                               self.min_points).mean()

    def _clearance(self, s: pd.Series):
        return self._peak(s) / s.abs().pow(1. / 2).rolling(
            self.window, self.min_points).mean().pow(2)

    def _rms(self, s: pd.Series):
        return (s.pow(2).rolling(
            self.window, self.min_points).mean(skipna=True).pow(1 / 2.))

    def _shape(self, s: pd.Series):
        return self._rms(s) / s.abs().rolling(
            self.window, self.min_points).mean(skipna=True)

    def _crest(self, s: pd.Series):
        return self._peak(s) / self._rms(s)

    def transform(self, X):
        X_new = pd.DataFrame(index=X.index)
        for c in X.columns:
            for stats in self.to_compute:
                X_new[f'{c}_{stats}'] = getattr(self, f'_{stats}')(X[c])

        return X_new


class ExpandingStatistics(TransformerStep):
    """Compute diverse number of features using an expandign window

    For each feature present in the life a number of feature will be computed for each time stamp

    The possible features are:

    - Kurtosis 
    - Skewness
    - Max
    - Min
    - Std
    - Peak
    - Impulse
    - Clearance
    - RMS
    - Shape
    - Crest
    - Hurst


    Parameters
    ----------
    min_points : int, optional
        The minimun number of points of the expanding window, by default 2
    to_compute : List[str], optional
        List of the features to compute, by default None
        Valid values are: 
        'kurtosis', 'skewness', 'max', 'min', 'std', 'peak', 'impulse',
        'clearance', 'rms', 'shape', 'crest', 'hurst'    
    name : Optional[str], optional
        Name of the step, by default None

    """
    def __init__(self,
                 min_points=2,
                 to_compute:List[str]=None,
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
        return s.expanding(min_periods=max(self.min_points, 50)).apply(
            lambda s: hurst_exponent(s, method='RS'))

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
    """Compute the difference between two set of features

    .. highlight:: python
    .. code-block:: python

        X[features1] - X[features2]

    Parameters
    ----------
    feature_set1 : list
        Feature list of the first group to substract
    feature_set2 : list
        Feature list of the second group to substract
    name : Optional[str], optional
        Name of the step, by default None    
    
    """
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
        self.feature_names_computed = False

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names_computed:
            self.feature_set1 = [self.column_name(X, c) for c in self.feature_set1]
            self.feature_set2 = [self.column_name(X, c) for c in self.feature_set2]
            feature_names_computed = True
        new_X = X[self.feature_set1].copy()
        new_X = (new_X - X[self.feature_set2].values)
        return new_X


class EMD(TransformerStep):
    """Compute the empirical mode decomposition of each feature

    Parameters
    ----------
    n : int
        Number of modes to compute
    name : Optional[str], optional
        [description], by default 'EMD'
    """
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
    """Compute how many changes there are in a categorical variable
    ['a', 'a', 'b', 'c] -> [0, 0, 1, 1]
    """
    def transform(self, X:pd.DataFrame)->pd.DataFrame:
        return (X != X.shift(axis=0))


class Interactions(TransformerStep):
    """Compute pairwise interactions between the features
    """
    def transform(self, X:pd.DataFrame)->pd.DataFrame:
        """Transform the given life computing the iteractions between features

        Parameters
        ----------
        X : pd.DataFrame
            Input life

        Returns
        -------
        pd.DataFrame
            A new dataframe with the same index as the input
            with n*(n-1) / 2 columns with the interactions between the features
        """
        X_new = pd.DataFrame(index=X.index)
        for c1, c2 in itertools.combinations(X.columns, 2):
            X_new[f'{c1}_{c2}'] = X[c1] * X[c2]
        return X_new
