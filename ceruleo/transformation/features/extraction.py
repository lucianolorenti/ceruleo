import itertools
import logging
from typing import Dict, List, Optional

import emd
import mmh3
import numpy as np
import pandas as pd
from numpy.lib.arraysetops import isin
from pandas.core.window.expanding import Expanding
from pyts.transformation import ROCKET as pyts_ROCKET
from functools import reduce
import pywt

# try:
#    from temporis.transformation.features.hurst import hurst_exponent
# except:
#    pass
from ceruleo.transformation import TransformerStep
from ceruleo.transformation.features.rolling_windows import apply_rolling_data
from ceruleo.transformation.functional.transformers import Transformer
from ceruleo.transformation.utils import SKLearnTransformerWrapper

logger = logging.getLogger(__name__)


class TimeToPreviousBinaryValue(TransformerStep):
    """Return a column with increasing number"""

    def time_to_previous_event(self, X: pd.DataFrame, c: str):
        def min_idex(group):
            if group.iloc[0, 0] == 0:
                return np.nan
            else:
                return np.min(group.index)

        X_c_cumsum = X[[c]].cumsum()
        min_index = X_c_cumsum.groupby(c).apply(min_idex)
        X_merged = X_c_cumsum.merge(
            pd.DataFrame(min_index, columns=["start"]), left_on=c, right_index=True
        )
        return X_merged.index - X_merged["start"]

    def transform(self, X: pd.DataFrame):
        new_X = pd.DataFrame(index=X.index)
        for c in X.columns:
            new_X[f"ttp_{c}"] = self.time_to_previous_event(X, c)
        return new_X


class ColumnWiseSum(TransformerStep):
    """
    Compute the column-wise sum each column
    """

    def __init__(self, column_name: str, name: Optional[str] = None):
        super().__init__(name=name)
        self.column_name = column_name

    def transform(self, X: pd.DataFrame):
        return pd.DataFrame(X.sum(axis=1), columns=[self.column_name])


class SampleNumber(TransformerStep):
    """Return a column with increasing number"""

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=X.index)
        df["sample_number"] = list(range(X.shape[0]))
        return df


class OneHotCategorical(TransformerStep):
    """Compute a one-hot encoding for a given feature

    Parameters:
        feature: Feature name from which compute the one-hot encoding
        name: Step name, by default None
    """

    def __init__(
        self,
        *,
        feature: Optional[str] = None,
        categories: Optional[List[any]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.feature = feature
        self.categories = categories
        self.fixed_categories = True
        if self.categories is None:
            self.categories = set()
            self.fixed_categories = False
        self.encoder = None

    def partial_fit(self, X: pd.DataFrame, y=None):
        if self.fixed_categories:
            return self
        if self.feature is None:
            self.feature = X.columns[0]
        self.categories.update(set(X[self.feature].unique()))
        return self

    def fit(self, X: pd.DataFrame, y=None):
        if self.fixed_categories:
            return self
        if self.feature is None:
            self.feature = X.columns[0]
        self.categories.update(set(X[self.feature].unique()))
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        categories = sorted(list([c for c in self.categories if c is not None]))
        d = pd.Categorical(X[self.feature], categories=categories)

        df = pd.get_dummies(d)
        df.index = X.index
        return df


class HashingEncodingCategorical(TransformerStep):
    """Compute a simple numerical encoding for a given feature

    Parameters:
        nbins: Number of bins after the hash
        feature: Feature name from which compute the simple encoding
        name: Step name
    """

    def __init__(
        self, *, nbins: int, feature: Optional[str] = None, name: Optional[str] = None
    ):
        super().__init__(name=name)
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
                x = x.to_bytes((x.bit_length() + 7) // 8, "little")
            return (mmh3.hash(x) & 0xFFFFFFFF) % self.nbins

        if self.feature is None:
            self.feature = X.columns[0]
        X_new = pd.DataFrame(index=X.index)
        X_new["encoding"] = X[self.feature].map(hash)
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

    def __init__(self, *, feature: Optional[str] = None, name: Optional[str] = None):
        super().__init__(name=name)
        self.feature = feature
        self.categories = set()
        self.encoder = None

    def partial_fit(self, X: pd.DataFrame, y=None):
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

        self.categories.update(set(X[self.feature].unique()))

        return self

    def fit(self, X: pd.DataFrame, y=None):
        """Compute the set of possible categories

        Parameters
        ----------
        X : pd.DataFrame
            The input life


        Returns
        -------
        OneHotCategorical
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
        categories = sorted(list([c for c in self.categories if c is not None]))
        d = pd.Categorical(X[self.feature], categories=categories)
        return pd.DataFrame({"encoding": d.codes}, index=X.index)




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

    def __init__(
        self, *,to_compute: Optional[List[str]] = None, name: Optional[str] = None
    ):

        super().__init__(name=name)
        valid_stats = [
            "kurtosis",
            "skewness",
            "max",
            "min",
            "std",
            "peak",
            "impulse",
            "clearance",
            "rms",
            "shape",
            "crest",
        ]
        if to_compute is None:
            self.to_compute = valid_stats
        else:
            for f in to_compute:
                if f not in valid_stats:
                    raise ValueError(
                        f"Invalid feature to compute {f}. Valids are {valid_stats}"
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
        return s.max(skipna=True) - s.min(skipna=True)

    def _impulse(self, s: pd.Series):
        m = s.abs().mean()
        if m > 0:
            return self._peak(s) / m
        else:
            return 0

    def _clearance(self, s: pd.Series):
        m = s.abs().pow(1.0 / 2).mean()
        if m > 0:
            return (self._peak(s) / m) ** 2
        else:
            return 0

    def _rms(self, s: pd.Series):
        return np.sqrt(s.pow(2).mean(skipna=True))

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

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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
                X_new[f"{c}_{stats}"] = getattr(self, f"_{stats}")(X[c])
        return X_new


class RollingStatistics(TransformerStep):
    """Compute diverse number of features using an rolling window.

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

    def __init__(
        self,
        *,
        window: int = 15,
        min_points=2,
        to_compute: Optional[List[str]] = None,
        specific: Optional[Dict[str, List[str]]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.window = window
        self.min_points = min_points
        valid_stats = [
            "mean",
            "kurtosis",
            "skewness",
            "max",
            "min",
            "std",
            "peak",
            "impulse",
            "clearance",
            "rms",
            "shape",
            "crest",
            "deviance",
            "std_atan",
            "std_acosh",
            "std_asinh",
            "energy",
        ]

        if to_compute is not None and specific is not None:
            raise ValueError('Only one of to_compute or specific should be used')
        self.specific = specific
        self.to_compute = to_compute
        if to_compute is None:
            if specific is None:
                self.to_compute = valid_stats
            else:
                self.specific = specific
        else:
            for f in to_compute:
                if f not in valid_stats:
                    raise ValueError(
                        f"Invalid feature to compute {f}. Valids are {valid_stats}"
                    )
            self.to_compute = to_compute

    def partial_fit(self, X, y=None):
        return self

    def fit(self, X, y=None):
        return self

    def _std_asinh(self, X, rolling, abs_rolling):
        return (
            X.apply(np.arcsinh).rolling(self.window, self.min_points).std(skipna=True)
        )

    def _std_acosh(self, X, rolling, abs_rolling):
        return (
            X.apply(np.arccosh).rolling(self.window, self.min_points).std(skipna=True)
        )

    def _energy(self, X, rolling, abs_rolling):
        return X.pow(2).rolling(self.window, self.min_points).sum()

    def _std_atan(self, X, rolling, abs_rolling):
        return X.apply(np.arctan).rolling(self.window, self.min_points).std(skipna=True)

    def _mean(self, X, rolling, abs_rolling):
        return rolling.mean(skipna=True)

    def _kurtosis(self, X, rolling, abs_rolling):
        return rolling.kurt(skipna=True)

    def _skewness(self, X, rolling, abs_rolling):
        return rolling.skew(skipna=True)

    def _max(self, X, rolling, abs_rolling):
        return rolling.max(skipna=True)

    def _min(self, X, rolling, abs_rolling):
        return rolling.min(skipna=True)

    def _std(self, X, rolling, abs_rolling):
        return rolling.std(skipna=True)

    def _peak(self, X, rolling, abs_rolling):
        return rolling.max(skipna=True) - rolling.min(skipna=True)

    def _impulse(self, X, rolling, abs_rolling):
        return self._peak(X, rolling, abs_rolling) / abs_rolling.mean()

    def _deviance(self, X, rolling, abs_rolling):
        return (X - rolling.mean()) / rolling.std()

    def _clearance(self, X, rolling, abs_rolling):
        return self._peak(X, rolling, abs_rolling) / X.abs().pow(1.0 / 2).rolling(
            self.window, self.min_points
        ).mean().pow(2)

    def _rms(self, X, rolling, abs_rolling):
        return (
            X.pow(2)
            .rolling(self.window, self.min_points)
            .mean(skipna=True)
            .pow(1 / 2.0)
        )

    def _shape(self, X, rolling, abs_rolling):
        return self._rms(X, rolling, abs_rolling) / abs_rolling.mean(skipna=True)

    def _crest(self, X, rolling, abs_rolling):
        return self._peak(X, rolling, abs_rolling) / self._rms(X, rolling, abs_rolling)

    def _compute_column_names(self, X:pd.DataFrame):
        columns = []
        if self.to_compute is not None:
            for stats in self.to_compute:
                for c in X.columns:
                    columns.append(f"{c}_{stats}")
        else:
            for c in self.specific.keys():
                for stats in self.specific[c]:
                    columns.append(f"{c}_{stats}")
        return columns 

    def _transform_all_features(self, X:pd.DataFrame, X_new:pd.DataFrame, rolling, abs_rolling):

        for stats in self.to_compute:
            columns_to_assign = [f"{c}_{stats}" for c in X.columns]
            out = getattr(self, f"_{stats}")(X, rolling, abs_rolling)
            X_new.loc[:, columns_to_assign] = out.values


    def _transform_specific(self,  X:pd.DataFrame, X_new:pd.DataFrame, rolling, abs_rolling):
        for c in self.specific.keys():
            for stats in self.specific[c]: 
                feature = f"{c}_{stats}"
                out = getattr(self, f"_{stats}")(X[c], rolling[c], abs_rolling[c])
                X_new.loc[:, feature] = out.values

    def transform(self, X:pd.DataFrame):
        columns = self._compute_column_names(X)
        
        X_new = pd.DataFrame(index=X.index, columns=columns)
        rolling = X.rolling(self.window, self.min_points)
        abs_rolling = X.abs().rolling(self.window, self.min_points)
        if self.to_compute is not None:
            self._transform_all_features(X, X_new, rolling, abs_rolling)
        else:
            self._transform_specific(X, X_new, rolling, abs_rolling)
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

    def __init__(
        self, *, min_points=2, to_compute: List[str] = None, specific: Optional[Dict[str, List[str]]] = None, name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.min_points = min_points
        valid_stats = [
            "kurtosis",
            "skewness",
            "max",
            "min",
            "std",
            "peak",
            "impulse",
            "clearance",
            "rms",
            "shape",
            "crest",
            "mean",
            "deviance",
            "std_atan",
            "energy",
            "std_acosh",
            "std_asinh",
        ]
        not_default = [
            'energy',
            'deviance'
        ]
        if to_compute is not None and specific is not None:
            raise ValueError('Only one of to_compute or specific should be used')
        self.specific = specific
        self.to_compute = to_compute
        if to_compute is None:
            if specific is None:
                self.to_compute = list(set(valid_stats) - set(not_default))
            else:
                self.specific = specific
        else:
            for f in to_compute:
                if f not in valid_stats:
                    raise ValueError(
                        f"Invalid feature to compute {f}. Valids are {valid_stats}"
                    )
            self.to_compute = to_compute

    def partial_fit(self, X, y=None):
        return self

    def fit(self, X, y=None):
        return self

    def _std_asinh(
        self,
        x: pd.Series,
        s: Expanding,
        s_abs: Expanding,
        s_abs_sqrt: Expanding,
        s_sq: Expanding,
    ):
        return x.apply(np.arcsinh).expanding(self.min_points).std(skipna=True)

    def _std_acosh(
        self,
        x: pd.Series,
        s: Expanding,
        s_abs: Expanding,
        s_abs_sqrt: Expanding,
        s_sq: Expanding,
    ):
        return x.apply(np.arccosh).expanding(self.min_points).std(skipna=True)

    def _energy(
        self,
        x: pd.Series,
        s: Expanding,
        s_abs: Expanding,
        s_abs_sqrt: Expanding,
        s_sq: Expanding,
    ):
        return x.pow(2).expanding(self.min_points).sum(skipna=True)

    def _std_atan(
        self,
        x: pd.Series,
        s: Expanding,
        s_abs: Expanding,
        s_abs_sqrt: Expanding,
        s_sq: Expanding,
    ):
        return x.apply(np.arctan).expanding(self.min_points).std(skipna=True)

    def _kurtosis(
        self,
        x: pd.Series,
        s: Expanding,
        s_abs: Expanding,
        s_abs_sqrt: Expanding,
        s_sq: Expanding,
    ):
        return s.kurt(skipna=True)

    def _skewness(
        self,
        x: pd.Series,
        s: Expanding,
        s_abs: Expanding,
        s_abs_sqrt: Expanding,
        s_sq: Expanding,
    ):
        return s.skew(skipna=True)

    def _max(
        self,
        x: pd.Series,
        s: Expanding,
        s_abs: Expanding,
        s_abs_sqrt: Expanding,
        s_sq: Expanding,
    ):
        return s.max(skipna=True)

    def _min(
        self,
        x: pd.Series,
        s: Expanding,
        s_abs: Expanding,
        s_abs_sqrt: Expanding,
        s_sq: Expanding,
    ):
        return s.min(skipna=True)

    def _std(
        self,
        x: pd.Series,
        s: Expanding,
        s_abs: Expanding,
        s_abs_sqrt: Expanding,
        s_sq: Expanding,
    ):
        return s.std(skipna=True)

    def _peak(
        self,
        x: pd.Series,
        s: Expanding,
        s_abs: Expanding,
        s_abs_sqrt: Expanding,
        s_sq: Expanding,
    ):
        return s.max(skipna=True) - s.min(skipna=True)

    def _impulse(
        self,
        x: pd.Series,
        s: Expanding,
        s_abs: Expanding,
        s_abs_sqrt: Expanding,
        s_sq: Expanding,
    ):
        return self._peak(x, s, s_abs, s_abs_sqrt, s_sq) / s_abs.mean()

    def _deviance(
        self,
        x: pd.Series,
        s: Expanding,
        s_abs: Expanding,
        s_abs_sqrt: Expanding,
        s_sq: Expanding,
    ):
        return (x - s.mean(skipna=True)) / (s.std(skipna=True) + 0.00000000001)

    def _clearance(
        self,
        x: pd.Series,
        s: Expanding,
        s_abs: Expanding,
        s_abs_sqrt: Expanding,
        s_sq: Expanding,
    ):
        return self._peak(x, s, s_abs, s_abs_sqrt, s_sq) / s_abs_sqrt.mean().pow(2)

    # def _hurst(
    #    self,
    #    x: pd.Series,
    #    s: Expanding,
    #    s_abs: Expanding,
    #    s_abs_sqrt: Expanding,
    #    s_sq: Expanding,
    # ):
    #    return s.apply(lambda s: hurst_exponent(s, method="RS"))

    def _rms(
        self,
        x: pd.Series,
        s: Expanding,
        s_abs: Expanding,
        s_abs_sqrt: Expanding,
        s_sq: Expanding,
    ):
        return s_sq.mean(skipna=True).pow(1 / 2.0)

    def _mean(
        self,
        x: pd.Series,
        s: Expanding,
        s_abs: Expanding,
        s_abs_sqrt: Expanding,
        s_sq: Expanding,
    ):
        return s.mean(skipna=True)

    def _shape(
        self,
        x: pd.Series,
        s: Expanding,
        s_abs: Expanding,
        s_abs_sqrt: Expanding,
        s_sq: Expanding,
    ):
        return self._rms(x, s, s_abs, s_abs_sqrt, s_sq) / s_abs.mean(skipna=True)

    def _crest(
        self,
        x: pd.Series,
        s: Expanding,
        s_abs: Expanding,
        s_abs_sqrt: Expanding,
        s_sq: Expanding,
    ):
        return self._peak(x, s, s_abs, s_abs_sqrt, s_sq) / self._rms(
            x, s, s_abs, s_abs_sqrt, s_sq
        )


    def _compute_column_names(self, X:pd.DataFrame):
        columns = []
        if self.to_compute is not None:
            for stats in self.to_compute:
                for c in X.columns:
                    columns.append(f"{c}_{stats}")
        else:
            for c in self.specific.keys():
                for stats in self.specific[c]:
                    columns.append(f"{c}_{stats}")
        return columns 

    def _transform_all_features(self, X:pd.DataFrame, X_new:pd.DataFrame, expanding, s_abs, s_abs_sqrt, s_sq):

        for stats in self.to_compute:
            columns_to_assign = [f"{c}_{stats}" for c in X.columns]
            out = getattr(self, f"_{stats}")(
                    X, expanding, s_abs, s_abs_sqrt, s_sq
                )
            X_new.loc[:, columns_to_assign] = out.values


    def _transform_specific(self,  X:pd.DataFrame, X_new:pd.DataFrame, expanding, s_abs, s_abs_sqrt, s_sq):
        for c in self.specific.keys():
            for stats in self.specific[c]: 
                feature = f"{c}_{stats}"
                out = getattr(self, f"_{stats}")(
                    X[c], expanding[c], s_abs[c], s_abs_sqrt[c], s_sq[c]
                )
                X_new.loc[:, feature] = out.values

    def transform(self, X:pd.DataFrame):
        columns = self._compute_column_names(X)
        
        X_new = pd.DataFrame(index=X.index, columns=columns)
        expanding = X.expanding(self.min_points)
        s_abs = X.abs().expanding(self.min_points)
        s_abs_sqrt = X.abs().pow(1.0 / 2).expanding(self.min_points)
        s_sq = X.pow(2).expanding(self.min_points)
        if self.to_compute is not None:
            self._transform_all_features(X, X_new, expanding, s_abs, s_abs_sqrt, s_sq)
        else:
            self._transform_specific(X, X_new, expanding, s_abs, s_abs_sqrt, s_sq)
        return X_new


class Difference(TransformerStep):
    """Compute the difference between two set of features

    Example:

        X[features1] - X[features2]

    Parameters
    
        feature_set1: Feature list of the first group to substract
        feature_set2:Feature list of the second group to substract
        name: Name of the step, by default None

    """

    def __init__(
        self, *, feature_set1: list, feature_set2: list, name: Optional[str] = None
    ):
        super().__init__(name=name)
        if len(feature_set1) != len(feature_set2):
            raise ValueError(
                "Feature set 1 and feature set 2 must have the same length"
            )
        self.feature_set1 = feature_set1
        self.feature_set2 = feature_set2
        self.feature_names_computed = False

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names_computed:
            self.feature_set1 = [self.find_feature(X, c) for c in self.feature_set1]
            self.feature_set2 = [self.find_feature(X, c) for c in self.feature_set2]
            feature_names_computed = True
        new_X = X[self.feature_set1].copy()
        new_X = new_X - X[self.feature_set2].values
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

    def __init__(self, *, n: int, name: Optional[str] = "EMD"):
        super().__init__(name=name)
        self.n = n

    def transform(self, X):
        new_X = pd.DataFrame(index=X.index)
        for c in X.columns:
            try:
                imf = emd.sift.sift(X[c].values, max_imfs=self.n)
                for j in range(self.n):
                    if j < imf.shape[1]:
                        new_X[f"{c}_{j}"] = imf[:, j]
                    else:
                        new_X[f"{c}_{j}"] = np.nan
            except Exception as e:
                for j in range(self.n):
                    new_X[f"{c}_{j}"] = np.nan

        return new_X


class SlidingNonOverlappingEMD(TransformerStep):
    def __init__(
        self, *, window_size: int, max_imfs: int, keep: Optional[int] = None, **kwargs
    ):
        super().__init__(*kwargs)
        self.window_size = window_size
        self.strides = window_size
        self.max_imfs = max_imfs
        if keep is None:
            keep = self.max_imfs
        assert keep <= self.max_imfs
        self.keep = keep

    def transform(self, X: pd.DataFrame):
        def _emd(values: np.ndarray):
            out = np.zeros((values.shape[0], self.keep))
            try:
                v = emd.sift.sift(values, max_imfs=self.max_imfs)
                out[:, : self.keep] = v[:, : self.keep]
            except emd.support.EMDSiftCovergeError:
                pass

            return out

        column_list = []
        for c in X.columns:
            for i in range(self.keep):
                column_list.append(f"imf_{i}_{c}")
        out = pd.DataFrame(index=X.index, columns=column_list, dtype=np.float32)
        for c in X.columns:
            emd_computed = apply_rolling_data(
                X[c].values, _emd, self.window_size, self.strides
            )

            out.loc[:, [f"imf_{i}_{c}" for i in range(self.keep)]] = emd_computed
        return out


class EMDFilter(TransformerStep):
    """Filter the signals using Empirical Mode decomposition

    Parameters
    ----------
    n: int
       Number of
    """

    def __init__(self, *, n: int, min_imf:int, max_imf:int, name: Optional[str] = "EMD"):
        super().__init__(name=name)
        self.n = n
        self.min_imf = min_imf
        self.max_imf = max_imf

    def transform(self, X):
        new_X = pd.DataFrame(index=X.index)

        for c in X.columns:
            try:
                imf = emd.sift.sift(X[c].values, max_imfs=self.n)
                new_X[c] = np.sum(imf[:, self.min_imf:self.max_imf], axis=1)
            except Exception as e:
                new_X[c] = X[c]

        return new_X


class ChangesDetector(TransformerStep):
    """Compute how many changes there are in a categorical variable
    ['a', 'a', 'b', 'c] -> [0, 0, 1, 1]
    """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X != X.shift(axis=0)


class Interactions(TransformerStep):
    """Compute pairwise interactions between the features"""

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_new = pd.DataFrame(index=X.index)
        for c1, c2 in itertools.combinations(X.columns, 2):
            X_new[f"{c1}_{c2}"] = X[c1] * X[c2]
        return X_new


class ROCKET(SKLearnTransformerWrapper):
    def __init__(
        self,
        *,
        n_kernels=1000,
        kernel_sizes=(7, 9, 11),
        random_state=None,
        name: Optional[str] = "ROCKET",
    ):
        transformer = pyts_ROCKET(
            n_kernels=n_kernels, kernel_sizes=kernel_sizes, random_state=random_state
        )
        super().__init__(transformer=transformer, name=name)

    def _column_names(self, X):
        a = ["Filter {i}" for i in range(self.transformer.n_kernels * 2)]
        return a


class DoG(TransformerStep):
    def __init__(
        self,
        *,
        window_size: int,
        std: List[float],
        min_points: int = 1,
        center: bool = False,
        **kwargs,
    ):
        super().__init__(*kwargs)
        self.window_size = window_size
        self.std = std
        self.center = center
        self.min_points = min_points

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X1 = X.rolling(
            window=self.window_size,
            win_type="gaussian",
            center=self.center,
            min_periods=self.min_points,
        ).mean(std=self.std[0])
        X2 = X.rolling(
            window=self.window_size,
            win_type="gaussian",
            center=self.center,
            min_periods=self.min_points,
        ).mean(std=self.std[1])
        return X1 - X2


def wrcoef(X, coef_type, coeffs, wavename, level):
    N = np.array(X).size
    a, ds = coeffs[0], list(reversed(coeffs[1:]))

    if coef_type == "a":
        return pywt.upcoef("a", a, wavename, level=level)[:N]
    elif coef_type == "d":
        return pywt.upcoef("d", ds[level - 1], wavename, level=level)[:N]
    else:
        raise ValueError("Invalid coefficient type: {}".format(coef_type))


class SlidingNonOverlappingWaveletDecomposition(TransformerStep):
    """

    # TODO TEST
    X = signal
    coeffs = pywt.wavedec(X, 'db1', level=level)
    A4 = wrcoef(X, 'a', coeffs, 'db1', level)
    D4 = wrcoef(X, 'd', coeffs, 'db1', level)
    D3 = wrcoef(X, 'd', coeffs, 'db1', 3)
    D2 = wrcoef(X, 'd', coeffs, 'db1', 2)
    D1 = wrcoef(X, 'd', coeffs, 'db1', 1)
    r = A4 + D4 + D3 + D2 + D1
    assert(np.mean(r-X) < 0.00000)

    Parameters
    ----------
    TransformerStep : [type]
        [description]
    """

    def __init__(
        self, *, window_size: int, level: int, wavelet: str, keep: List[str], **kwargs
    ):
        super().__init__(*kwargs)
        self.wavelet = wavelet
        self.level = level
        self.keep = keep
        self.window_size = window_size
        self.strides = window_size

    def transform(self, X: pd.DataFrame):
        def _wavelet(values: np.ndarray):
            coeffs = pywt.wavedec(values, self.wavelet, level=self.level)
            out = np.zeros((values.shape[0], len(self.keep)))
            for i, s in enumerate(self.keep):
                part, level = s
                out[:, i] = wrcoef(
                    values, part.lower(), coeffs, self.wavelet, int(level)
                )
            return out

        column_list = []
        for c in X.columns:
            for name in self.keep:
                column_list.append(f"wavelet_{name}_{c}")
        out = pd.DataFrame(index=X.index, columns=column_list, dtype=np.float32)
        for c in X.columns:
            wv_computed = apply_rolling_data(
                X[c].values, _wavelet, self.window_size, self.strides
            )

            out.loc[:, [f"wavelet_{name}_{c}" for name in self.keep]] = wv_computed
        return out
