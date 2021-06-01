from typing import Optional, Union

import numpy as np
import pandas as pd
from rul_pm.transformation.features.extraction import (compute, roll_matrix,
                                                       stats_order)
from rul_pm.transformation.transformerstep import TransformerStep
from rul_pm.transformation.utils.kurtogram import fast_kurtogram
from scipy.signal import firwin, lfilter, savgol_filter
from sklearn.cluster import MiniBatchKMeans


class SavitzkyGolayTransformer(TransformerStep):
    """Filter each feature using LOESS

    Parameters
    ----------
    window : int
        Window size of the filter
    order : int, optional
        Order of the filter, by default 2
    name : Optional[str], optional
        Step name, by default None
    """

    def __init__(self, window: int, order: int = 2, name: Optional[str] = None):

        super().__init__(name)
        self.window = window
        self.order = order

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Return a new dataframe with the features filtered

        Parameters
        ----------
        X : pd.DataFrame
            Input life


        Returns
        -------
        pd.DataFrame
            A new DatafFrame with the same index as the input with the features filtered
        """
        if X.shape[0] > self.window:
            return pd.DataFrame(
                savgol_filter(X, self.window, self.order, axis=0),
                columns=X.columns,
                index=X.index,
            )
        else:
            return X


class MeanFilter(TransformerStep):
    """Filter each feature using a rolling mean filter

    Parameters
    ----------
    window : int
        Size of the rolling window
    min_periods : int, optional
        Minimum number of points of the rolling window, by default 15
    name : Optional[str], optional
        Name of the step, by default None
    """

    def __init__(self, window: int, min_periods: int = 15, name: Optional[str] = None):

        super().__init__(name)
        self.window = window
        self.min_periods = min_periods

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X.rolling(self.window, min_periods=self.min_periods).mean(skip_na=True)


class MedianFilter(TransformerStep):
    """Filter each feature using a rolling median filter

    Parameters
    ----------
    window : int
        Size of the rolling window
    min_periods : int, optional
        Minimum number of points of the rolling window, by default 15
    name : Optional[str], optional
        Name of the step, by default None
    """

    def __init__(self, window: int, min_periods: int = 15, name: Optional[str] = None):
        super().__init__(name)
        self.window = window
        self.min_periods = min_periods

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X.rolling(self.window, min_periods=self.min_periods).median(skip_na=True)


class OneDimensionalKMeans(TransformerStep):
    """Clusterize each feature into a number of clusters

    Parameters
    ----------
    n_clusters : int
        Number of clusters to obtain per cluster
    """

    def __init__(self, n_clusters: int = 5, name: Optional[str] = None):
        super().__init__(name)
        self.clusters = {}
        self.n_clusters = n_clusters

    def partial_fit(self, X):
        if len(self.clusters) == 0:
            for c in X.columns:
                self.clusters[c] = MiniBatchKMeans(n_clusters=self.n_clusters)

        for c in X.columns:
            self.clusters[c].partial_fit(np.atleast_2d(X[c]).T)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Transform the input dataframe

        Parameters
        ----------
        X : pd.DataFrame
            Input life


        Returns
        -------
        pd.DataFrame
            A new DataFrame with the same index as the input.
            Each feature is replaced with the clusters of each point
        """
        X = X.copy()
        for c in X.columns:
            X[c] = self.clusters[c].cluster_centers_[
                self.clusters[c].predict(np.atleast_2d(X[c]).T)
            ]
        return X


class MultiDimensionalKMeans(TransformerStep):
    """Clusterize data points and replace each feature with the centroid feature its belong

    Parameters
    ----------
    n_clusters : int, optional
        Number of clusters to obtain by default 5
    name : Optional[str], optional
        Name of the step, by default None
    """

    def __init__(self, n_clusters: int = 5, name: Optional[str] = None):
        super().__init__(name)
        self.n_clusters = n_clusters
        self.clusters = MiniBatchKMeans(n_clusters=self.n_clusters)

    def partial_fit(self, X):
        self.clusters.partial_fit(X)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Transform the input life with the centroid information

        Parameters
        ----------
        X : pd.DataFrame
            Input life


        Returns
        -------
        pd.DataFrame
            A new DataFrame in which each point was replaced by the
            centroid its belong
        """

        X = X.copy()
        X[:] = self.clusters.cluster_centers_[self.clusters.predict(X)]
        return X


class KurtogramBandPassFiltering(TransformerStep):
    """[summary]

    Parameters
    ----------
    numtaps : int
        Length of the filter.
    sampling_rate : Optional[Union[float, str]], optional
        The sampling rate. If it is missing 1 is the default
        The column name of the dataframe with the sampling rate
    kurtogram_levels: Optional[int]
        The number of levels of the kurtogram.
        If it is missing  np.log2(N) - 7  levels will be used. Default None
    """
    def __init__(self, numtaps:int,sampling_rate: Optional[Union[float, str]] = None, kurtogram_levels:Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.sampling_rate = sampling_rate
        self.numtaps = numtaps

        if self.sampling_rate is None:
            self.sampling_rate = 1.0

        self.kurtogram_levels = kurtogram_levels


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        
        X_new = X.copy()
        if isinstance(self.sampling_rate, str):                
            fs = X[self.sampling_rate].iloc[0]
            X_new = X_new.loc[:, X_new.columns != self.sampling_rate]
        else:
            fs = self.sampling_rate 
        N = X_new.shape[0]
        if self.kurtogram_levels is None:
            kurtogram_levels = int(np.log2(N) - 7)
        else:
            kurtogram_levels = self.kurtogram_levels
        for c in X_new.columns:            
            
            _, _, _, BW, fc = fast_kurtogram(X[c].values, fs, kurtogram_levels)     
            lw = max(fc - BW/2, 0.0001)
            up = min(fc + BW/2, (fs/2.1))
            taps_hamming = firwin(
                self.numtaps,
                [lw, up],
                pass_zero=False,
                window="hamming",
                scale=True,
                fs=fs,
            )
            X_new[c] = lfilter(taps_hamming, fs, X[c].values)
        return X_new
