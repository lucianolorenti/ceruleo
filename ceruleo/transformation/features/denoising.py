from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cluster import MiniBatchKMeans
from ceruleo.transformation import TransformerStep


class SavitzkyGolayTransformer(TransformerStep):
    """Filter each feature using LOESS

    Parameters:    
        window: window size of the filter
        order:  Order of the filter, by default 2
        name: Step name
    """

    def __init__(self, window: int, order: int = 2, name: Optional[str] = None):

        super().__init__(name=name)
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

    def __init__(
        self,
        window: int,
        center=True,
        min_periods: int = 15,
        name: Optional[str] = None,
    ):

        super().__init__(name=name)
        self.window = window
        self.min_periods = min_periods
        self.center = center

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X.rolling(
            self.window, min_periods=self.min_periods, center=self.center
        ).mean(skip_na=True)


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
        super().__init__(name=name)
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
        super().__init__(name=name)
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
        super().__init__(name=name)
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


class EWMAFilter(TransformerStep):
    """Filter each feature using EWM

    Parameters
    ----------
    window : int
        Size of  window
    name : Optional[str], optional
        Name of the step, by default None
    """

    def __init__(self, span: float, name: Optional[str] = None):
        super().__init__(name=name)
        self.span = span

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X.ewm(span=self.span).mean(skip_na=True)


class GaussianFilter(TransformerStep):
    """Apply a gaussian filter

    Parameters
    ----------
    window_size : int
        Size of the gaussian filter
    std : float
        Standard deviation of the filter
    min_points : int, optional
        Minimun nomber of points of the rolling window, by default 1
    center : bool, optional
        Wether the guassian window should be centered, by default False
    """

    def __init__(
        self,
        window_size: int,
        std: float,
        min_points: int = 1,
        center: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.std = std
        self.center = center
        self.min_points = min_points

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.rolling(
            window=self.window_size,
            win_type="gaussian",
            center=self.center,
            min_periods=self.min_points,
        ).mean(std=self.std)
