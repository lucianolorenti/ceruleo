from typing import Optional


import numpy as np

from rul_pm.transformation.features.extraction import compute, roll_matrix, stats_order
from rul_pm.transformation.transformerstep import TransformerStep

from sklearn.cluster import MiniBatchKMeans


class MeanFilter(TransformerStep):
    def __init__(self, window: int, min_periods: int = 15, name: Optional[str] = None):
        super().__init__(name)
        self.window = window
        self.min_periods = min_periods

    def transform(self, X, y=None):
        return X.rolling(self.window, min_periods=self.min_periods).mean(skip_na=True)


class OneDimensionalKMeans(TransformerStep):
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

    def transform(self, X, y=None):
        X = X.copy()
        for c in X.columns:
            X[c] = self.clusters[c].cluster_centers_[
                self.clusters[c].predict(np.atleast_2d(X[c]).T)
            ]
        return X


class MultiDimensionalKMeans(TransformerStep):
    def __init__(self, n_clusters: int = 5, name: Optional[str] = None):
        super().__init__(name)
        self.clusters = MiniBatchKMeans(n_clusters=self.n_clusters)
        self.n_clusters = n_clusters

    def partial_fit(self, X):

        self.clusters.partial_fit(X)
        return self

    def transform(self, X, y=None):
        X = X.copy()
        print(self.clusters.cluster_centers_[self.clusters.predict(X)])
        X[:, :] = self.clusters.cluster_centers_[self.clusters.predict(X)]
        return X