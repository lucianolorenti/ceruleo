from typing import Optional

import numpy as np
import pandas as pd
from rul_pm.transformation.transformerstep import TransformerStep
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from tdigest import TDigest


class IQROutlierRemover(TransformerStep):
    """
    X = np.random.rand(500, 5) * np.random.randn(500, 5) * 15
    imput = IQRImputer(1.5)
    imput.fit(X)
    X_t = imput.transform(X)
    """
    def __init__(self,
                 margin=1.5,
                 proportion_to_sample=1.0,
                 name: Optional[str] = None):
        super().__init__(name)
        self.margin = margin
        self.proportion_to_sample = proportion_to_sample
        self.tdigest_dict = None

    def partial_fit(self, X):
        if self.proportion_to_sample < 1:
            sampled_points = np.random.choice(X.shape[0],
                                              int(X.shape[0] *
                                                  self.proportion_to_sample),
                                              replace=False)
            X = X.iloc[sampled_points, :]
        if self.tdigest_dict is None:
            self.tdigest_dict = {c: TDigest() for c in X.columns}
        for c in X.columns:
            self.tdigest_dict[c].batch_update(X[c].values)

        self.Q1 = {
            c: self.tdigest_dict[c].percentile(25)
            for c in self.tdigest_dict.keys()
        }
        
        self.Q3 = {
            c: self.tdigest_dict[c].percentile(75)
            for c in self.tdigest_dict.keys()
        }
        
        self.IQR = {c: self.Q3[c] - self.Q1[c] for c in self.Q1.keys()}

        return self

    def fit(self, X):
        if self.proportion_to_sample < 1:
            sampled_points = np.random.choice(X.shape[0],
                                              int(X.shape[0] *
                                                  self.proportion_to_sample),
                                              replace=False)
            X = X.iloc[sampled_points, :]
        self.Q1 = X.quantile(0.25)
        self.Q3 = X.quantile(0.75)
        self.IQR = (self.Q3 - self.Q1).to_dict()
        self.Q1 = self.Q1.to_dict()
        self.Q3 = self.Q3.to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        check_is_fitted(self, 'Q1')
        check_is_fitted(self, 'Q3')
        check_is_fitted(self, 'IQR')
        for c in X.columns:
            mask = ((X[c] < (self.Q1[c] - self.margin * self.IQR[c])) |
                    (X[c] > (self.Q3[c] + self.margin * self.IQR[c])))
            X.loc[mask, c] = np.nan
        return X


class ZScoreOutlierRemover(BaseEstimator, TransformerMixin):
    """
    X = np.random.rand(500, 5) * np.random.randn(500, 5) * 15
    imput = ZScoreImputer(1.5)
    imput.fit(X)
    X_t = imput.transform(X)
    """
    def __init__(self, number_of_std_allowed):
        self.number_of_std_allowed = number_of_std_allowed
        self.scaler = StandardScaler()

    def fit(self, X):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        X_new = self.scaler.transform(X)
        X_new[np.abs(X_new) > self.number_of_std_allowed] = np.nan
        return pd.DataFrame(X_new, columns=X.columns, index=X.index)


class EWMAOutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, lambda_=1.5):
        self.lambda_ = lambda_

    def fit(self, X, y=None):
        mean = np.nanmean(X, axis=0)
        s = np.sqrt(self.lambda_ / (2 - self.lambda_)) * np.nanstd(X, axis=0)
        self.UCL = mean + 3 * s
        self.LCL = mean - 3 * s
        return self

    def transform(self, X):
        mask = ((X < (self.LCL)) | (X > (self.UCL)))
        X[mask] = np.nan
        return pd.DataFrame(X, columns=X.columns, index=X.index)
