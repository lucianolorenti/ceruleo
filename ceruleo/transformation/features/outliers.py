from typing import Optional

import numpy as np
import pandas as pd
from ceruleo.transformation import TransformerStep
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from ceruleo.transformation.features.tdigest import TDigest
from ceruleo.transformation.utils import QuantileEstimator
from sklearn.ensemble import IsolationForest


class IQROutlierRemover(TransformerStep):
    """Remove values outside (Q1 - margin*IQR, Q2 + margin*IQR)

    If clip is True the values will be clipped between the range,
    otherwise the values are going to be replaced by inf and -inf



    Parameters:
        lower_quantile: Lower quantile threshold for the non-anomalous values
        upper_quantile: Upper quantile threshold for the non-anomalous values
        margin: How many times the IQR gets multiplied
        proportion_to_sample: If you want to compute the quantiles in an smaller proportion of data
            you can specify it
        clip: Wether to clip the values outside the range.

    """

    def __init__(
        self,
        lower_quantile: float = 0.25,
        upper_quantile: float = 0.75,
        margin=1.5,
        proportion_to_sample=1.0,
        clip: bool = False,
        name: Optional[str] = None,
        prefer_partial_fit: bool = False,
    ):

        super().__init__(name=name, prefer_partial_fit=prefer_partial_fit)
        self.margin = margin
        self.proportion_to_sample = proportion_to_sample
        self.tdigest_dict = None
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.clip = clip

    def partial_fit(self, X):
        if X.shape[0] == 1:
            return self
        if self.proportion_to_sample < 1:
            sampled_points = np.random.choice(
                X.shape[0], int(X.shape[0] * self.proportion_to_sample), replace=False
            )
            X = X.iloc[sampled_points, :]
        if self.tdigest_dict is None:
            self.tdigest_dict = {c: TDigest(100) for c in X.columns}
        for c in X.columns:
            self.tdigest_dict[c] = self.tdigest_dict[c].merge_unsorted(X[c].values)

        self.Q1 = {
            c: self.tdigest_dict[c].estimate_quantile(self.lower_quantile)
            for c in self.tdigest_dict.keys()
        }

        self.Q3 = {
            c: self.tdigest_dict[c].estimate_quantile(self.upper_quantile)
            for c in self.tdigest_dict.keys()
        }

        self.IQR = {c: self.Q3[c] - self.Q1[c] for c in self.Q1.keys()}
        return self

    def fit(self, X):
        if self.proportion_to_sample < 1:
            sampled_points = np.random.choice(
                X.shape[0], int(X.shape[0] * self.proportion_to_sample), replace=False
            )
            X = X.iloc[sampled_points, :]
        self.Q1 = X.quantile(self.lower_quantile)
        self.Q3 = X.quantile(self.upper_quantile)
        self.IQR = (self.Q3 - self.Q1).to_dict()
        self.Q1 = self.Q1.to_dict()
        self.Q3 = self.Q3.to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        check_is_fitted(self, "Q1")
        check_is_fitted(self, "Q3")
        check_is_fitted(self, "IQR")
        for c in X.columns:
            min_value = self.Q1[c] - self.margin * self.IQR[c]
            mask = X[c] < min_value
            if not self.clip:
                X.loc[mask, c] = -np.inf
            else:
                X.loc[mask, c] = min_value
            max_value = self.Q3[c] + self.margin * self.IQR[c]
            mask = X[c] > (max_value)
            if not self.clip:
                X.loc[mask, c] = np.inf
            else:
                X.loc[mask, c] = max_value
        return X

    def description(self):
        name = super().description()
        data = []
        for k in self.Q1.keys():
            data.append((k, {"Q1": self.Q1[k], "Q3": self.Q3[k], "IQR": self.IQR[k]}))
        return (name, data)


class BeyondQuartileOutlierRemover(TransformerStep):
    """Remove values outside (Q1, Q3)

    If clip is True the values will be clipped between the range,
    otherwise the values are going to be replaced by inf and -inf



    Parameters:
        lower_quantile:  Lower quantile threshold for the non-anomalous values
        upper_quantile: Upper quantile threshold for the non-anomalous values
        clip: Wether to clip the values outside the range.

    """

    def __init__(
        self,
        lower_quantile: float = 0.25,
        upper_quantile: float = 0.75,
        subsample: float = 1.0,
        clip: bool = False,
        name: Optional[str] = None,
        prefer_partial_fit:bool=False
    ):

        super().__init__(name=name, prefer_partial_fit=prefer_partial_fit)
        self.tdigest_dict = None
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.clip = clip
        self.subsample= subsample
        self.Q1 = None
        self.Q3 = None
        self.quantile_estimator = None

    def partial_fit(self, X):
        if X.shape[0] == 1:
            return self
        if self.quantile_estimator is None:
            self.quantile_estimator = QuantileEstimator(
               tdigest_size=100, subsample=self.subsample
            )

        self.quantile_estimator.update(X.select_dtypes(include="number"))
        return self

    def fit(self, X):
        if self.subsample < 1:

            sampled_points = np.random.choice(
                X.shape[0], int(X.shape[0] * self.subsample), replace=False
            )
            X = X.iloc[sampled_points, :]
        self.Q1 = X.quantile(self.lower_quantile)
        self.Q3 = X.quantile(self.upper_quantile)

        return self

    def transform(self, X):

        if self.Q1 is None:
            self.Q1 = self.quantile_estimator.estimate_quantile(self.lower_quantile)
            self.Q3 = self.quantile_estimator.estimate_quantile(self.upper_quantile)

        new_X = X.copy()
        

        if self.clip:
            new_X.clip(lower=self.Q1, upper=self.Q3, inplace=True, axis=1)
        else:
            new_X[new_X < self.Q1] = -np.inf
            new_X[new_X > self.Q3] = np.inf            
        return new_X

    def description(self):
        name = super().description()
        data = []
        for k in self.Q1.keys():
            data.append((k, {"Q1": self.Q1[k], "Q3": self.Q3[k], "IQR": self.IQR[k]}))
        return (name, data)


class ZScoreOutlierRemover(TransformerStep):
    """
    X = np.random.rand(500, 5) * np.random.randn(500, 5) * 15
    imput = ZScoreImputer(1.5)
    imput.fit(X)
    X_t = imput.transform(X)
    """

    def __init__(
        self,
        *,
        number_of_std_allowed,
        name: str = None,
        prefer_partial_fit: bool = False,
    ):
        super().__init__(name=name, prefer_partial_fit=prefer_partial_fit)
        self.number_of_std_allowed = number_of_std_allowed
        self.scaler = StandardScaler()

    def fit(self, X):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        X_new = self.scaler.transform(X)
        X_new[np.abs(X_new) > self.number_of_std_allowed] = np.nan
        return pd.DataFrame(X_new, columns=X.columns, index=X.index)


class EWMAOutOfRange(TransformerStep):
    """
    Compute the EWMA limits  and mark as NaN points outside UCL and LCL
    """

    def __init__(
        self,
        *,
        lambda_=0.5,
        return_mask: bool = False,
        name: Optional[str] = None,
        prefer_partial_fit: bool = False,
    ):
        super().__init__(name=name, prefer_partial_fit=prefer_partial_fit)
        self.lambda_ = lambda_
        self.UCL = None
        self.LCL = None
        self.columns = None
        self.return_mask = return_mask

    def partial_fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.columns.values
        else:
            self.columns = [c for c in self.columns if c in X.columns]
        if self.LCL is not None:
            self.LCL = self.LCL.loc[self.columns].copy()
            self.UCL = self.UCL.loc[self.columns].copy()
        LCL, UCL = self._compute_limits(X[self.columns].copy())
        self.LCL = np.minimum(LCL, self.LCL) if self.LCL is not None else LCL
        self.UCL = np.maximum(UCL, self.UCL) if self.UCL is not None else UCL
        return self

    def _compute_limits(self, X):

        mean = np.nanmean(X, axis=0)
        s = np.sqrt(self.lambda_ / (2 - self.lambda_)) * np.nanstd(X, axis=0)
        UCL = mean + 3 * s
        LCL = mean - 3 * s
        return (pd.Series(LCL, index=self.columns), pd.Series(UCL, index=self.columns))

    def fit(self, X, y=None):
        self.columns = X.columns
        LCL, UCL = self._compute_limits(X)
        self.LCL = LCL
        self.UCL = UCL
        return self

    def transform(self, X):
        mask = (X[self.columns] < (self.LCL)) | (X[self.columns] > (self.UCL))
        if self.return_mask:
            return mask.astype("int")
        else:
            X = X.copy()
            X[mask] = np.nan
            return X


class RollingMeanOutlierRemover(TransformerStep):
    def __init__(
        self,
        *,
        window: int = 15,
        lambda_: float = 3,
        return_mask: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.window = window
        self.lambda_ = lambda_
        self.return_mask = return_mask

    def transform(self, X):
        r = X.rolling(self.window, min_periods=1)
        std = r.quantile(0.75) -  r.quantile(0.25)
        upper = r.median() + (self.lambda_ * std)
        lower = r.median() - (self.lambda_ * std)
        mask = (X > upper) | (X < lower)
        if self.return_mask:
            return mask.astype("int")
        else:
            X = X.copy()
            X[(X > upper)] = np.minimum(upper.values, X) 
            X[(X < upper)] = np.maximum(lower.values, X) 

            #X[mask] = np.nan
            return X


class IsolationForestOutlierRemover(TransformerStep):
    def __init__(self, *, n_estimators=100, **kwargs):
        super().__init__(prefer_partial_fit=False, **kwargs)
        self.n_estimators = n_estimators
        self.forests = {}

    def fit(self, X: pd.DataFrame):
        for c in X.columns:
            self.forests[c] = IsolationForest(n_estimators=self.n_estimators).fit(X[c].values.reshape(-1, 1) )
        return self

    def transform(self, X: pd.DataFrame):
        X_new = X.copy()
        for c in X.columns:
            r = self.forests[c].predict(X[c].values.reshape(-1, 1) )
            X_new.loc[r == -1, c] = np.nan
        return X_new
