from functools import partial
from typing import List, Optional

import numpy as np
import pandas as pd
from ceruleo.transformation import TransformerStep
from pandas.api.types import is_timedelta64_dtype
from scipy.stats import norm


class TargetToClasses(TransformerStep):
    """Transform the RUL values into a discrete set of classes

    Parameters
        bins: Bins limits
    """

    def __init__(self, bins: List[float]):
        super().__init__()
        self.bins = bins

    def transform(self, X):
        X_new = pd.DataFrame(index=X.index)
        X_new["RUL_class"] = np.digitize(X.iloc[:, 0].values, self.bins, right=False)
        return X_new


class PicewiseRUL(TransformerStep):
    """Transform the RUL clipping the maximum value

    Parameters

        max_life:  Maximum threshold for clipping the RUL values
        name:  Name of the step, by default None
    """

    def __init__(self, *, max_life: float = np.inf, name: Optional[str] = None):
        super().__init__(name=name)
        self.max_life = max_life

    def transform(self, X: pd.DataFrame):
        return X.clip(0, self.max_life)


class PicewiseRULQuantile(PicewiseRUL):
    """Clip the RUL  values using a quantile value as threshold

    Parameters

        quantile: Value between 0 and 1 to compute the RUL threshold
        name: Name of the step
    """

    def __init__(self, quantile: float, name: Optional[str] = None):
        super().__init__(name)
        self.quantile = quantile

    def fit(self, X, y=None):
        self.max_life = np.quantile(X, self.quantile)
        return self


class RULBinarizer(TransformerStep):
    """Convert the RUL target into a binary vector

    Useful for determining point of failures.


    Parameters
        t: Starting point of failure
        Every sample with a RUL greater than t will be 1 and the rest 0
    """

    def __init__(self, t: float, **kwargs):
        super().__init__(**kwargs)
        self.t = t

    def transform(self, X):
        return (X > self.t).astype("int")
