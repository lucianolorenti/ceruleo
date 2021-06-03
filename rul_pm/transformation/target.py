from typing import Optional

import numpy as np
import pandas as pd
from rul_pm.transformation.transformerstep import TransformerStep
from rul_pm.transformation.utils.utils import sliding_window_view

class TargetToClasses(TransformerStep):
    def __init__(self, bins):
        super().__init__()
        self.bins = bins

    def transform(self, X):
        X_new = pd.DataFrame(index=X.index)
        X_new["RUL_class"] = np.digitize(X.iloc[:, 0].values, self.bins, right=False)
        return X_new


class HealthPercentage(TransformerStep):
    def transform(self, X):
        return (X / X.iloc[0, 0]) * 100


class PicewiseRUL(TransformerStep):
    """Transform the RUL clipping the maximum value

    Parameters
    ----------
    max_life : float, optional
        Maximum threshold for clipping the RUL values, by default np.inf
    name : Optional[str], optional
        Name of the step, by default None    """
    def __init__(self, max_life: float = np.inf, name: Optional[str] = None):
        super().__init__(name)
        self.max_life = max_life

    def transform(self, X):
        """Clip the maximum value of the true RUL

        Parameters
        ----------
        X : np.array
            Vector with the true labels

        Returns
        -------
        np.array
            Cliiped RUL values
        """
        return np.clip(X, 0, self.max_life)


class PicewiseRULQuantile(PicewiseRUL):
    """Transform the RUL clipping the maximum value using a quantile value as threshold

    Parameters
    ----------
    quantile : float, optional
        Value between 0 and 1 to compute the RUL threshold
    name : Optional[str], optional
        Name of the step, by default None    """
    def __init__(self, quantile:float, name: Optional[str] = None):
        super().__init__(name)
        self.quantile = quantile

    def fit(self, X, y=None):
        """Clip the maximum value of the true RUL

        Parameters
        ----------
        X : np.array
            Vector with the true labels

        Returns
        -------
        np.array
            Cliiped RUL values
        """
        self.max_life = np.quantile(X, self.quantile)
        return self


class RULBinarizer(TransformerStep):
    def __init__(self, t:float, **kwargs):
        super().__init__(**kwargs)
        self.t = t       
    

    def transform(self, X):
        return (X < self.t).astype('int')



class TargetSplitIntoWindows(TransformerStep):
    """Split the target features of the lives in windows[summary]

    Parameters
    ----------
    window_size : int
        Size of each window
    stride : int
        Strides of the windows
    """
    def __init__(self, window_size:int, stride:int, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.stride = stride

    def transform(self, X:pd.DataFrame) -> pd.Series:
        """Transform the given life in a list of DataFrame, each element is a windowed vision of the signal

        Parameters
        ----------
        X : pd.DataFrame
            The input life

        Returns
        -------
        List[pd.DataFrame]
            The life data splitted in windows
        """
        indices = sliding_window_view(X.index.values, (self.window_size,), (self.stride, ))
        return pd.Series([X.loc[indices[w, :], :].mode().values[0][0] for w in range(indices.shape[0])])
