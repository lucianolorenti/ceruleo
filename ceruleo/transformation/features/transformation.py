from typing import Optional


import pandas as pd
from ceruleo.transformation import TransformerStep
from ceruleo.transformation.features.tdigest import TDigest
import numpy as np
from scipy.signal import find_peaks


class MeanCentering(TransformerStep):
    """Center the data with respect to the mean"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N = 0
        self.sum = None

    def fit(self, X: pd.DataFrame, y=None):
        """
        Compute the mean of the dataset

        Parameters:
            X: The input dataset
        """
        self.mean = X.mean()
        return self

    def partial_fit(self, X: pd.DataFrame, y=None):
        """
        Compute incrementally the mean of the dataset

        Parameters:
            X: The input life
        """
        if self.sum is None:
            self.sum = X.sum()
        else:
            self.sum += X.sum()

        self.N += X.shape[0]
        self.mean = self.sum / self.N
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Center the input life

        Parameters:
            X: The input life
            
        Returns:
            A new DataFrame with the same index as the input with the data centered with respect to the mean of the fiited dataset
        """
        return X - self.mean


class MedianCentering(TransformerStep):
    """Center the data with respect to the median"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tdigest_dict = None
        self.median = None

    def fit(self, X: pd.DataFrame, y=None):
        """
        Compute the median of the dataset

        Parameters:
            X: The input dataset
        """
        self.median = X.median()
        return self

    def partial_fit(self, X: pd.DataFrame, y=None):
        """
        Compute incrementally the median of the dataset

        Parameters:
            X: The input life
        """
        if X.shape[0] < 2:
            return self

        if self.tdigest_dict is None:
            self.tdigest_dict = {c: TDigest(100) for c in X.columns}
        for c in X.columns:
            self.tdigest_dict[c] = self.tdigest_dict[c].merge_unsorted(X[c].values)

        self.median = pd.Series(
            {
                c: self.tdigest_dict[c].estimate_quantile(0.5)
                for c in self.tdigest_dict.keys()
            }
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Center the input life

        Parameters:
            X: The input life
            
        Returns:
            A new DataFrame with the same index as the input with the data centered with respect to the mean of the fiited dataset
        """
        return X - self.median


class Square(TransformerStep):
    """Compute the square of the values of each feature"""

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input life with the square of the values

        Parameters:
            X: The input life

        Returns:
            A new dataframe with the same index as the input with the square of the values
        """
        return X.pow(2)


class Sqrt(TransformerStep):
    """Compute the sqrt of the values of each feature"""

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input life with the sqrt of the values

        Parameters:
            X: The input life
            
        Returns:
            A new dataframe with the same index as the input with the sqrt of the values
        """
        return X.pow(1.0 / 2)


class Scale(TransformerStep):
    """
    Scale each feature by a given vaulue

    Parameters:
        scale_factor: Scale factor to apply to each feature
        name: Name of the step, by default None
    """

    def __init__(self, *, scale_factor: float, name: Optional[str] = None):
        super().__init__(name=name)
        self.scale_factor = scale_factor

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return the scaled life

        Parameters:
            X: The input life
            
        Returns:
            Return a new DataFrame with the same index as the input with the scaled features
        """
        return X * self.scale_factor


class ExpandingCentering(TransformerStep):
    """
    Center the life using an expanding window
    """

    #.. raw:: html
    #<p>Formula: \(X - X.expanding().mean())</p>

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the live centering it using an expanding window

        Parameters:
            X: The input life
            
        Returns:
            Return a new DataFrame with the same index as the input with the data centered
        """
        return X - X.expanding().mean()


class RollingCentering(TransformerStep):
    """
    Center the life using an rolling window

    """



    def __init__(self, window: int, min_points: int, name: Optional[str] = None):
        super().__init__(name=name)
        self.window = window
        self.min_points = min_points

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the live centering it using a rolling window

        Parameters:
            X: The input life
            
        Returns:
            Return a new DataFrame with the same index as the input with the data centered
        """
        return X - X.rolling(window=self.window, min_periods=self.min_points).mean()


class ExpandingNormalization(TransformerStep):
    """Normalize the life features using an expanding window
    """

    #.. highlight:: python
    #.. code-block:: python

    #(X - X.expanding().mean()) / (X.expanding().std())


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the live normalized it using an expanding window

        Parameters:
            X: The input life
            
        Returns:
            Return a new DataFrame with the same index as the input with the data normalized
        """
        return (X - X.expanding().mean()) / (X.expanding().std())


class Accumulate(TransformerStep):
    """
    Compute the accumulated sum of each feature.

    This is useful to compute the count of binary features.

    Parameters:
        normalize: Weather to apply the normalization or not, by default False
    """ 

    def __init__(self, *, normalize: bool = False,  name: Optional[str] = None):
        super().__init__(name=name)
        self.normalize = normalize

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input life computing the cumulated sum

        Parameters:
            X: The Input life
            
        Returns:
            Return a new DataFrame with the same index as the input with the cumulated sum of the features
        """
        X1 = X.cumsum()
        if self.normalize:
            return X1 / X1.abs().apply(np.sqrt)
        else:
            return X1


class Diff(TransformerStep):
    """Compute the 1 step difference of each feature."""

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input life computing the 1 step difference

        Parameters:
            X: The Input life

        Returns:
            Return a new DataFrame with the same index as the input with the 1 step difference of the features
        """
        return X.diff()


class StringConcatenate(TransformerStep):
    """
    Return a new DataFrame with a single column with the concatenation of the values of each row. The method works only on strings and the values are separated by -"""

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input life computing the 1 step difference

        Parameters:
            X: The Input life
            
        Returns:
            Return a new DataFrame with the same index as the input with the difference of the features
        """
        new_X = pd.DataFrame(index=X.index)
        new_X["concatenation"] = X.agg("-".join, axis=1)
        return new_X


class Apply(TransformerStep):
    """Apply the input function element-wise"""

    def __init__(self, *, fun, name: Optional[str] = None):
        super().__init__(name=name)
        self.fun = fun

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input life applying the input function to each element

        Parameters
            X: The Input life
            
        Returns:
            Return a new DataFrame with results of the function application to each element.
        """
        return X.apply(self.fun)


class Clip(TransformerStep):
    """
    Clip values onto a predefined range

    Parameters:
        lower: The lower value
        upper: The Upper value
    """

    def __init__(self, *, lower: float, upper: float, **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input life clipping the values onto a predefined range

        Parameters:
            X: The Input life
        
        Returns:
            Return a new DataFrame with the same index as the input with the clipped values
        """
        return X.clip(self.lower, self.upper)


class SubstractLinebase(TransformerStep):
    """Subtract the values in the first row from all the rows in the input life"""

    def __init__(self, *args):
        super().__init__(*args)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """ 
        Apply the transformation to the input life

        Parameters:
            X: The Input life

        Returns:
            Return a new DataFrame with the same index as the input with the subtraction of the first row
        """
        return X - X.iloc[0, :]


class Peaks(TransformerStep):
    """Find Peaks in the input life. Return a new DataFrame with the same index as the input with 1 in the position of the peaks and 0 otherwise"""

    distance: float

    def __init__(self, *, distance:float, name : Optional[str] = None):
        super().__init__(name=name)
        self.distance = distance

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """ 
        Detect the peaks in the input life

        Parameters:
            X: The Input life
        
        Returns:
            Return a new DataFrame with the same index as the input with the peaks marked as 1 and 0 otherwise. 
        """
        new_X = pd.DataFrame(np.zeros(X.shape), index=X.index, columns=X.columns)
        for i, c in enumerate(X.columns):
            peaks_positions, _ = find_peaks(X[c].values, distance=self.distance)
            new_X.iloc[peaks_positions, i] = 1

        return new_X
