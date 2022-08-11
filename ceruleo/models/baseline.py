from typing import Optional, Union

import numpy as np
from ceruleo.dataset.ts_dataset import AbstractLivesDataset
from ceruleo.dataset.utils import iterate_over_target
from ceruleo.results.results import FittedLife

from ceruleo.dataset.transformed import TransformedDataset


class BaselineModel:
    """Predict the RUL using the mean of the median value of the duration
       of the dataset

    Parameters:

        mode: Method for computing the duration of the dataset
              Possible values are: 'mean' and 'median'
    """

    def __init__(self, mode: str = "mean", RUL_threshold: Optional[float] = None):
        self.mode = mode
        self.RUL_threshold = RUL_threshold

    def fit(self, ds: Union[TransformedDataset, AbstractLivesDataset]):
        """Compute the mean or median RUL using the given dataset

        Parameters:
            ds:  Dataset from which obtain the true RUL
        """
        true = []
        for y in iterate_over_target(ds):
            y = y
            degrading_start, time = FittedLife.compute_time_feature(
                y, self.RUL_threshold
            )

            true.append(y.iloc[0] + time[degrading_start])

        if self.mode == "mean":
            self.fitted_RUL = np.mean(true)
        elif self.mode == "median":
            self.fitted_RUL = np.median(true)

    def predict(self, ds: TransformedDataset):
        """Predict the whole life using the fitted values

        Parameters:
        
            ds: Dataset iterator from which obtain the true RUL

        Returns:
        
            d: Predicted target
        """
        output = []
        for y in iterate_over_target(ds):
            _, time = FittedLife.compute_time_feature(y, self.RUL_threshold)
            y_pred = np.clip(self.fitted_RUL - time, 0, self.fitted_RUL)
            output.append(y_pred)
        return np.concatenate(output)


class FixedValueBaselineModel:
    """A model that predicts always  the same duration for each run-to-failure cycle

    Parameters:

        value: Fixed RUL
    """

    def __init__(self, *, value: float):
        self.value = value

    def fit(self, *args):
        return self

    def predict(self, ds: TransformedDataset, RUL_threshold: Optional[float] = None):
        """Predict the whole life using the fixed values

        Parameters:

            ds: Dataset iterator from which obtain the true RUL

        Returns:

            true_RUL: Predicted target
        """
        output = []
        for y in iterate_over_target(ds):
            _, time = FittedLife.compute_time_feature(y, RUL_threshold)
            y_pred = np.clip(self.value - time, 0, self.value)
            output.append(y_pred)
        return np.concatenate(output)
