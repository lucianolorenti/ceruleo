from typing import Optional

import numpy as np
from rul_pm.dataset.lives_dataset import AbstractLivesDataset
from rul_pm.iterators.iterators import LifeDatasetIterator
from rul_pm.models.model import TrainableModel
from rul_pm.results.results import FittedLife


class BaselineModel(TrainableModel):
    """Predict the RUL using the mean of the median value of the duration
       of the dataset

    Parameters
    ----------
    mode: str
        Method for computing the duration of the dataset
        Possible values are: 'mean' and 'median'
    """
    def __init__(self, mode:str='mean'):
        self.mode = mode

    def fit(self, ds:LifeDatasetIterator):
        """Compute the mean or median RUL using the given dataset

        Parameters
        ----------
        ds : LifeDatasetIterator
            Dataset iterator from which obtain the true RUL
        """
        true = []
        for _, y, _ in ds:
            true.append(y[0])

        if self.mode == 'mean':
            self.fitted_RUL = np.mean(true)
        elif self.mode == 'median':
            self.fitted_RUL = np.median(true)

    def predict(self, ds: LifeDatasetIterator, RUL_threshold:Optional[float]=None):
        """Predict the whole life using the fitted values

        Parameters
        ----------
        ds : LifeDatasetIterator
            Dataset iterator from which obtain the true RUL

        Returns
        -------
        np.array
            Predicted target
        """
        output = []
        for _,y, _ in ds:
            _, time = FittedLife.compute_time_feature(y, RUL_threshold)
            y_pred = np.clip(
                self.fitted_RUL-time,  0, self.fitted_RUL)
            output.append(y_pred)
        return np.concatenate(output)




class FixedValueBaselineModel(TrainableModel):
    """[summary]

    Parameters
    ----------
    value: float
        Fixed RUL
    """
    def __init__(self, value:float):
        self.value = value

    def predict(self, ds: LifeDatasetIterator, RUL_threshold:Optional[float]=None):
        """Predict the whole life using the fixed values

        Parameters
        ----------
        ds : LifeDatasetIterator
            Dataset iterator from which obtain the true RUL

        Returns
        -------
        np.array
            Predicted target
        """
        output = []
        for _, y, _ in ds:
            time = FittedLife.compute_time_feature(y, RUL_threshold)
            y_pred = np.clip(
                self.value+time,  0, self.fitted_RUL)
            output.append(y_pred)
        return np.concatenate(output)
