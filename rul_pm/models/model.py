import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from rul_pm.transformation.pipeline import LivesPipeline
from rul_pm.utils import progress_bar

logger = logging.getLogger(__name__)


def json_to_str(elem):
    if callable(elem):
        return elem.__name__
    elif isinstance(elem, np.ndarray):
        # return elem.tolist()
        return []
    else:
        return str(elem)


class TrainableModel:
    """Base class of a trainable model"""

    @property
    def name(self):
        """Name of the model

        Returns
        -------
            str: The class name as string
        """
        return type(self).__name__

    def get_params(self, deep: Optional[bool] = False):
        """Obtain the model params

        Parameters
        ----------
        deep: Optiona[bool]. Defaults to False.


        Returns
        ------.
        dict: Dictionary with the model parameters
        """
        return {}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(
        self,
        train_dataset: Iterable,
        *args,
        validation_dataset: Optional[Iterable] = None,
        **kwargs
    ):
        """Abstract method for fitting the model

        Parameters
        ----------
        train_dataset: Iterable
                       Train dataset iterator
        validation_dataset: Optional[Iterable]. Defaults to None.
                            Validation dataset iterator

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def predict(self, df: Iterable):
        """Abstract predict method

        Parameters
        ----------
        df: Iterable
            Iterator of the dataset
        """
        raise NotImplementedError

    def build_model(self):
        """Abstract method for building the model"""
        raise NotImplementedError

    @property
    def model(self):
        if self._model is None:
            self._model = self.build_model()
        return self._model

    def reset(self):
        pass

    def save(self, file: Path):
        raise NotImplementedError
