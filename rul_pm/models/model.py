import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from rul_pm.iterators.batcher import get_batcher
from rul_pm.store.store import store
from rul_pm.transformation.transformers import Transformer

logger = logging.getLogger(__name__)


def json_to_str(elem):
    if callable(elem):
        return elem.__name__
    elif isinstance(elem, np.ndarray):
        # return elem.tolist()
        return []
    else:
        return str(elem)


class TrainableModelInterface:
    def fit(self, ds):
        raise NotImplementedError

    def predict(self, df):
        raise NotImplementedError


class TrainableModel(TrainableModelInterface):
    """
    Base Model class to fit and predict

    Parameters
    -----------
    window: int
            Lookback window size
    batch_size: int
                Batch size
    step: Union[int, Tuple[str, int]]
          Stride
    transformer : Transformer
                  Transformer to be applied on the dataset
    shuffle: Union[bool, str]
             Check rul_pm.iterators.iterator

    patience: int. Default 4
              Patiente of early stopping

    cache_size: int. Default 30
                LRU Cache size of the iterator

    evenly_spaced_points: int
                Determine wether the window should include points in wich
                the RUL does not have gaps larger than the parameter

    sample_weight: str. Default = 'equal'
                Choose the weight of each sample. Possible values are
                'equal', 'proportional_to_length'.
                If 'equal' is choosed, each sample weights 1,
                if 'proportional_to_length' is choosed, each sample
                weight 1 / life length
    """

    def __init__(self,
                 window: int = 50,
                 batch_size: int = 256,
                 step: Union[int, Tuple[str, int]] = 1,
                 transformer: Transformer = None,
                 shuffle: Union[bool, str] = 'all',
                 patience: int = 4,
                 output_size: int = 1,
                 cache_size: int = 30,
                 evenly_spaced_points: Optional[int] = None,
                 sample_weight: str = 'equal',
                 add_last: bool = True):

        self.window = window
        self.batch_size = batch_size
        self.step = step
        self.transformer = transformer
        self.shuffle = shuffle
        self.patience = patience
        self.model_filename_ = None
        self._model_filepath = None
        self.cache_size = cache_size
        self._model = None
        self.output_size = output_size
        self.evenly_spaced_points = evenly_spaced_points
        self.sample_weight = sample_weight
        self.add_last = add_last

    @property
    def computed_step(self):
        if isinstance(self.step, int):
            return self.step
        elif isinstance(self.step, tuple):
            if self.step[0] == 'auto':
                return int(self.window / self.step[1])
        raise ValueError('Invalid step parameter')

    @property
    def name(self):
        return type(self).__name__

    def get_params(self, deep=False):
        return {
            'window': self.window,
            'batch_size': self.batch_size,
            'step': self.step,
            'shuffle': self.shuffle,
            'patience': self.patience,
            'transformer': self.transformer
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    @property
    def model_filepath(self):
        if self._model_filepath is None:
            self._model_filepath = (
                store.model_filename(self) +
                store.keras_extension())

        return self._model_filepath

    def true_values(self, dataset, step=None, transformer=None):
        step = self.step if step is None else step
        batcher = get_batcher(dataset,
                              self.window,
                              2048,
                              transformer if transformer is not None else self.transformer,
                              step,
                              shuffle=False,
                              add_last=self.add_last)
        batcher.restart_at_end = False
        trues = []
        for _, y, _ in batcher:
            trues.extend(y)
        return np.concatenate(trues)

    @property
    def n_features(self):
        return self.transformer.n_features

    def fit(self, train_dataset, validation_dataset, *args, **kwargs):
        raise NotImplementedError

    def predict(self, df):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    @property
    def input_shape(self):
        return (self.window, self.transformer.n_features)

    @property
    def model(self):
        if self._model is None:
            self._model = self.build_model()
        return self._model

    def reset(self):
        pass

    def _create_batchers(self, train_dataset, validation_dataset):
        logger.debug('Creating batchers')
        train_batcher = get_batcher(train_dataset,
                                    self.window,
                                    self.batch_size,
                                    self.transformer,
                                    self.computed_step,
                                    shuffle=self.shuffle,
                                    output_size=self.output_size,
                                    cache_size=self.cache_size,
                                    evenly_spaced_points=self.evenly_spaced_points,
                                    sample_weight=self.sample_weight,
                                    add_last=self.add_last)

        val_batcher = get_batcher(validation_dataset,
                                  self.window,
                                  self.batch_size,
                                  self.transformer,
                                  1,
                                  shuffle=False,
                                  output_size=self.output_size,
                                  cache_size=self.cache_size,
                                  restart_at_end=False,
                                  add_last=self.add_last)
        return train_batcher, val_batcher
