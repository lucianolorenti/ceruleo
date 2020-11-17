import hashlib
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Tuple, TypeVar, Union

import numpy as np
import tensorflow as tf
from rul_pm.iterators.batcher import Batcher, get_batcher
from rul_pm.iterators.iterators import WindowedDatasetIterator
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


class TrainableModel:
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
    models_path: Path
                 Location where the models are stored
    patience: int. Default 4
              Patiente of the Early stopping
    cache_size: int. Default 30
                LRU Cache size of the iterator
    """

    def __init__(self,
                 window: int,
                 batch_size: int,
                 step: Union[int, Tuple[str, int]],
                 transformer: Transformer,
                 shuffle: Union[bool, str],
                 models_path: Path,
                 patience: int = 4,
                 output_size: int = 1,
                 cache_size: int = 30):
        if isinstance(models_path, str):
            models_path = Path(models_path)

        self.window = window
        self.batch_size = batch_size
        self.step = step
        self.transformer = transformer
        self.shuffle = shuffle
        self.patience = patience
        self.models_path = models_path
        self.model_filename_ = None
        self._model_filepath = None
        self.cache_size = cache_size
        self._model = None
        self.output_size = output_size

    @property
    def computed_step(self):
        if isinstance(self.step, int):
            return self.step
        elif isinstance(self.step, tuple):
            if self.step[0] == 'auto':
                return int(self.window / self.step[1])
        raise ValueError('Invalid step parameter')

    @property
    def model_filename(self):
        if self.model_filename_ is None:
            hash_object = self._hash_parameters()
            self.model_filename_ = self.name + '_' + hash_object
        return self.model_filename_

    @property
    def name(self):
        raise NotImplementedError

    def _hash_parameters(self):
        return hashlib.md5(self._parameter_to_json()).hexdigest()

    def _parameter_to_json(self):
        return json.dumps(self.get_params(),
                          sort_keys=True,
                          default=json_to_str).encode('utf-8')

    def get_params(self, deep=False):
        return {
            'window': self.window,
            'batch_size': self.batch_size,
            'step': self.step,
            'shuffle': self.shuffle,
            'patience': self.patience,
            'transformer': self.transformer,
            'models_path': self.models_path
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _get_params_serializable(self):
        params = self.get_params()
        params['transformer'] = params['transformer'].description()
        return params

    @property
    def model_filepath(self):
        if self._model_filepath is None:
            self._model_filepath = str(self.models_path / self.model_filename)
        return self._model_filepath

    @property
    def results_filename(self):
        return self.model_filepath + 'results_.pkl'

    def load_results(self):
        with open(self.results_filename, 'rb') as infile:
            return pickle.load(infile)

    def _results(self):
        return {
            'parameters': self._get_params_serializable(),
            'model_file': self.model_filepath
        }

    def save_results(self):
        logger.info(f'Writing results {self.results_filename}')

        with open(self.results_filename, 'wb') as outfile:
            pickle.dump(self._results(), outfile)

    def true_values(self, dataset, step=None, transformer=None):
        step = self.step if step is None else step
        batcher = get_batcher(dataset,
                              self.window,
                              512,
                              transformer if transformer is not None else self.transformer,
                              step,
                              shuffle=False)
        batcher.restart_at_end = False
        trues = []
        for _, y in batcher:
            trues.extend(y)
        return trues

    @property
    def n_features(self):
        return self.transformer.n_features

    def fit(self, ds):
        raise NotImplementedError

    def predict(self, df):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    @property
    def model(self):
        if self._model is None:
            self._model = self.build_model()
        return self._model

    def reset(self):
        pass
