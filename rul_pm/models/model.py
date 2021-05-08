
import logging

import numpy as np
from rul_pm.store.store import store
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


class TrainableModelInterface:
    def fit(self, ds):
        raise NotImplementedError

    def predict(self, df):
        raise NotImplementedError


class TrainableModel(TrainableModelInterface):
    @property
    def name(self):
        return type(self).__name__

    def get_params(self, deep=False):
        return {
            'iterator': self.dataset_iterator.get_params(),
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def true_values(self, dataset_iterator):
        orig_transformer = dataset_iterator.transformer.transformerX
        dataset_iterator.transformer.transformerX = LivesPipeline(
            steps=[('empty', 'passthrough')])        
        d =  np.concatenate([y for _, y, _ in dataset_iterator])
        dataset_iterator.transformer.transformerX = orig_transformer
        return d

    def fit(self, train_dataset, validation_dataset, *args, **kwargs):
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



