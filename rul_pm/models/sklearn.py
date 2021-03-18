import numpy as np
import pandas as pd
from rul_pm.iterators.batcher import get_batcher
from rul_pm.models.model import TrainableModel
from tqdm.auto import tqdm

from sklearn.metrics import mean_squared_error


class SKLearnModel(TrainableModel):
    def __init__(self,
                 model,
                 **kwargs):
        super().__init__(
            **kwargs)
        self._model = model
        if not hasattr(self.model, 'fit'):
            raise ValueError('Model must allow to fit')

    def build_model(self):
        return self._model

    def fit(self, train_dataset, refit_transformer=True, **kwargs):
        if refit_transformer:
            self.transformer.fit(train_dataset)
        X, y, sample_weight = self.get_data(train_dataset)
        self.model.fit(X, y.ravel(), **kwargs)
        return self

    def predict(self, dataset):
        X, _, _ = self.get_data(dataset, shuffle=False)
        return self.model.predict(X)

    def get_params(self, deep):
        out = super().get_params(deep=deep)
        out['model'] = self.model
        if deep and hasattr(self.model, 'get_params'):
            for key, value in self.model.get_params(deep=True).items():
                out['model__%s' % key] = value
        return out

    def set_params(self, **params):
        ## TODO Check invalid parameters
        model_params = {}
        for name, value in params.items():
            if '__' in name:
                model_params[name.split('__')[1]] = value
        for name in model_params.keys():
            params.pop(f'model__{name}')

        super().set_params(**params)
        self.model.set_params(**model_params)
        return self



class BatchSKLearnModel(TrainableModel):
    def __init__(self,
                 model,
                 window,
                 batch_size,
                 step,
                 transformer,
                 shuffle,
                 cache_size=30):
        super().__init__(window,
                         batch_size,
                         step,
                         transformer,
                         shuffle,
                         cache_size=cache_size)
        self._model = model
        if not hasattr(self.model, 'partial_fit'):
            raise ValueError('Model must allow to partial_fit')

    def build_model(self):
        return self._model

    def fit(self, train_dataset, validation_dataset, n_epochs=15):
        train_batcher = get_batcher(train_dataset,
                                    self.window,
                                    self.batch_size,
                                    self.transformer,
                                    self.computed_step,
                                    shuffle=self.shuffle,
                                    cache_size=self.cache_size)

        train_batcher.restart_at_end = False
        for r in range(n_epochs):
            for X, y in tqdm(train_batcher):
                self.model.partial_fit(np.reshape(
                    X, (X.shape[0], X.shape[1]*X.shape[2])), y)
            y_true, y_pred = self.predict(validation_dataset)
            print(mean_squared_error(y_true, y_pred))

    def predict(self, dataset):
        y_true = []
        y_pred = []
        val_batcher = get_batcher(dataset,
                                  self.window,
                                  self.batch_size,
                                  self.transformer,
                                  20,
                                  shuffle=False,
                                  cache_size=self.cache_size)
        val_batcher.restart_at_end = False
        for X, y in tqdm(val_batcher):
            y_true.extend(y)
            y_pred.extend(np.squeeze(self.model.predict(
                np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2])))))
        return y_true, y_pred
