import numpy as np
from rul_pm.iterators.batcher import get_batcher
from rul_pm.models.model import TrainableModel
from tqdm.auto import tqdm

from sklearn.metrics import mean_squared_error


class SKLearnModel(TrainableModel):
    def __init__(self,
                 model,
                 window,
                 batch_size,
                 step,
                 transformer,
                 shuffle,
                 models_path,
                 patience=4,
                 cache_size=30):
        super().__init__(window,
                         batch_size,
                         step,
                         transformer,
                         shuffle,
                         models_path,
                         patience=patience,
                         cache_size=cache_size)
        self._model = model

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
