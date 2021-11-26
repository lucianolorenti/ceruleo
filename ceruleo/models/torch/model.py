from typing import Optional

import numpy as np
from ceruleo.iterators.batcher import get_batcher
from ceruleo.models.model import TrainableModel
from torchsummary import summary as model_summary
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

LOSSES = {
    'mae': F.l1_loss,
    'mse': F.mse_loss
}


class TorchTrainableModel(TrainableModel):
    def __init__(self, learning_rate: float = 0.001, loss: str = 'mse', **kwargs):
        super(TorchTrainableModel, self).__init__(**kwargs)
        self.learning_rate = learning_rate
        self._optimizer = None
        self._scheduler = None
        self.loss = LOSSES[loss]

    def build_optimizer(self):
        raise NotImplementedError

    def build_scheduler(self):
        raise None

    def _create_batchers(self, train_dataset, validation_dataset):
        train_batcher, val_batcher = super(TorchTrainableModel, self)._create_batchers(
            train_dataset, validation_dataset)
        train_batcher.restart_at_end = False
        return train_batcher, val_batcher

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._optimizer = self.build_optimizer()
        return self._optimizer

    @property
    def scheduler(self):
        if self._scheduler is None:
            self._scheduler = self.build_scheduler()
        return self._scheduler

    def summary(self):
        model_summary(self.model, self.input_shape)

    def predict(self, dataset, step=None, batch_size=512, evenly_spaced_points: Optional[int] = None):
        step = self.computed_step if step is None else step

        batcher = get_batcher(dataset,
                              self.window,
                              batch_size,
                              self.transformer,
                              step,
                              shuffle=False,
                              output_size=self.output_size,
                              cache_size=self.cache_size,
                              evenly_spaced_points=evenly_spaced_points,
                              restart_at_end=False)

        y_pred = []
        for X, _ in batcher:
            y_pred.append(self.model(torch.Tensor(X)).detach().numpy())

        return np.concatenate(y_pred)

    def fit(self, train_dataset, validation_dataset, epochs: int = 100, refit_transformer: bool = True,
            print_summary: bool = True):
        if refit_transformer:
            self.transformer.fit(train_dataset)

        if print_summary:
            self.summary()
        train_batcher, val_batcher = self._create_batchers(
            train_dataset, validation_dataset)
        for i in tqdm(range(epochs)):
            pbar = tqdm(total=len(train_batcher))
            train_loss = []
            for X, y in train_batcher:
                pbar.set_description('epoch %i' % (i+1))
                self.optimizer.zero_grad()
                y_pred = self.model(torch.Tensor(X))

                single_loss = self.loss(y_pred, torch.Tensor(y))
                single_loss.backward()
                self.optimizer.step()
                train_loss.append(single_loss.detach().numpy())
                pbar.set_postfix(train_loss=np.mean(train_loss))
                pbar.update()

            val_loss = []
            for X, y in val_batcher:
                y_pred = self.model(torch.Tensor(X))
                single_loss = self.loss(y_pred, torch.Tensor(y))
                val_loss.append(single_loss.detach().numpy())

            pbar.set_postfix(train_loss=np.mean(train_loss),
                             val_loss=np.mean(val_loss))

            pbar.close()
