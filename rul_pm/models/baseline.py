from rul_pm.models.model import TrainableModelInterface
from rul_pm.dataset.lives_dataset import AbstractLivesDataset
import numpy as np


class BaselineModel(TrainableModelInterface):
    def __init__(self, rul_column: str):
        self.rul_column = rul_column

    def fit(self, ds: AbstractLivesDataset):
        self.mean_RUL = np.mean([life[self.rul_column].iloc[0] for life in ds])

    def predict(self, ds: AbstractLivesDataset):
        output = []
        for life in ds:
            n_samples = life.shape[0]
            y_pred = np.clip(
                np.ones(n_samples) * self.mean_RUL -
                np.linspace(0, n_samples, n_samples),  0, self.mean_RUL)
            output.append(y_pred)
        return np.concatenate(output)

    def true_values(self, ds: AbstractLivesDataset):
        return np.concatenate([life[self.rul_column] for life in ds])
