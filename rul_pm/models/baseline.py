from rul_pm.results.results import compute_rul_line
from rul_pm.models.model import TrainableModelInterface
from rul_pm.dataset.lives_dataset import AbstractLivesDataset
import numpy as np


class BaselineModelAbstract(TrainableModelInterface):
    def __init__(self, transformer):
        self.transformer = transformer 

    def true_values(self, ds: AbstractLivesDataset):
        true = []
        for l in ds:
            _, y, _ =self.transformer.transform(l)
            true.extend(y)
        return np.array(true)

class BaselineModel(BaselineModelAbstract):
    def __init__(self, transformer, mode:str='mean'):
        super().__init__(transformer)        
        self.mode = mode

    def fit(self, ds: AbstractLivesDataset):
                
        true = []
        for l in ds:
            _, y, _ =self.transformer.transform(l)
            true.append(y[0])

        if self.mode == 'mean':
            self.fitted_RUL = np.mean(true)
        elif self.mode == 'median':
            self.fitted_RUL = np.median(true)

    def predict(self, ds: AbstractLivesDataset):
        output = []
        for life in ds:
            _, y, _ =self.transformer.transform(life)
            y_pred = np.clip(
                self.fitted_RUL+np.cumsum(np.diff(y)),  0, self.fitted_RUL)
            output.append(y_pred)
        return np.concatenate(output)




class FixedValueBaselineModel(BaselineModelAbstract):
    def __init__(self, transformer: str, value):
        super().__init__(transformer)
        self.value = value

    def fit(self, ds: AbstractLivesDataset):
        return self        

    def predict(self, ds: AbstractLivesDataset):
        output = []
        for life in ds:
            n_samples = life.shape[0]
            y_pred = compute_rul_line(self.value, life.shape[0])
            output.append(y_pred)
        return np.concatenate(output)
