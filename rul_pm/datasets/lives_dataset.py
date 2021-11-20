from abc import abstractmethod, abstractproperty
from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset
import pandas as pd

class AbstractLivesDataset(AbstractTimeSeriesDataset):
    @abstractproperty
    def rul_column(self) -> str:
        raise NotImplementedError

    def duration(self, life: pd.DataFrame) -> float:
        return life[self.rul_column].max()