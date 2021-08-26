from abc import abstractmethod, abstractproperty
from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset


class AbstractLivesDataset(AbstractTimeSeriesDataset):
    @abstractproperty
    def rul_column(self) -> str:
        raise NotImplementedError