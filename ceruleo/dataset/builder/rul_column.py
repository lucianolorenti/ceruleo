from abc import ABC, abstractmethod
import pandas as pd 
import numpy as np

class RULColumn(ABC):
    @abstractmethod
    def get(self, data: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
    
class NumberOfRowsRULColumn(RULColumn):
    def get(self, data: pd.DataFrame) -> np.ndarray:
        return np.arange(data.shape[0] - 1, -1, -1)



    