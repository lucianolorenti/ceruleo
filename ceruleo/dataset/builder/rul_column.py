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
    
class CycleRULColumn(RULColumn):
    def __init__(self, cycle_column: str):
        self.cycle_column = cycle_column
    
    def get(self, data: pd.DataFrame) -> np.ndarray:
        return data[self.cycle_column].max() - data[self.cycle_column]
    
class DatetimeRULColumn(RULColumn):
    datetime_column: str
    units: str

    def __init__(self, datetime_column: str, units: str = "h"):
        self.datetime_column = datetime_column
        self.units = units
    
    def get(self, data: pd.DataFrame) -> pd.Series:
        q =  (data[self.datetime_column].max() - data[self.datetime_column]).dt.total_seconds() 
        if self.units == "d":
            return q / (3600*24)
        if self.units == "h":
            return q / 3600
        elif self.units == "min":
            return q / 60
        elif self.units == "s":
            return q
        elif self.units == "ms":
            return q * 1000
        


    