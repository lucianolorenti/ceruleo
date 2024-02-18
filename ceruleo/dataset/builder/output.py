from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd

from ceruleo.dataset.ts_dataset import PDMDataset, PDMInMemoryDataset

class OutputMode(ABC):

    @abstractmethod
    def store(self, cycle_id:str, df: pd.DataFrame):
        pass

    @abstractmethod
    def build_dataset(self, builder:"DatasetBuilder") ->  PDMDataset:
        pass


class InMemoryOutputMode(OutputMode):
    def __init__(self):
        self.out = {}

    def store(self, cycle_id:str, df: pd.DataFrame):
        self.out[cycle_id] = df

    def build_dataset(self, builder:"DatasetBuilder") -> PDMDataset:
        return PDMInMemoryDataset(
            list(self.out.values()),
            "RUL"
        )

class LocalStorageOutputMode(OutputMode):
    output_path: Path 

    def __init__(self, output_path: Path):
        self.output_path = output_path

    def store(self, cycle_id:str, df: pd.DataFrame):
        (self.output_path / "processed" / "cycles").mkdir(exist_ok=True, parents=True)
        
