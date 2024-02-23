from enum import Enum
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from ceruleo.dataset.ts_dataset import PDMDataset, PDMInMemoryDataset

logger = logging.getLogger(__name__)


class OutputMode(ABC):
    medatada_columns: dict

    def __init__(self):
        self.medatada_columns = {}
        self.cycles_metadata = {}

    @abstractmethod
    def store_cycle(self, cycle_id: str, df: pd.DataFrame):
        pass

    def store(self, cycle_id: str, df: pd.DataFrame, metadata: dict = {}):
        self.store_metadata(cycle_id, df, metadata)
        self.store_cycle(cycle_id, df)

    def set_metadata_columns(self, columns: dict) -> "OutputMode":
        self.medatada_columns = columns
        return self

    @abstractmethod
    def build_dataset(self, builder: "DatasetBuilder") -> PDMDataset:
        pass

    def extract_metadata(self, df: pd.DataFrame) -> dict:
        m = {
            k: df[k].iloc[0] if k in df.columns else v
            for k, v in self.medatada_columns.items()
        }
        m["Number of samples"] = df.shape[0]
        m["Duration"] = df.RUL.max()
        return m
    
    def finish(self):
        pass
        

    def store_metadata(
        self, cycle_id: str, df: pd.DataFrame, additional_medatada: dict
    ):
        metadata = self.extract_metadata(df)
        metadata["cycle_id"] = cycle_id
        metadata.update(additional_medatada)
        self.cycles_metadata[cycle_id] = metadata


class InMemoryOutputMode(OutputMode):
    def __init__(self):
        super().__init__()
        self.cycles = {}

    def store_cycle(self, cycle_id: str, df: pd.DataFrame):
        self.cycles[cycle_id] = df

    def build_dataset(self, builder: "DatasetBuilder") -> PDMDataset:
        return PDMInMemoryDataset(list(self.cycles.values()), "RUL")


class DatasetFormat(Enum):
    CSV = "csv"
    PARQUET = "parquet"
    FEATHER = "feather"


class LocalStorageOutputMode(OutputMode):
    output_path: Path
    output_format: DatasetFormat

    def __init__(
        self, output_path: Path, output_format: DatasetFormat = DatasetFormat.CSV
    ):
        super().__init__()
        self.output_path = output_path
        self.output_format = output_format

    def store_cycle(self, cycle_id: str, df: pd.DataFrame):
        output_cycles_path = self.output_path / "processed" / "cycles"
        if not output_cycles_path.exists():
            logger.info(f"Creating {output_cycles_path}")
            output_cycles_path.mkdir(exist_ok=True, parents=True)
        logger.info(f"Storing cycle {cycle_id}")
        if self.output_format == DatasetFormat.CSV:
            df.to_csv(output_cycles_path / f"{cycle_id}.csv", index=False)
        elif self.output_format == DatasetFormat.PARQUET:
            df.to_parquet(output_cycles_path / f"{cycle_id}.parquet", index=False)
        elif self.output_format == DatasetFormat.FEATHER:
            df.to_feather(output_cycles_path / f"{cycle_id}.feather")
        else:
            raise ValueError(f"Unsupported output format {self.output_format}")

    def finish(self):
        output_cycles_path = self.output_path / "processed" / "cycles"
        pd.DataFrame(self.cycles_metadata).T.to_csv(output_cycles_path / "cycles.csv")

    def build_dataset(self, builder: "DatasetBuilder") -> PDMDataset:
        return PDMDataset(
            self.output_path
        )
