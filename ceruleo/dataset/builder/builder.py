import logging
import os
from pathlib import Path
from typing import List, Tuple, Union


import pandas as pd
from tqdm.auto import tqdm

from ceruleo.dataset.builder.cycles_splitter import (
    CyclesSplitter,
    FailureDataCycleSplitter,
)
from ceruleo.dataset.builder.output import OutputMode
from ceruleo.dataset.builder.rul_column import RULColumn
from ceruleo.dataset.ts_dataset import PDMDataset

logger = logging.getLogger(__name__)


class DatasetBuilder:
    splitter: CyclesSplitter
    output_mode: OutputMode
    rul_column: RULColumn
    

    def __init__(self):
        """Initializes the builder.
        """
        self.output_mode = None
        self.splitter = None

    @staticmethod
    def one_file_format():
        return DatasetBuilder()

    def set_splitting_method(self, splitter: CyclesSplitter):
        self.splitter = splitter
        return self

    def set_machine_id_feature(self, name: str):
        self._machine_type_feature = name
        return self

    def set_rul_column_method(self, rul_column: RULColumn):
        self.rul_column = rul_column
        return self

    def set_output_mode(self, output_mode: OutputMode):
        self.output_mode = output_mode
        return self

    def _validate(self):
        if self.output_mode is None:
            raise ValueError("Output mode not set")
        if self.splitter is None:
            raise ValueError("Splitting method not set")

    def build(self, input_path: Path):
        self._validate()
        self.splitter.split(input_path, self.output_mode)


    def load_dataframe(self, path: Union[str, Path]) -> pd.DataFrame:
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        if path.suffix == ".csv":
            return pd.read_csv(path)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".xlsx":
            return pd.read_excel(path)
        raise ValueError(f"Unsupported file format {path.suffix}")

    def build_from_data_fault_pairs_files(
        self, data_fault_pairs: Union[Tuple[str, str], List[Tuple[str, str]]]
    )-> PDMDataset:
        if not isinstance(data_fault_pairs, list):
            data_fault_pairs = [data_fault_pairs]

        if not isinstance(self.splitter, FailureDataCycleSplitter):
            raise ValueError(
                "This method is only available for FailureDataCycleSplitter"
            )
        
            
        common_path_prefix = os.path.commonprefix([data for data, fault in data_fault_pairs])

        for i, (data, fault) in enumerate(tqdm(data_fault_pairs)):
            df_data = self.load_dataframe(data)
            df_faults = self.load_dataframe(fault)
            cycles_in_file = self.splitter.split(df_data, df_faults)
            for j, ds in enumerate(cycles_in_file):
                cycle_id = f"{i+1}_{j+1}"
                self._build_and_store_cycle(ds, cycle_id, metadata={
                    "filename": str(data.relative_to(common_path_prefix)),
                    "fault_filename": str(fault.relative_to(common_path_prefix))
                })
        self.output_mode.finish()
        return self.output_mode.build_dataset(self)

    def build_from_df(self, data:pd.DataFrame) -> PDMDataset:
        self._validate()
        for i, ds in enumerate(self.splitter.split(data)):
            self._build_and_store_cycle(ds, i+1)
        self.output_mode.finish()
        return self.output_mode.build_dataset(self)

    def _build_and_store_cycle(self, ds:pd.DataFrame, cycle_id: any, metadata: dict = {}):
        ds["RUL"] = self.rul_column.get(ds)
        self.output_mode.store(f"Cycle_{cycle_id}", ds, metadata)

