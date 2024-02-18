from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm 
import logging

from ceruleo.dataset.builder.cycles_splitter import CyclesSplitter
from ceruleo.dataset.builder.output import OutputMode
from ceruleo.dataset.builder.rul_column import RULColumn 


logger = logging.getLogger(__name__)



class DatasetBuilder:
    splitter: CyclesSplitter
    output_mode: OutputMode
    rul_column: RULColumn

    def __init__(self):
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

    def build(self, input_path:Path):
        self._validate()
        self.splitter.split(input_path, self.output_mode)

    def build_from_df(self, *args):
        self._validate()
        for i, ds in enumerate(self.splitter.split(*args)):
            ds["RUL"] = self.rul_column.get(ds)
            self.output_mode.store(f"Cycle_{i}", ds)
        return self.output_mode.build_dataset(self)









