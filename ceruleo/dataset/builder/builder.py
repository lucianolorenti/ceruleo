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

    def build_from_df(self, df: pd.DataFrame):
        self._validate()
        self.splitter.split(df, self.output_mode)
        return self.output_mode.build_dataset()








# Three types of failure specification:
# Upload only one file
# One increasing feature that is the cumulated time of the piece being in place
# A column life id
# A column with a life en indicator
# Upload two files: data + list of failures
# A list of failures
# Upload multiple files
# Separated cycles
