from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm 
import logging

from ceruleo.dataset.builder.cycles_splitter import CyclesSplitter
from ceruleo.dataset.builder.rul_column import RULColumn 


logger = logging.getLogger(__name__)



class DatasetBuilder:
    splitter: CyclesSplitter
    rul_column: RULColumn

    def __init__(self):
        self.metadata_mapping = None
        self.splitter = None
        self.rul_column = None

    def set_splitting_method(self, splitter: CyclesSplitter):
        self.splitter = splitter
        return self

    def set_rul_column(self, rul_column:RULColumn):
        self.rul_column = rul_column

    def set_life_id_feature(self, name: str):
        self._life_id_feature = name
        return self

    def set_life_end_indicator_feature(self, name: str):
        self._life_end_indicator_feature = name
        return self

    def set_machine_id_feature(self, name: str):
        self._machine_type_feature = name
        return self

    def set_failure_list(self, failures: pd.DataFrame):
        self._failures = failures
        return self

    def build(self, input_path:Path, output_path: Path):
        (output_path / "processed" / "cycles").mkdir(exist_ok=True, parents=True)


        self.splitter.split(input_path)








# Three types of failure specification:
# Upload only one file
# One increasing feature that is the cumulated time of the piece being in place
# A column life id
# A column with a life en indicator
# Upload two files: data + list of failures
# A list of failures
# Upload multiple files
# Separated cycles
