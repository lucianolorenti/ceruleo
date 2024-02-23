import gzip
import io
import logging
import os
import pickle
import shutil
import tarfile
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import gdown
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ceruleo import CACHE_PATH, DATA_PATH
from ceruleo.dataset.builder.builder import DatasetBuilder
from ceruleo.dataset.builder.cycles_splitter import FailureDataCycleSplitter
from ceruleo.dataset.builder.output import DatasetFormat, LocalStorageOutputMode
from ceruleo.dataset.builder.rul_column import NumberOfRowsRULColumn
from ceruleo.dataset.ts_dataset import AbstractPDMDataset, PDMDataset

logger = logging.getLogger(__name__)

COMPRESSED_FILE = "phm_data_challenge_2018.tar.gz"
FOLDER = "phm_data_challenge_2018"


URL = "https://drive.google.com/uc?id=15Jx9Scq9FqpIGn8jbAQB_lcHSXvIoPzb"
OUTPUT = COMPRESSED_FILE


def download(url: str, path: Path):
    logger.info("Downloading dataset...")
    gdown.download(url, str(path / OUTPUT), quiet=False)


class FailureType(Enum):
    """Failure types availables for the dataset.

    Possible values are:
    ```
    FailureType.FlowCoolPressureDroppedBelowLimit
    FailureType.FlowcoolPressureTooHighCheckFlowcoolPump
    FailureType.FlowcoolLeak
    ```
    """

    FlowCoolPressureDroppedBelowLimit = "FlowCool Pressure Dropped Below Limit"
    FlowcoolPressureTooHighCheckFlowcoolPump = (
        'Flowcool Pressure Too High Check Flowcool Pump'
    )
    FlowcoolLeak = "Flowcool leak"
    FlowcoolPressureTooHighCheckFlowcoolPumpNoWaferID = 'Flowcool Pressure Too High Check Flowcool Pump [NoWaferID]'


    @staticmethod
    def that_starth_with(s: str):
        for f in FailureType:
            if s.startswith(f.value):
                return f
        return None


class PHMDataset2018(PDMDataset):
    """PHM 2018 Dataset

    The 2018 PHM dataset is a public dataset released by Seagate which contains the execution of 20 different
    ion milling machines. They distinguish three different failure causes and provide 22 features,
    including user-defined variables and sensors.

    Three faults are present in the dataset

    * Fault mode 1 occurs when flow-cool pressure drops.
    * Fault mode 2 occurs when flow-cool pressure becomes too high.
    * Fault mode 3 represents flow-cool leakage.

    [Dataset reference](https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-prognostics-and-health-management-society-2018-b/phm-data-challenge-6/)

    Example:

    ```py
    dataset = PHMDataset2018(
        failure_types=FailureType.FlowCoolPressureDroppedBelowLimit,
        tools=['01_M02']
    )
    ```



    Parameters:
        path: Path where the dataset is located
    """

    failure_types: Optional[List[FailureType]]
    tools: Optional[List[str]]

    def __init__(
        self,
        path: Path = DATA_PATH,
        url: str = URL,
        failure_types: Optional[Union[FailureType, List[FailureType]]] = None,
        tools: Optional[Union[str, List[str]]] = None,
    ):
        self.url = url
        super().__init__(path / "phm_data_challenge_2018")
        self._prepare_dataset()
        self.failure_types = failure_types
        self.tools = tools

        if self.failure_types is not None:
            if not isinstance(self.failure_types, list):
                self.failure_types = [failure_types]
            self.cycles_metadata = self.cycles_metadata[
                self.cycles_metadata["Fault name"].isin(
                    [f.value for f in self.failure_types]
                )
            ]
 
        
        if self.tools is not None:
            if not isinstance(self.tools, list):
                self.tools = [tools]
            self.cycles_metadata = self.cycles_metadata[
                self.cycles_metadata["Tool"].isin(self.tools)
            ]

    def _prepare_dataset(self):
        if self.cycles_table_filename.is_file():
            return
        if not (self.dataset_path / "raw" / "train").is_dir():
            self.prepare_raw_dataset()
        files = list(Path(self.dataset_path / "raw" / "train").resolve().glob("*.csv"))
        faults_files = list(
            Path(self.dataset_path / "raw" / "train" / "train_faults")
            .resolve()
            .glob("*.csv")
        )

        def get_key_from_filename(filename: str) -> str:
            return "_".join(filename.split("_")[0:2])

        fault_files_map = {get_key_from_filename(f.name): f for f in faults_files}
        data_fault_pairs = [
            (file, fault_files_map[get_key_from_filename(file.name)]) for file in files
        ]

        (
            DatasetBuilder()
            .set_splitting_method(
                FailureDataCycleSplitter(
                    data_time_column="time", fault_time_column="time"
                )
            )
            .set_rul_column_method(NumberOfRowsRULColumn())
            .set_output_mode(
                LocalStorageOutputMode(
                    self.dataset_path, output_format=DatasetFormat.PARQUET
                ).set_metadata_columns(
                    {"Tool": "Tool_data", "Fault name": "fault_name"}
                )
            )
            .set_index_column("time")
            .prepare_from_data_fault_pairs_files(
                data_fault_pairs,
            )
        )

    @property
    def n_time_series(self) -> int:
        return self.cycles_metadata.shape[0]

    def _load_life(self, filename: str) -> pd.DataFrame:
        return pd.read_parquet(filename)

    def get_time_series(self, i: int) -> pd.DataFrame:
        df = self._load_life(self.cycles_metadata.iloc[i]["Filename"])
        return df

    @property
    def rul_column(self) -> str:
        return "RUL"

    def prepare_raw_dataset(self):
        """Download and unzip the raw files

        Args:
            path (Path): Path where to store the raw dataset
        """

        def track_progress(members):
            for member in tqdm(members, total=70):
                yield member

        path = self.dataset_path / "raw"
        path.mkdir(parents=True, exist_ok=True)
        print(path / OUTPUT)
        if not (path / OUTPUT).resolve().is_file():
            download(self.url, path)
        logger.info("Decompressing  dataset...")
        with tarfile.open(path / OUTPUT, "r") as tarball:

            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory

            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")

                tar.extractall(path, members, numeric_owner=numeric_owner)

            safe_extract(tarball, path=path, members=track_progress(tarball))
        shutil.move(
            str(path / "phm_data_challenge_2018" / "train"), str(path / "train")
        )
        shutil.move(str(path / "phm_data_challenge_2018" / "test"), str(path / "test"))
        shutil.rmtree(str(path / "phm_data_challenge_2018"))
        (path / OUTPUT).unlink()
