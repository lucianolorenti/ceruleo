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
from ceruleo.dataset.builder.builder import DatasetBuilder
from ceruleo.dataset.builder.cycles_splitter import FailureDataCycleSplitter
from ceruleo.dataset.builder.rul_column import NumberOfRowsRULColumn

import gdown
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ceruleo import CACHE_PATH, DATA_PATH
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

    Possible values are

    ```py
    FailureType.FlowCoolPressureDroppedBelowLimit
    FailureType.FlowcoolPressureTooHighCheckFlowcoolPump
    FailureType.FlowcoolLeak
    ```

    """

    FlowCoolPressureDroppedBelowLimit = "FlowCool Pressure Dropped Below Limit"
    FlowcoolPressureTooHighCheckFlowcoolPump = (
        "Flowcool Pressure Too High Check Flowcool Pump"
    )
    FlowcoolLeak = "Flowcool leak"

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

    def __init__(
        self,
        path: Path = DATA_PATH,
        url: str = URL
    ):
        self.url = url
        super().__init__(path)



    def _prepare_dataset(self):
        if self.lives_table_filename.is_file():
            return
        if not (self.dataset_path / "raw" / "train").is_dir():
            self.prepare_raw_dataset(self.dataset_path)
        files = list(
            Path(self.dataset_path / "raw" / "train").resolve().glob("*.csv")
        )
        faults_files = list(
            Path(self.dataset_path / "raw" / "train" / "train_faults")
            .resolve()
            .glob("*.csv")
        )

        (
            DatasetBuilder()
            .set_splitting_method(FailureDataCycleSplitter())
            .set_rul_column(NumberOfRowsRULColumn())
            .add_cycle_metadata({
                "Tool": "Tool",
                "Failure Type": "Failure Type",
            })
            .build(
                raw=self.dataset_path / "raw",
                output=self.dataset_path / "processed",
            )
        )

    @property
    def n_time_series(self) -> int:
        return self.lives.shape[0]

    def _load_life(self, filename: str) -> pd.DataFrame:
        with gzip.open(self.procesed_path / filename, "rb") as file:
            df = pickle.load(file)
        return df

    def get_time_series(self, i: int) -> pd.DataFrame:
        df = self._load_life(self.lives.iloc[i]["Filename"])
        #df.index = pd.to_timedelta(df.index, unit="s")
        #df = df[df["FIXTURESHUTTERPOSITION"] == 1]
        #df["RUL"] = np.arange(df.shape[0] - 1, -1, -1)

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

        path = self.output_path / "raw"
        path.mkdir(parents=True, exist_ok=True)
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
        shutil.move(str(path / "phm_data_challenge_2018" / "train"), str(path / "train"))
        shutil.move(str(path / "phm_data_challenge_2018" / "test"), str(path / "test"))
        shutil.rmtree(str(path / "phm_data_challenge_2018"))
        (path / OUTPUT).unlink()