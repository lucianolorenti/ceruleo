import gzip
import io
import logging
import os
import pickle
import tarfile
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import gdown
import pandas as pd
from joblib import Memory
from rul_pm import CACHE_PATH, DATASET_PATH
from rul_pm.datasets.lives_dataset import AbstractLivesDataset
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

memory = Memory(CACHE_PATH, verbose=0)

COMPRESSED_FILE = "phm_data_challenge_2018.tar.gz"
FOLDER = "phm_data_challenge_2018"


URL = "https://drive.google.com/uc?id=15Jx9Scq9FqpIGn8jbAQB_lcHSXvIoPzb"
OUTPUT = COMPRESSED_FILE


def download(path: Path):
    logger.info("Downloading dataset...")
    gdown.download(URL, str(path / OUTPUT), quiet=False)


def prepare_raw_dataset(path: Path):
    def track_progress(members):
        for member in tqdm(members):
            yield member

    path.mkdir(parents=True, exist_ok=True)
    if not (path / OUTPUT).resolve().is_file():
        download(path)
    logger.info("Decompressing  dataset...")
    with tarfile.open(path / OUTPUT, "r") as tarball:
        tarball.extractall(path=path.parent, members=track_progress(tarball))
    (path / OUTPUT).unlink()


class FailureType(Enum):
    FlowCoolPressureDroppedBelowLimit = "TTF_FlowCool Pressure Dropped Below Limit"
    FlowcoolPressureTooHighCheckFlowcoolPump = (
        "TTF_Flowcool Pressure Too High Check Flowcool Pump"
    )
    FlowcoolLeak = "TTF_Flowcool leak"


def prepare_dataset(dataset_path: Path):
    (dataset_path / "processed" / "lives").mkdir(exist_ok=True, parents=True)

    files = list(Path(dataset_path / "raw" / "train").resolve().glob("*.csv"))
    ttf_files = list(
        Path(dataset_path / "raw" / "train" / "train_ttf").resolve().glob("*.csv")
    )
    files = {file.stem: file for file in files}
    ttf_files = {file.stem: file for file in ttf_files}
    dataset_data = []
    for filename in tqdm(ttf_files.keys(), "Processing files"):
        data_file = files[filename]
        logger.info(f"Loading data file {files[filename]}")
        data = pd.read_csv(data_file).dropna().set_index("time")

        c = data.columns.tolist()
        ttf_data_file = ttf_files[filename]
        ttf_data = pd.read_csv(ttf_data_file).set_index("time")
        data = pd.merge(data, ttf_data, on="time", how="left")
        for failure_type in FailureType:
            time = data.loc[:, failure_type.value].dropna()
            if len(time) == 0:
                continue
            time_diff = time.diff()
            lives_limits = [
                time.index[0],
                *time_diff.where(time_diff > 0).dropna().index.tolist(),
                time.index[-1],
            ]

            for i in tqdm(range(len(lives_limits) - 1), leave=False):
                start = lives_limits[i]
                end = lives_limits[i + 1] - 1
                slice = data.loc[start:end, :][c + [failure_type.value]]
                tool = slice["Tool"].iloc[0]
                output_filename = f"Life_{i}_{tool}_{failure_type.value}.pkl.gzip"
                dataset_data.append(
                    (tool, slice.shape[0], failure_type.value, output_filename)
                )
                with gzip.open(
                    dataset_path / "processed" / "lives" / output_filename, "wb"
                ) as file:
                    pickle.dump(slice, file)

    df = pd.DataFrame(
        dataset_data, columns=["Tool", "Number of samples", "Failure Type", "Filename"]
    )
    df.to_csv(dataset_path / "processed" / "lives" / "lives_db.csv")


class PHMDataset2018(AbstractLivesDataset):
    def __init__(
        self,
        failure_types: Union[FailureType, List[FailureType]] = [l for l in FailureType],
        tools: Union[str, List[str]] = "all",
        path: Path = DATASET_PATH,
    ):
        super().__init__()
        if not isinstance(failure_types, list):
            failure_types = [failure_types]
        self.failure_types = failure_types

        if isinstance(tools, str) and tools != "all":
            tools = [tools]
        self.tools = tools

        self.path = path
        self.dataset_path = path / FOLDER

        self.procesed_path = self.dataset_path / "processed" / "lives"
        self.lives_table_filename = self.procesed_path / "lives_db.csv"
        if not self.lives_table_filename.is_file():
            if not (self.dataset_path / "raw" / "train").is_dir():
                prepare_raw_dataset(self.dataset_path)
            prepare_dataset(self.dataset_path)

        self.lives = pd.read_csv(self.lives_table_filename)
        if tools != "all":
            self.lives = self.lives[self.lives["Tool"].isin(self.tools)]
        self.lives = self.lives[
            self.lives["Failure Type"].isin([a.value for a in self.failure_types])
        ]

    @property
    def n_time_series(self) -> int:
        return self.lives.shape[0]

    def get_time_series(self, i: int) -> pd.DataFrame:
        """
        Paramters
        ---------
        i:int


        Returns
        -------
        pd.DataFrame
            DataFrame with the data of the life i
        """
        with gzip.open(
            self.procesed_path / self.lives.iloc[i]["Filename"], "rb"
        ) as file:
            df = pickle.load(file)
        df['RUL'] = df[df.columns[-1]]
        return df

    @property
    def rul_column(self) -> str:
        return 'RUL'
