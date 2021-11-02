import logging
import tarfile
from enum import Enum
from pathlib import Path
from typing import Optional
import io
import os
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






def prepare_dataset(path: Path):
    def track_progress(members):
        for member in tqdm(members):      
            yield member  

    path.mkdir(parents=True, exist_ok=True)
    if not (path / OUTPUT).resolve().is_file():
        download(path)
    logger.info("Decompressing  dataset...")
    with tarfile.open(path / OUTPUT, 'r') as tarball:
        tarball.extractall(path=path.parent, members = track_progress(tarball))
    (path / OUTPUT).unlink()



@memory.cache
def cached_data(data_file: Path, ttf_file: Path, ttf_column: str):
    data = pd.read_csv(data_file).set_index("time")
    time = pd.read_csv(ttf_file).set_index("time").loc[:, ttf_column].dropna()
    return pd.merge(data, time, on="time", how="left")


class FailureType(Enum):
    FlowCoolPressureDroppedBelowLimit = "TTF_FlowCool Pressure Dropped Below Limit"
    FlowcoolPressureTooHighCheckFlowcoolPump = (
        "TTF_Flowcool Pressure Too High Check Flowcool Pump"
    )
    FlowcoolLeak = "TTF_Flowcool leak"


class SubsetType(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class PHMDataset2018(AbstractLivesDataset):
    def __init__(
        self,
        subset_type: SubsetType,
        failure_type: FailureType,
        path: Optional[Path] = DATASET_PATH,
    ):
        self.path = path
        self.dataset_path = path / FOLDER
        self.subset_type = subset_type
        if not (self.dataset_path / "train").is_dir():
            prepare_dataset(self.dataset_path)            

        self.failure_type = failure_type
        self.files = list(Path(self.dataset_path / "train").resolve().glob("*.csv"))
        self.ttf_files = list(
            Path(self.dataset_path / "train" / "train_ttf").resolve().glob("*.csv")
        )
        self.lives_limits = {}

        self.files = {file.name: file for file in self.files}
        self.ttf_files = {file.name: file for file in self.ttf_files}
        self.nlives = 0
        self._process_ttf_files()

    @property
    def n_time_series(self) -> int:
        return self.nlives

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
        (file, start, end) = self.lives_list[i]
        data_file = self.files[file]
        ttf_file = self.ttf_files[file]
        return cached_data(data_file, ttf_file, self.failure_type.value).loc[
            start:end, :
        ]

    def _process_ttf_files(self):
        self.nlives = 0
        self.lives_list = []
        for filename in self.ttf_files.keys():
            time = (
                pd.read_csv(self.ttf_files[filename])
                .set_index("time")
                .loc[:, self.failure_type.value]
                .dropna()
            )
            if len(time) == 0:
                continue
            time_diff = time.diff()
            lives_limits = [
                time.index[0],
                *time_diff.where(time_diff > 0).dropna().index.tolist(),
                time.index[-1],
            ]
            for i in range(len(lives_limits) - 1):
                start = lives_limits[i]
                end = lives_limits[i + 1] - 1
                self.lives_list.append((filename, start, end))

            nlives = len(lives_limits) - 1
            self.nlives += nlives
            self.lives_limits[filename] = lives_limits

    @property
    def rul_column(self) -> str:
        return self.failure_type.value
