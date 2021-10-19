from pathlib import Path
from typing import Optional
from rul_pm import DATASET_PATH
from enum import Enum
from rul_pm.datasets.lives_dataset import AbstractLivesDataset
import pandas as pd

COMPRESSED_FILE = "phm_data_challenge_2018.tar.gz"
FOLDER = "phm_data_challenge_2018"


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
        if not self.dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset path not found in {self.dataset_path}")

        self.failure_type = failure_type
        self.files = list(Path(self.dataset_path / "train").resolve().glob("*.csv"))
        self.ttf_files = list(
            Path("phm_data_challenge_2018/train/train_ttf").resolve().glob("*.csv")
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

        Returns
        -------
        pd.DataFrame
            DataFrame with the data of the life i
        """
        (file, start, end) = self.lives_list[i]
        data = pd.read_csv(self.files[file]).set_index("time")
        time = (
            pd.read_csv(self.ttf_files[file])
            .set_index("time")
            .loc[:, self.failure_type.value]
            .dropna()
        )
        return pd.merge(data, time, on="time", how="inner").loc[start:end, :]

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
            for i in range(len(lives_limits)- 1):
                start  = lives_limits[i]
                end = lives_limits[i+1]
                self.lives_list.append((filename, start, end))

            nlives = len(lives_limits) - 1
            self.nlives += nlives
            self.lives_limits[filename] = lives_limits
