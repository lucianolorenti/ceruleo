from typing import List, Optional, Union

import numpy as np
import pandas as pd
from rul_pm.datasets.lives_dataset import AbstractLivesDataset
from temporis import DATA_PATH

CMAPSS_PATH = DATA_PATH / "C_MAPSS"

# Features used by
# Multiobjective Deep Belief Networks Ensemble forRemaining Useful Life Estimation in
# Prognostics Chong Zhang, Pin Lim, A. K. Qin,Senior Member, IEEE, and Kay Chen Tan,Fellow, IEEE
sensor_indices = np.array([2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]) + (4 - 1)

dependent_vars = ["RemainingUsefulLife"]
index_columns_names = ["UnitNumber", "Cycle"]
operational_settings_columns_names = ["OpSet" + str(i) for i in range(1, 4)]
sensor_measure_columns_names = ["SensorMeasure" + str(i) for i in range(1, 22)]
input_file_column_names = (
    index_columns_names
    + operational_settings_columns_names
    + sensor_measure_columns_names
)

operation_mode = {"FD001": 0, "FD002": 1, "FD003": 2, "FD004": 3}
engines = ["FD001", "FD002", "FD003", "FD004"]


def process_file_test(file):
    test_data = pd.read_csv(
        CMAPSS_PATH / ("test_" + file + ".txt"),
        names=input_file_column_names,
        delimiter=r"\s+",
        header=None,
    )
    truth_data = pd.read_csv(
        CMAPSS_PATH / ("RUL_" + file + ".txt"), delimiter=r"\s+", header=None
    )
    truth_data.columns = ["truth"]
    truth_data["UnitNumber"] = np.array(range(truth_data.shape[0])) + 1
    test_rul = test_data.groupby("UnitNumber")["Cycle"].max().reset_index()
    test_rul.columns = ["UnitNumber", "Elapsed"]
    test_rul = test_rul.merge(truth_data, on=["UnitNumber"], how="left")
    test_rul["Max"] = test_rul["Elapsed"] + test_rul["truth"]

    test_data = test_data.merge(test_rul, on=["UnitNumber"], how="left")
    test_data["RUL"] = test_data["Max"] - test_data["Cycle"]
    test_data.drop(["Max"], axis=1, inplace=True)
    return test_data


def prepare_train_data(data, factor: float = 0):
    """
    Paramaters
    ----------
    data: pd.DataFrame
          Dataframe with the file content
    cutoff: float.
             RUL cutoff
    """
    df = data.copy()
    fdRUL = df.groupby("UnitNumber")["Cycle"].max().reset_index()
    fdRUL = pd.DataFrame(fdRUL)
    fdRUL.columns = ["UnitNumber", "max"]
    df = df.merge(fdRUL, on=["UnitNumber"], how="left")
    df["RUL"] = df["max"] - df["Cycle"]
    df.drop(columns=["max"], inplace=True)

    return df


def process_file_train(file):
    df = pd.read_csv(
        CMAPSS_PATH / ("train_" + file + ".txt"),
        sep=r"\s+",
        names=input_file_column_names,
        header=None,
    )
    df = prepare_train_data(df)
    df["OpMode"] = operation_mode[file]
    return df


class CMAPSSDataset(AbstractLivesDataset):
    def __init__(
        self, train: bool = True, models: Optional[Union[str, List[str]]] = None
    ):
        if models is not None and isinstance(models, str):
            models = [models]
        self._validate_model_names(models)
        if train:
            processing_fun = process_file_train
        else:
            processing_fun = process_file_test
        self.lives = []

        for engine in engines:
            if models is not None and engine not in models:
                continue
            for _, g in processing_fun(engine).groupby("UnitNumber"):
                g.drop(columns=["UnitNumber"], inplace=True)
                g["Engine"] = engine
                self.lives.append(g)

    def _validate_model_names(self, models):
        if models is not None:
            for model in models:
                if model not in operation_mode:
                    raise ValueError(
                        f"Invalid model: valid model are {list(operation_mode.keys())}"
                    )

    def get_time_series(self, i):
        """

        Returns
        -------
        pd.DataFrame
            DataFrame with the data of the life i
        """
        return self.lives[i]

    @property
    def n_time_series(self):
        return len(self.lives)

    @property
    def rul_column(self) -> str:
        return "RUL"
