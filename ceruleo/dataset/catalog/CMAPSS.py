import logging
import zipfile
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from ceruleo import DATA_PATH
from ceruleo.dataset.ts_dataset import AbstractLivesDataset
from ceruleo.utils.download import download

logger = logging.getLogger(__name__)

DATASET_PATH = DATA_PATH / "C_MAPSS"


URL = "https://data.nasa.gov/download/ff5v-kuh6/application%2Fzip"



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



        

def obtain_raw_files(raw_data_path: Path = DATASET_PATH, ):
    """Download and unzip the raw files

    Parameters:
    
        raw_data_path: Path where to store the dataset
    """
    raw_data_path = raw_data_path / "files"
    logger.info("Dataset not processed.")
    if not raw_data_path.is_dir():
        raw_data_path.mkdir(exist_ok=True, parents=True)
        ZIP_FILE = raw_data_path / "CMAPSSData.zip"
        if not ZIP_FILE.is_file():
            logger.info('Downloading file')            
            download(URL, ZIP_FILE)
        logger.info("Unzipping")
        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall(raw_data_path)
        logger.info("Removing zip file")
        ZIP_FILE.unlink()





def process_file_test(file):
    test_data = pd.read_csv(
        DATASET_PATH / "files" / ("test_" + file + ".txt"),
        names=input_file_column_names,
        delimiter=r"\s+",
        header=None,
    )
    truth_data = pd.read_csv(
        DATASET_PATH / "files" / ("RUL_" + file + ".txt"), delimiter=r"\s+", header=None
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
        DATASET_PATH / "files"/ ("train_" + file + ".txt"),
        sep=r"\s+",
        names=input_file_column_names,
        header=None,
    )
    df = prepare_train_data(df)
    df["OpMode"] = operation_mode[file]
    return df


class CMAPSSDataset(AbstractLivesDataset):
    """C-MAPSS Dataset

    C-MAPSS stands for 'Commercial Modular Aero-Propulsion System Simulation' and it is a tool for the simulation 
    of realistic large commercial turbofan engine data. Each flight is a combination of a 
    series of flight conditions with a reasonable linear transition period to allow the 
    engine to change from one flight condition to the next. The flight conditions are arranged
    to cover a typical ascent from sea level to 35K ft and descent back down to sea level. 
     
    The fault was injected at a given time in one of the flights and persists throughout the 
    remaining flights, effectively increasing the age of the engine. The intent is to identify which 
    flight and when in the flight the fault occurred.

    [Dataset reference](https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq)

    Available models are:
    
        - FD001
        - FD002
        - FD003
        - FD004


    Example:
    
        ``` py
        train_dataset = CMAPSSDataset(train=True, models='FD001')

        validation_dataset = CMAPSSDataset(train=False, models='FD001')
        ```

    Parameters:
    
        train: Wether to obtain the train data provided
        models: Names of the models
    """
    def __init__(
        self, train: bool = True, models: Optional[Union[str, List[str]]] = None
    ):
        super().__init__()
        obtain_raw_files(DATASET_PATH)
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
        return self.lives[i]

    @property
    def n_time_series(self):
        return len(self.lives)

    @property
    def rul_column(self) -> str:
        return "RUL"
