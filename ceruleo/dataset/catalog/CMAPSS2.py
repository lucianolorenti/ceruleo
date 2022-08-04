import gc
import gzip
import logging
import pickle
import zipfile
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import pandas as pd
from ceruleo import DATA_PATH
from ceruleo.dataset.ts_dataset import AbstractLivesDataset
from tqdm.auto import tqdm

from ceruleo.utils.download import download

logger = logging.getLogger(__name__)

DATASET_PATH = DATA_PATH / "C_MAPSS2"

URL = "https://ti.arc.nasa.gov/c/47/"


def load_file(filename: Union[Path, str], train: bool = True):
    with h5py.File(filename, "r") as hdf:
        if train:
            W = np.array(hdf.get("W_dev"))  # W
            X_sv = np.array(hdf.get("X_s_dev"))  # X_s
            X_v = np.array(hdf.get("X_v_dev"))  # X_v
            T = np.array(hdf.get("T_dev"))  # T
            Y = np.array(hdf.get("Y_dev"))  # RUL
            A = np.array(hdf.get("A_dev"))  # Auxiliary

        else:
            W = np.array(hdf.get("W_test"))  # W
            X_sv = np.array(hdf.get("X_s_test"))  # X_s
            X_v = np.array(hdf.get("X_v_test"))  # X_v
            T = np.array(hdf.get("T_test"))  # T
            Y = np.array(hdf.get("Y_test"))  # RUL
            A = np.array(hdf.get("A_test"))  # Auxiliary

        # Varnams
        W_var = np.array(hdf.get("W_var"))
        X_s_var = np.array(hdf.get("X_s_var"))
        X_v_var = np.array(hdf.get("X_v_var"))
        T_var = np.array(hdf.get("T_var"))
        A_var = np.array(hdf.get("A_var"))

        # from np.array to list dtype U4/U5
        W_var = list(np.array(W_var, dtype="U20"))
        X_s_var = list(np.array(X_s_var, dtype="U20"))
        X_v_var = list(np.array(X_v_var, dtype="U20"))
        T_var = list(np.array(T_var, dtype="U20"))
        A_var = list(np.array(A_var, dtype="U20"))

        data = pd.DataFrame(
            np.concatenate((W, X_sv, X_v, T, Y, A), axis=1),
            columns=W_var + X_s_var + X_v_var + T_var + ["RUL"] + A_var,
        )
        return data


class CMAPSS2PreProcessor:
    def __init__(self, path: Path = DATASET_PATH):
        self.path = path
        self.raw_data_path = path / "data_set"
        self.lives_table_path = path / "lives_data.pkl"

    def process_raw_dataframe(
        self, data:pd.DataFrame, file_h5: Path, i: int, unit: int, train: bool
    ):
        unit = int(unit)
        output_dir = f"LIFE_{file_h5.name}_{unit}_{i}"
        number_of_samples = data.shape[0]
        data.to_parquet(self.path / output_dir)
        return {
            "Raw File": file_h5.name,
            "Unit": unit,
            "Index": i,
            "Output Dir": output_dir,
            "Train": train,
            "Number of samples": number_of_samples,
        }

    def process_raw_file(self, file_h5: Path, train: bool):
        info = []
        try:
            df = load_file(file_h5, train=train)
            for i, (unit, data) in enumerate(df.groupby(by=["unit"])):
                info.append(self.process_raw_dataframe(data, file_h5, i, unit, train))
            gc.collect()
        except OSError as e:
            logger.error(f"Cannot open file {file_h5}")
            logger.error(e)
        return info

    def process_raw_files(self):
        info = []
        for file_h5 in tqdm(list(self.raw_data_path.glob("*.h5"))):
            info.extend(self.process_raw_file(file_h5, train=True))
            info.extend(self.process_raw_file(file_h5, train=False))

        df = pd.DataFrame(info)
        with open(self.lives_table_path, "wb") as file:
            pickle.dump(df, file)



    def obtain_raw_files(self):
        logger.info("Dataset not processed.")
        if not self.raw_data_path.is_dir():
            ZIP_FILE = self.path / "data_set.zip"
            if not ZIP_FILE.is_file():
                logger.info('Downloading file')
                self.raw_data_path.mkdir(exist_ok=True, parents=True)
                download(URL, ZIP_FILE)
            logger.info("Unzipping")
            with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
                zip_ref.extractall(self.path)
            logger.info("Removing zip file")
            ZIP_FILE.unlink()

    def run(self):
        if not (self.lives_table_path).is_file():
            self.obtain_raw_files()
        self.process_raw_files()



class CMAPSS2Dataset(AbstractLivesDataset):
    """C-MAPSS-2 Dataset

    The dataset provides a new realistic dataset of run-to-failure trajectories for a small fleet of aircraft
    engines under realistic flight conditions.

    The damage propagation modelling used for the generation of this synthetic dataset builds on
    the modeling strategy from previous work .
    The dataset was generated with the Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dynamical model.
    The data set is been provided by the Prognostics CoE at NASA Ames in collaboration with ETH Zurich and PARC.

    [Dataset reference](https://data.phmsociety.org/2021-phm-conference-data-challenge/)

    Parameters:

        train: Wether to obtain the train data provided
        models: Names of the models
    """
    def __init__(
        self,
        path: Path = DATASET_PATH,
        train: Optional[bool] = None,
    ):
        super().__init__()
        self.path = path
        LIVES_TABLE_PATH = path / "lives_data.pkl"
        if not (LIVES_TABLE_PATH).is_file():
            pr = CMAPSS2PreProcessor()
            pr.run()
        with open(LIVES_TABLE_PATH, "rb") as file:
            self.lives = pickle.load(file)

        if train is not None:
            self.lives = self.lives[self.lives["Train"] == train]

    def get_time_series(self, i):
        df_path = self.lives.iloc[i]["Output Dir"]
        df = pd.read_parquet(self.path / df_path)
        return df

    @property
    def n_time_series(self):
        return len(self.lives)

    @property
    def rul_column(self) -> str:
        return "RUL"
