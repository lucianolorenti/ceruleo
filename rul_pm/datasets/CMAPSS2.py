import gzip
import logging
import pickle
import zipfile
from pathlib import Path
from typing import Optional, Union

import gc
import h5py
import numpy as np
import pandas as pd
from rul_pm import DATASET_PATH
from rul_pm.utils.download import download_file
from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

DATASET_PATH = DATASET_PATH / "C_MAPSS2"
DATASET_PATH.mkdir(exist_ok=True, parents=True)
URL = "http:"


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
        self, data, file_h5: Path, i: int, unit: int, train: bool
    ):
        unit = int(unit)
        output_file = f"LIFE_{file_h5.name}_{unit}_{i}.pkl.gz"
        number_of_samples = data.shape[0]
        with gzip.open(self.path / output_file, "wb") as file:
            pickle.dump(data, file)
        return {
            "Raw File": file_h5.name,
            "Unit": unit,
            "Index": i,
            "Output File": output_file,
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
        except OSError:
            logger.error(f"Cannot open file {file_h5}")
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
                self.download_file(URL)
            logger.info("Unzipping")
            with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
                zip_ref.extractall(self.path)
            logger.info("Removing zip file")
            # ZIP_FILE.unlink()

    def run(self):
        if not (self.lives_table_path).is_file():
            self.obtain_raw_files()
        self.process_raw_files()

    def download(self, URL):
        logger.info("Downloading file")
        pass


class CMAPSS2Dataset(AbstractTimeSeriesDataset):
    def __init__(
        self,
        path: Path = DATASET_PATH,
        train: Optional[bool] = None,
    ):
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
        """

        Returns
        -------
        pd.DataFrame
            DataFrame with the data of the life i
        """
        df_path = self.lives.iloc[i]["Output file"]
        with gzip.open(self.path / df_path, "rb") as file:
            return pickle.load(file)

    @property
    def n_time_series(self):
        return len(self.lives)
