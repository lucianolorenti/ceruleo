import numpy as np
import pandas as pd
from rul_pm import DATA_PATH
from rul_pm.dataset.lives_dataset import AbstractLivesDataset
from scipy.io import loadmat
from pathlib import Path

DATA_PATH = Path("/home/luciano/fuentes/lru_gcd_bearing/rul_pm/dataset/data/")
MFPT_PATH = DATA_PATH / "MFPT_Fault_Data_Sets"

signal_size = 1024




def data_load(filename, label):
    """
    This function is mainly used to generate test data and training data.
    filename:Data location
    """
    if label == 0:
        fl = loadmat(filename)["bearing"][0][0][1]  # Take out the data
    else:
        fl = loadmat(filename)["bearing"][0][0][2]  # Take out the data

    return pd.DataFrame({"gs": np.squeeze(fl)})


folders = [
    "1 - Three Baseline Conditions",
    #    "2 - Three Outer Race Fault Conditions"
    "3 - Seven More Outer Race Fault Conditions",
    "4 - Seven Inner Race Fault Conditions",
]


class MFPTDataset(AbstractLivesDataset):
    def __init__(self):
        self.lives = []
        mat_files = list((MFPT_PATH / folders[0]).glob("*.mat"))
        mat_files = sorted(mat_files, key=lambda x: x.stem)

        self.lives.append(data_load(mat_files[0], label=0))

        label = 1
        for folder in folders[1:]:
            mat_files = list((MFPT_PATH / folders[1]).glob("*.mat"))
            mat_files = sorted(mat_files, key=lambda x: x.stem)
            for file in mat_files:
                self.lives.append(data_load(file, label=label))
                label += 1

    def get_life(self, i):
        """

        Returns
        -------
        pd.DataFrame
            DataFrame with the data of the life i
        """
        return self.lives[i]

    @property
    def nlives(self):
        return len(self.lives)

