import io
import numpy as np
import pandas as pd
from rul_pm import DATA_PATH
from rul_pm.dataset.lives_dataset import AbstractLivesDataset
from scipy.io import loadmat
from pathlib import Path

DATA_PATH = Path("/home/luciano/fuentes/lru_gcd_bearing/rul_pm/dataset/data/")
MFPT_PATH = DATA_PATH / "MFPT_Fault_Data_Sets"



ROLLER_DIAMETER = 0.235
PITCH_DIAMETER = 1.245
NUMBER_OF_ELEMENTS = 8
CONTACT_ANGLE = 0
SHAFT_RATE = 25

def BPFO():
    return 0.5*(NUMBER_OF_ELEMENTS * SHAFT_RATE)*(1- (ROLLER_DIAMETER/PITCH_DIAMETER * np.cos(CONTACT_ANGLE)))

def BPFI():
    return 0.5*(NUMBER_OF_ELEMENTS * SHAFT_RATE)*(1 + (ROLLER_DIAMETER/PITCH_DIAMETER * np.cos(CONTACT_ANGLE)))

def FTF():
    return 0.5*(SHAFT_RATE)*(1 - (ROLLER_DIAMETER/PITCH_DIAMETER * np.cos(CONTACT_ANGLE)))

def BSF():
    return 0.5*((PITCH_DIAMETER*SHAFT_RATE)/ROLLER_DIAMETER)*(1 - (ROLLER_DIAMETER/PITCH_DIAMETER * np.cos(CONTACT_ANGLE))**2)

def data_load(filename, label):
    """
    This function is mainly used to generate test data and training data.
    filename:Data location
    """
    data = loadmat(filename)["bearing"]
    data = {name: np.squeeze(data[0][0][i]) for i, (name, _) in enumerate(data.dtype.descr)}
    df =  pd.DataFrame(data)

    df['label'] = label
    df['time'] = np.arange(0, df.shape[0]) / df.sr
    df['filename'] = filename.stem
   
    return df

CLASS_NORMAL = 0
CLASS_OUTER_RACE_FAULT = 1
CLASS_INER_RACE_FAULT = 2

folders = [
    ("1 - Three Baseline Conditions", CLASS_NORMAL),
    ("2 - Three Outer Race Fault Conditions", CLASS_OUTER_RACE_FAULT),
    ("3 - Seven More Outer Race Fault Conditions", CLASS_OUTER_RACE_FAULT),
    ("4 - Seven Inner Race Fault Conditions", CLASS_INER_RACE_FAULT)
]


class MFPTDataset(AbstractLivesDataset):
    def __init__(self):
        self.lives = []
        mat_files = list((MFPT_PATH / folders[0][0]).glob("*.mat"))
        mat_files = sorted(mat_files, key=lambda x: x.stem)

        self.lives.append(data_load(mat_files[0], label=0))

        for folder, label in folders[1:]:
            mat_files = list((MFPT_PATH / folder).glob("*.mat"))
            mat_files = sorted(mat_files, key=lambda x: x.stem)
            for file in mat_files:
                self.lives.append(data_load(file, label=label))


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


    

