import numpy as np
import pandas as pd
from rul_pm import DATA_PATH
from rul_pm.dataset.lives_dataset import AbstractLivesDataset

MFPT_PATH = DATA_PATH / "MFPT_Fault_Data_Sets"








class MFPTDataset(AbstractLivesDataset):
    def __init__(self):
        pass



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
