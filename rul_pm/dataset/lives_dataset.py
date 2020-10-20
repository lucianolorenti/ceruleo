import numpy as np
import pandas as pd

class AbstractLivesDataset:

    def __getitem__(self, i:int):
        """

        Returns
        -------
        pd.DataFrame
            DataFrame with the data of the life i
        """
        raise NotImplementedError

    @property
    def nlives(self):
        """
        Return
        ------
        int:
            The number of lives in the dataset
        """
        raise NotImplementedError

    def __len__(self):
        """
        Return
        ------
        int: 
            The number of lives in the dataset
        """
        return self.nlives

    def toPandas(self, proportion=1.0):
        """
        Create a dataset with the lives concatenated

        Parameters
        ----------
        proportion: float
                    Proportion of lives to use.

        Returns
        -------
        
        pd.DataFrame:
            Return a DataFrame with all the lives concatenated
        """
        df = []
        for i in range(self.nlives):
            if proportion < 1.0 and np.random.rand() > proportion:
                continue
            df.append(self[i])
        return pd.concat(df)

