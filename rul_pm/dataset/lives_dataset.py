import numpy as np
import pandas as pd
from tqdm.auto import tqdm


class AbstractLivesDataset:

    def get_life(self, i: int):
        """

        Returns
        -------
        pd.DataFrame
            DataFrame with the data of the life i
        """
        raise NotImplementedError

    def __getitem__(self, i: int):
        df = self.get_life(i)
        df['life'] = i
        return df

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
        common_features = self.commonFeatures()
        for i in tqdm(range(self.nlives)):
            if proportion < 1.0 and np.random.rand() > proportion:
                continue
            current_life = self[i][common_features]
            df.append(current_life)
        return pd.concat(df)

    @property
    def rul_column(self):
        """
        Return
        ------
        str:
            The name of the RUL column
        """
        raise NotImplementedError

    def commonFeatures(self):
        f = []
        for i in tqdm(range(self.nlives)):
            life = self[i]
            f.append(set(life.columns.values))
        return f[0].intersection(*f)
