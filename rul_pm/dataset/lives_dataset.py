from collections.abc import Iterable
from typing import List, Tuple, Union

import numpy as np
import pandas as pd


class AbstractLivesDataset:

    def get_life(self, i: int) -> pd.DataFrame:
        """

        Returns
        -------
        pd.DataFrame
            DataFrame with the data of the life i
        """
        raise NotImplementedError

    @property
    def rul_column(self) -> str:
        """
        Return
        ------
        str:
            The name of the RUL column
        """
        raise NotImplementedError

    @property
    def nlives(self) -> int:
        """
        Return
        ------
        int:
            The number of lives in the dataset
        """
        raise NotImplementedError

    def __getitem__(self, i: Union[int, Iterable]):
        if isinstance(i, Iterable):
            return FoldedDataset(self, i)
        else:
            df = self.get_life(i)
            df['life'] = i
            return df

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.nlives, 1)

    def __len__(self):
        """
        Return
        ------
        int:
            The number of lives in the dataset
        """
        return self.nlives

    def toPandas(self, proportion=1.0) -> pd.DataFrame:
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
        for i in range(self.nlives):
            if proportion < 1.0 and np.random.rand() > proportion:
                continue
            current_life = self[i][common_features]
            df.append(current_life)
        return pd.concat(df)

    def commonFeatures(self) -> List[str]:
        f = []
        for i in range(self.nlives):
            life = self[i]
            f.append(set(life.columns.values))
        return f[0].intersection(*f)


class FoldedDataset(AbstractLivesDataset):
    def __init__(self, dataset: AbstractLivesDataset, indices: list):
        self.dataset = dataset
        self.indices = indices

    @property
    def nlives(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        """

        Returns
        -------
        pd.DataFrame
            DataFrame with the data of the life i
        """
        return self.dataset[self.indices[i]]
