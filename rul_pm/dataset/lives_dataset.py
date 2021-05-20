"""The Dataset module provides a light interface to define a PM Dataset
"""
from collections.abc import Iterable
from typing import List, Tuple, Union

import numpy as np
import pandas as pd


class AbstractLivesDataset:
    """Base class of the dataset handled by this library.

        Methods for fitting and transform receives an instance
        that inherit from this class
    """
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

    def durations(self) -> List[float]:
        """Obtain the length of each life

        Returns
        -------
        List[float]
            List of durations
        """
        return [
            life[self.rul_column].iloc[0] for life in self
        ]

    def __getitem__(self, i: Union[int, Iterable]):
        """Obtain a life or an splice of the dataset using a FoldedDataset

        Parameters
        ----------
        i: Union[int, Iterable]
           If the paramter is an in it will return a pd.DataFrame with the i-th lfie
           If the parameter is a list of int it will return a FoldedDataset
           with the lifes whose id are present in the lsit


        Raises
        ------
        ValueError: WHen the list does not contain integer parameters

        Returns:
            pd.DataFrame: the i-th life
            FoldedDataset: The dataset with the lives specified by the list
        """
        if isinstance(i, Iterable):
            if not all(isinstance(item, (int, np.integer)) for item in i):
                raise ValueError('Invalid iterable index passed')

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
