import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset
import antropy as ant
from itertools import combinations


def common_features_null_proportion_below(dataset: AbstractTimeSeriesDataset, t: float):
    """
    Return the list of features such that each feature in each
    life have a null proportion smaller than the threshold
    """

    def null_prop_all_below_threshold(c, t):
        return np.all(np.array(c) < t)

    _, null_proportion_per_life = null_proportion(dataset)

    return [
        column
        for column in null_proportion_per_life.keys()
        if null_prop_all_below_threshold(null_proportion_per_life[column], t)
    ]


def null_proportion(dataset: AbstractTimeSeriesDataset, features:Optional[List[str]] = None):
    """
    Return mean and max null proportion for each column of each life of the dataset

    Parameters
    ----------
    dataset: AbstractTimeSeriesDataset

    Return
    ------
    pd.DataFrame: Dataframe that contains three columns
                  ['Feature', 'Max Null Proportion', 'Mean Null Proportion']

    dict: string -> list
          The key is the column name and the value is the list of null proportion
          for each life
    """
    common_features = dataset.common_features()
    if features:
        common_features = set(common_features).intersection(set(features))

    null_proportion_per_life = {}
    for life in dataset:
        d = life.isnull().mean().to_dict()
        for column in common_features:
            null_proportion_list = null_proportion_per_life.setdefault(column, [])
            null_proportion_list.append(d[column])

    for column in null_proportion_per_life.keys():
        null_proportion_per_life[column] = np.array(null_proportion_per_life[column])

    data = [
        (
            column,
            np.max(null_proportion_per_life[column]),
            np.mean(null_proportion_per_life[column]),
            np.sum(null_proportion_per_life[column] > 0.8),
        )
        for column in null_proportion_per_life.keys()
    ]

    df = pd.DataFrame(
        data,
        columns=[
            "Feature",
            "Max Null Proportion",
            "Mean Null Proportion",
            "Number of lives with more than 80% missing",
        ],
    )
    df.sort_values(by="Max Null Proportion", inplace=True, ascending=False)
    return df, null_proportion_per_life


def null_proportion_per_life(dataset: AbstractTimeSeriesDataset):
    """"""
    data = []
    for i, life in enumerate(dataset):
        null_prop = life.isnull().mean()
        number_of_completely_null = np.sum(null_prop > 0.99999)
        number_of_half_null = np.sum(null_prop > 0.5)
        number_of_25p_null = np.sum(null_prop > 0.25)
        mean_null_proportion = null_prop.mean().mean()
        data.append(
            (
                i,
                life.shape[1],
                number_of_completely_null,
                number_of_half_null,
                number_of_25p_null,
                mean_null_proportion,
            )
        )
    df = pd.DataFrame(
        data,
        columns=[
            "Life",
            "Number of features",
            "Number of completely null features",
            "N of features with 50% null",
            "N of features with 25% null",
            "Mean null propotion",
        ],
    )
    df.sort_values(
        by="Number of completely null features", inplace=True, ascending=False
    )
    return df
