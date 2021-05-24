import logging

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from rul_pm.graphics.control_charts import plot_ewma_

logger = logging.getLogger(__name__)


def common_features_null_proportion_below(dataset, t):
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


def null_proportion(dataset):
    """
    Return mean and max null proportion for each column of each life of the dataset

    Parameters
    ----------
    dataset: AbstractLivesDataset

    Return
    ------
    pd.DataFrame: Dataframe that contains three columns
                  ['Feature', 'Max Null Proportion', 'Mean Null Proportion']

    dict: string -> list
          The key is the column name and the value is the list of null proportion
          for each life
    """
    comon_features = [set(life.columns.tolist()) for life in dataset]
    comon_features = comon_features[0].intersection(*comon_features)

    null_proportion_per_life = {}
    for life in dataset:
        d = life.isnull().mean().to_dict()
        for column in comon_features:
            null_proportion_list = null_proportion_per_life.setdefault(column, [])
            null_proportion_list.append(d[column])

    data = [
        (
            column,
            np.max(null_proportion_per_life[column]),
            np.mean(null_proportion_per_life[column]),
        )
        for column in null_proportion_per_life.keys()
    ]

    df = pd.DataFrame(
        data, columns=["Feature", "Max Null Proportion", "Mean Null Proportion"]
    )
    df.sort_values(by="Max Null Proportion", inplace=True, ascending=False)
    return df, null_proportion_per_life


def null_proportion_per_life(dataset):
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


def variance_information(dataset):
    """
    Return mean and max null proportion for each column of each life of the dataset

    Parameters
    ----------
    dataset: AbstractLivesDataset

    Return
    ------
    pd.DataFrame: Dataframe that contains three columns
                  ['Feature', 'Max Null Proportion', 'Mean Null Proportion']

    dict: string -> list
          The key is the column name and the value is the list of null proportion
          for each life
    """
    comon_features = [set(life.columns.tolist()) for life in dataset]
    comon_features = comon_features[0].intersection(*comon_features)

    std_per_life = {}
    for life in dataset:
        d = life.std().to_dict()
        for column in comon_features:
            if column not in d:
                continue
            if not isinstance(d[column], float):
                continue
            std_list = std_per_life.setdefault(column, [])
            std_list.append(d[column])

    data = [
        (column, np.min(std_per_life[column]), np.mean(std_per_life[column]))
        for column in std_per_life.keys()
    ]

    df = pd.DataFrame(
        data, columns=["Feature", "Min std Proportion", "Mean std Proportion"]
    )
    df.sort_values(by="Min std Proportion", inplace=True, ascending=True)
    return df, std_per_life
