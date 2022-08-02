
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset
import antropy as ant
from itertools import combinations


def correlation_analysis(
    dataset: AbstractTimeSeriesDataset,
    corr_threshold: float = 0,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Correlation Analysis
    Compute the correlation between all the features given an Iterable of executions.
    Parameters
    ---------
    dataset: AbstractTimeSeriesDataset
        Dataset of time series
    corr_threshold: float
        Treshold to consider two features of a single execution higly correlated
    features: Optional[List[str]], default None
        List of features to consider when computing the correlations
    Returns
    -------
    pd.DataFrame:
    * A DataFrame with three columns:
        * Feature name 1
        * Feature name 2
        * Percentage of time-series with a high correlation
        * Mean correlation across the time-series
        * Std correlation across the time-series
        * Mean Abs correlation across the time-series
        * Std Abs correlation across the time-series
        * Max correlation across the time-series
        * Min correlation across the time-series
    """
    if features is None:
        features = sorted(list(dataset.common_features()))
    else:
        features = sorted(list(set(features).intersection(dataset.common_features())))
    features = dataset[0][features].corr().columns
    correlated_features = []
    for ex in dataset:
        ex = ex[features]
        corr_m = ex.corr().fillna(0)

        correlated_features_for_execution = []

        for f1, f2 in combinations(features, 2):
            if f1 == f2:
                continue

            correlated_features_for_execution.append((f1, f2, corr_m.loc[f1, f2]))

        correlated_features.extend(correlated_features_for_execution)

    df = pd.DataFrame(correlated_features, columns=["Feature 1", "Feature 2", "Corr"])
    output = df.groupby(by=["Feature 1", "Feature 2"]).mean()
    output.rename(columns={"Corr": "Mean Correlation"}, inplace=True)
    output["Std Correlation"] = df.groupby(by=["Feature 1", "Feature 2"]).std()

    def percentage_above_treshold(x):
        return (x["Corr"].abs() > corr_threshold).mean() * 100

    output["Percentage of lives with a high correlation"] = df.groupby(
        by=["Feature 1", "Feature 2"]
    ).apply(percentage_above_treshold)

    output["Abs mean correlation"] = df.groupby(by=["Feature 1", "Feature 2"]).apply(
        lambda x: x.abs().mean()
    )
    output["Std mean correlation"] = df.groupby(by=["Feature 1", "Feature 2"]).apply(
        lambda x: x.abs().std()
    )
    output["Max correlation"] = df.groupby(by=["Feature 1", "Feature 2"]).max()
    output["Min correlation"] = df.groupby(by=["Feature 1", "Feature 2"]).min()
    return output
