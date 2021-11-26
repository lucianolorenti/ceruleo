

from itertools import combinations
from typing import List, Optional, Tuple

import pandas as pd
from ceruleo.dataset.ts_dataset import AbstractTimeSeriesDataset
from ceruleo.dataset.utils import iterate_over_features


def correlation_analysis(
    dataset: AbstractTimeSeriesDataset,
    corr_threshold: float = 0.7,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Correlation Analysis
    Compute the correlation between all the features given an Iterable of executions.

    Parameters:
    
        dataset: Dataset of time series
        corr_threshold: Threshold to consider two features of a single execution highly correlated
        features: List of features to consider when computing the correlations

    Returns:

        pd.DataFrame: A DataFrame indexed with the column names with the following columns:

                    - Mean Correlation
                    - Std Correlation
                    - Percentage of lives with a high correlation
                    - Abs mean correlation
                    - Std mean correlation
                    - Max correlation
                    - Min correlation
    """
    if features is None:
        features = sorted(list(dataset.common_features()))
    else:
        features = sorted(list(set(features).intersection(dataset.common_features())))
    features = dataset.get_features_of_life(0)[features].corr().columns
    correlated_features = []
    
    for ex in iterate_over_features(dataset):
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