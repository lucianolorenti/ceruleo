from itertools import combinations
from typing import Dict, List, Optional, Tuple

import pandas as pd
from ceruleo.dataset.ts_dataset import AbstractPDMDataset
from ceruleo.dataset.utils import iterate_over_features
from pydantic import BaseModel

from ceruleo.utils import pydantic_to_dict


class CorrelationAnalysisElement(BaseModel):
    mean_correlation: float
    std_correlation: float
    max_correlation: float
    min_correlation: float
    abs_mean_correlation: float
    std_abs_mean_correlation: float


class CorrelationAnalysis(BaseModel):
    data: Dict[Tuple[str, str], CorrelationAnalysisElement]

    def get(self, feature_1: str, feature_2: str) -> CorrelationAnalysisElement:
        needle = (feature_1, feature_2)
        if needle not in self.data:
            needle = (feature_2, feature_1)

        if needle not in self.data:
            raise KeyError(f"Correlation between {feature_1} and {feature_2} not found")
        return self.data[needle]

    def to_pandas(self) -> pd.DataFrame:
        return (
            pd.DataFrame.from_dict(
                {(k[0], k[1]): pydantic_to_dict(v) for k, v in self.data.items()},
                orient="index",
            )
            .reset_index()
            .rename(columns={"level_0": "feature_1", "level_1": "feature_2"})
        )


def correlation_analysis(
    dataset: AbstractPDMDataset,
    features: Optional[List[str]] = None,
) -> CorrelationAnalysis:
    """
    Correlation Analysis
    Compute the correlation between all the features given an Iterable of executions.

    Parameters:
        dataset: Dataset of time series
        features: List of features to consider when computing the correlations

    Returns:
        A CorrelationAnalysis object with map indexed by two colun names and the following information:s

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
            correlated_features_for_execution.append((f1, f2, corr_m.loc[f1, f2]))

        correlated_features.extend(correlated_features_for_execution)

    df = pd.DataFrame(correlated_features, columns=["Feature 1", "Feature 2", "Corr"])
    output = df.groupby(by=["Feature 1", "Feature 2"]).agg(
        {
            "Corr": [
                "mean",
                "std",
                "max",
                "min",
            ]
        }
    )

    # Calculate additional statistics
    output["Abs mean correlation"] = df.groupby(by=["Feature 1", "Feature 2"])[
        "Corr"
    ].apply(lambda x: x.abs().mean())
    output["Std abs mean correlation"] = df.groupby(by=["Feature 1", "Feature 2"])[
        "Corr"
    ].apply(lambda x: x.abs().std())

    output.columns = [
        "mean_correlation",
        "std_correlation",
        "max_correlation",
        "min_correlation",
        "abs_mean_correlation",
        "std_abs_mean_correlation",
    ]

    output = output.fillna(0)
    return CorrelationAnalysis(
        data={
            (k[0], k[1]): CorrelationAnalysisElement(
                mean_correlation=v["mean_correlation"],
                std_correlation=v["std_correlation"],
                max_correlation=v["max_correlation"],
                min_correlation=v["min_correlation"],
                abs_mean_correlation=v["abs_mean_correlation"],
                std_abs_mean_correlation=v["std_abs_mean_correlation"],
            )
            for k, v in output.iterrows()
        }
    )
