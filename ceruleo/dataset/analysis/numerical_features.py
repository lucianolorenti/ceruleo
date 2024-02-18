from collections import defaultdict
from typing import Dict, List, Optional, Union

import antropy as ant
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from tqdm.auto import tqdm
from uncertainties import ufloat

from ceruleo.dataset.transformed import TransformedDataset
from ceruleo.dataset.ts_dataset import AbstractPDMDataset
from ceruleo.dataset.utils import iterate_over_features_and_target


def entropy(s: np.ndarray) -> float:
    """
    Approximate entropy

    The approximate entropy quantifies the amount of regularity and the unpredictability of fluctuations over time-series data.

    Parameters:
        s: A single feature

    Returns:
        Approximate entropy of feature s
    """
    return ant.app_entropy(s)


def correlation(s: np.ndarray, y: Optional[np.ndarray] = None) -> float:
    """
    Correlation of the feature with the target

    Parameters:
        s: A single feature
        y: The RUL target

    Returns:
        Correlation between the feature and the RUL target
    """
    N = s.shape[0]
    if not (s[0] == s).all():
        corr = spearmanr(s, np.arange(N), nan_policy="omit")
        corr = corr.correlation
    else:
        corr = np.nan
    return corr


def autocorrelation(s: np.ndarray) -> float:
    """
    Autocorrelation of a feature

    Parameters:
        s: A single feature

    Returns:
        Autocorrelation of the feature
    """
    diff = np.diff(s)
    return np.sum(diff**2) / s.shape[0]


def monotonicity(s: np.ndarray) -> float:
    """
    Monotonicity of a feature, the two extreme values are 0 if the feature is constant and 1 if it is strictly monotonic.

    Parameters:
        s: A single feature

    Returns:
        Monotonicity of the feature
    """
    N = s.shape[0]
    diff = np.diff(s)
    return 1 / (N - 1) * np.abs(np.sum(diff > 0) - np.sum(diff < 0))


def n_unique(s: np.ndarray) -> int:
    """
    Number of unique values in the array

    Parameters:
        s: A single feature

    Returns:
        Number of unique values
    """
    return len(np.unique(s))


def null(s: np.ndarray) -> float:
    """
    Null proportion for a given feature

    Parameters:
        s: A feature

    Returns:
        Null proportion
    """
    return np.mean(~np.isfinite(s))


def mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Mutual information between a feature and the target

    [Reference](Remaining Useful Life Prediction Using Ranking Mutual Information Based Monotonic Health Indicator)

    Parameters:
        x: A single feature
        y: RUL Target

    Returns:
        Mutual information between x and y

    """
    x = x.reshape(-1, 1)
    x = np.nan_to_num(x)
    return mutual_info_regression(x, y)


metrics = {
    "std": lambda x, y: np.std(x),
    "correlation": lambda x, y: correlation(x, y),
    "autocorrelation": lambda x, y: autocorrelation(x),
    "monotonicity": lambda x, y: monotonicity(x),
    "number_of_unique_elements": lambda x, y: n_unique(x),
    "mutual_information": mutual_information,
    "null": lambda x, y: null(x),
    "entropy": lambda x, y: entropy(x),
}


def analysis_single_time_series(
    X: np.ndarray,
    y: np.ndarray,
    column_names: List[str],
    data: Optional[Dict] = None,
    what_to_compute: List[str] = [],
) -> dict:
    """
    Compute the analysis for a single run-to-failure cycle

    Parameters:
        X: Input Features
        y: RUL Target
        column_names: Column names of the features
        data: Initial data
        what_to_compute: Features to compute

    Returns:
        Dictionary containing the computed info
    """

    if data is None:
        data = defaultdict(lambda: defaultdict(list))
    if len(what_to_compute) == 0:
        what_to_compute = list(sorted(metrics.keys()))
    for column_index in range(len(column_names)):
        column_name = column_names[column_index]
        for what in what_to_compute:
            x_ts = np.squeeze(X.loc[:, column_name].values)

            m = metrics[what](x_ts, y)

            data[column_name][what].append(m)
    return data


def merge_analysis(data: dict) -> pd.DataFrame:
    data_df = defaultdict(lambda: defaultdict(list))
    for column_name in data.keys():
        for what in data[column_name]:
            data_df[column_name][f"{what} Mean"] = ufloat(
                np.nanmean(data[column_name][what]),
                np.nanstd(data[column_name][what]),
            )
            data_df[column_name][f"{what} Max"] = np.nanmax(data[column_name][what])
            data_df[column_name][f"{what} Min"] = np.nanmin(data[column_name][what])
    return pd.DataFrame(data_df).T


def analysis(
    dataset: Union[TransformedDataset, AbstractPDMDataset],
    *,
    show_progress: bool = False,
    what_to_compute: List[str] = [],
) -> pd.DataFrame:
    """
    Compute analysis of numerical features

    Parameters:
        dataset: A transformed dataset with features and target
        show_progress: Wether to show the progress when computing the features
        what_to_compute: Elements available to compute:

            - std
            - Correlation
            - Autocorrelation
            - Monotonicity
            - Number of unique elements
            - Mutual information
            - Null
            - Entropy


    Returns:
        Dataframe with the columns specified by what_to_compute
    """

    if len(what_to_compute) == 0:
        what_to_compute = list(sorted(metrics.keys()))
    data = defaultdict(lambda: defaultdict(list))
    iterator = dataset
    if show_progress:
        iterator = tqdm(iterator)

    if isinstance(dataset, TransformedDataset):
        column_names = dataset.transformer.column_names
    else:
        column_names = dataset.numeric_features()
    for X, y in iterate_over_features_and_target(dataset):
        y = np.squeeze(y)
        data = analysis_single_time_series(X, y, column_names, data, what_to_compute)
    return merge_analysis(data)
