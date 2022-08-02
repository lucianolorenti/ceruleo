from collections import defaultdict
from typing import Dict, List, Optional

import antropy as ant
import numpy as np
import pandas as pd

from pyparsing import col
from scipy.stats import spearmanr
from temporis.dataset.transformed import TransformedDataset
from tqdm.auto import tqdm
from sklearn.feature_selection import mutual_info_regression
from uncertainties import ufloat


def entropy(s: np.ndarray):
    return ant.app_entropy(s)


def correlation(s: np.ndarray, y:Optional[np.ndarray]=None):
    N = s.shape[0]
    if np.all(s[0] == s):
        corr = spearmanr(s, np.arange(N), nan_policy="omit")
        corr = corr.correlation
    else:
        corr = np.nan
    return corr


def autocorrelation(s: np.ndarray):
    diff = np.diff(s)
    return np.sum(diff ** 2) / s.shape[0]


def monotonicity(s: np.ndarray):
    N = s.shape[0]
    diff = np.diff(s)
    return 1 / (N - 1) * np.abs(np.sum(diff > 0) - np.sum(diff < 0))


def n_unique(s: np.ndarray):
    return len(np.unique(s))

def null(s: np.ndarray):
    return np.sum(np.isfinite(s))

metrics = {
    "std": lambda x, y: np.std(x),
    "correlation": lambda x,y: correlation(x, y),
    "autocorrelation": lambda x,y:autocorrelation(x),
    "monotonicity": lambda x,y:monotonicity(x),
    "number_of_unique_elements": lambda x,y:n_unique(x),    
    'mutual_information': lambda x, y: mutual_info_regression(x.reshape(-1, 1), y),
    'null': lambda x, y: null(x)
}

# Mutual information
# Remaining Useful Life Prediction Using Ranking Mutual Information Based Monotonic Health Indicator


def analysis_single_time_series(
    X: np.ndarray,
    y: np.ndarray,
    column_names: List[str],
    data: Optional[Dict] = None,
    what_to_compute: List[str] = [],
):

    if data is None:
        data = defaultdict(lambda: defaultdict(list))
    if len(what_to_compute) == 0:
        what_to_compute = list(sorted(metrics.keys()))
    for column_index in range(X.shape[1]):
        column_name = column_names[column_index]
        for what in what_to_compute:
            x_ts = np.squeeze(X[:, column_index])

            m = metrics[what](x_ts, y)

            
            data[column_name][what].append(m)
    return data


def merge_analysis(data: dict):
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
    transformed_dataset: TransformedDataset,
    *,
    show_progress: bool,
    what_to_compute: List[str] = [],
):

    if len(what_to_compute) == 0:
        what_to_compute = list(sorted(metrics.keys()))
    data = defaultdict(lambda: defaultdict(list))
    iterator = transformed_dataset
    if show_progress:
        iterator = tqdm(iterator)

    for (X, y, _) in iterator:
        y = np.squeeze(y)
        data = analysis_single_time_series(
            X, y, transformed_dataset.transformer.column_names, data, what_to_compute
        )
    return merge_analysis(data)
