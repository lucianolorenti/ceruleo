from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import antropy as ant
import numpy as np
from pydantic import BaseModel
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from tqdm.auto import tqdm

from ceruleo.dataset.transformed import TransformedDataset
from ceruleo.dataset.ts_dataset import AbstractPDMDataset
from ceruleo.dataset.utils import iterate_over_features_and_target
import pandas as pd


class MetricType(str, Enum):
    std = "std"
    correlation = "correlation"
    autocorrelation = "autocorrelation"
    monotonicity = "monotonicity"
    number_of_unique_elements = "number_of_unique_elements"
    mutual_information = "mutual_information"
    null = "null"
    entropy = "entropy"

    @staticmethod
    def from_str(s: str) -> "MetricType":
        return MetricType(s)


class MetricValuesSummary(BaseModel):
    mean: float
    std: float
    max: float
    min: float


class NumericalFeaturesAnalysis(BaseModel):
    feature: str
    metric: Dict[MetricType, List[float]]

    def summarize(self) -> Dict[MetricType, MetricValuesSummary]:
        out = {}
        for metric in self.metric.keys():
            mean = np.nanmean(self.metric[metric])
            std = np.nanstd(self.metric[metric])
            max_ = np.nanmax(self.metric[metric])
            min_ = np.nanmin(self.metric[metric])
            out[metric] = MetricValuesSummary(mean=mean, std=std, max=max_, min=min_)
        return out
    
    def __getitem__(self, key: str) -> MetricValuesSummary:
        return self.metric[MetricType.from_str(key)]
    
    def _repr_html_(self) -> str:
        out = "<table>"
        out += "<tr><th>Metric</th><th>Mean</th><th>Std</th><th>Max</th><th>Min</th></tr>"
        for metric, summary in self.summarize().items():
            out += f"<tr><td>{metric}</td><td>{summary.mean}</td><td>{summary.std}</td><td>{summary.max}</td><td>{summary.min}</td></tr>"
        out += "</table>"
        return out


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
    Null percentage for a given feature

    Parameters:
        s: A feature

    Returns:
        Null percentage
    """
    return np.mean(~np.isfinite(s)) * 100


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
   
    "autocorrelation": lambda x, y: autocorrelation(x),
    "monotonicity": lambda x, y: monotonicity(x),
    "number_of_unique_elements": lambda x, y: n_unique(x),

    "null": lambda x, y: null(x),
    "entropy": lambda x, y: entropy(x),
    "mutual_information": mutual_information,
     "correlation": lambda x, y: correlation(x, y),
}


def analyze_single_cycle(
    X: np.ndarray,
    y: np.ndarray,
    out: Dict[str, Dict[MetricType, List[float]]],
    column_names: List[str],
    what_to_compute: List[str] = [],
):
    """
    Compute the analysis for a single run-to-failure cycle



    Parameters:
        X: Input Features
        y: RUL Target
        column_names: Column names of the features
        data: Initial data
        what_to_compute: Features to compute

    Returns:
        A dictionary with the analysis of the features

    """

    if len(what_to_compute) == 0:
        what_to_compute = list(sorted(metrics.keys()))
    for column_index in range(len(column_names)):
        column_name = column_names[column_index]
        for what in what_to_compute:
            x_ts = np.squeeze(X.loc[:, column_name].values)

            m = metrics[what](x_ts, y)
            metric_type = MetricType.from_str(what)
            out[column_name][metric_type].append(m)

    return out


def merge_cycle_analysis(
    data: Dict[str, Dict[MetricType, List[float]]],
) -> Dict[str, NumericalFeaturesAnalysis]:
    out = {k: NumericalFeaturesAnalysis(feature=k, metric={}) for k in data.keys()}
    for column_name in data.keys():
        for what in data[column_name]:
            metric_type = MetricType.from_str(what)
            out[column_name].metric[metric_type] = data[column_name][what]
    return out


def analyze(
    dataset: Union[TransformedDataset, AbstractPDMDataset],
    *,
    show_progress: bool = False,
    what_to_compute: List[str] = [],
) -> NumericalFeaturesAnalysis:
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
        NumericalFeaturesAnalysis
    """

    if len(what_to_compute) == 0:
        what_to_compute = list(sorted(metrics.keys()))
    iterator = dataset
    if show_progress:
        iterator = tqdm(iterator)

    if isinstance(dataset, TransformedDataset):
        column_names = dataset.transformer.column_names
    else:
        column_names = dataset.numeric_features()

    data_per_cycle = {
        k: {MetricType.from_str(what): [] for what in what_to_compute}
        for k in column_names
    }
    for X, y in iterate_over_features_and_target(dataset):
        y = np.squeeze(y)
        analyze_single_cycle(X, y, data_per_cycle, column_names, what_to_compute)

    return merge_cycle_analysis(data_per_cycle)


def analyze_as_dataframe(
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
        pd.DataFrame
    """
    rr = analyze(dataset, show_progress=show_progress, what_to_compute=what_to_compute)
        
   
    out: Dict[Tuple[str, str], List[float]] = {}

  
    for k, metrics in rr.items():
        metrics_summary = metrics.summarize()
        for metric_name, metric_values in metrics_summary.items():
            
            key_mean = (metric_name.value, "Mean value across the cycles")
            key_std = (metric_name.value, "Standard deviation across the cycles")
            key_max = (metric_name.value, "Maximum value found in a cycle")
            key_min = (metric_name.value, "Minimum value found in a cycle")
            
            if key_mean not in out:
                out[key_mean] = []
                out[key_std] = []
                out[key_max] = []
                out[key_min] = []

            out[key_mean].append(metric_values.mean)
            out[key_std].append(metric_values.std)
            out[key_max].append(metric_values.max)
            out[key_min].append(metric_values.min)

    return pd.DataFrame(out, index=rr.keys())
