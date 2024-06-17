import itertools
import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from ceruleo.dataset.ts_dataset import AbstractPDMDataset
from ceruleo.dataset.utils import iterate_over_features
from scipy.special import kl_div
from scipy.stats import wasserstein_distance
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def histogram_per_cycle(
    cycle: pd.DataFrame,
    feature: str,
    bins_to_use: np.ndarray,
    normalize: bool = True,
) -> List[np.ndarray]:
    """Compute the histogram of a feature in a run-to-failure cycle

    Args:
        cycle (pd.DataFrame): The run-to-failure cycle
        feature (str): The  feature to compute the histogram
        bins_to_use (np.ndarray): Number of bins to use
        normalize (bool, optional): Wheter to normalize the histogram. Defaults to True.

    Returns:
        List[np.ndarray]: The histogram of the feature
    """
    try:
        d = cycle[feature]
        h, _ = np.histogram(d, bins=bins_to_use)

        if normalize:
            h = h / np.sum(h)
            h += 1e-50
        return h
    except Exception as e:
        logger.info(f"Error {e} when computing the distribution for feature {feature}")


def compute_bins(ds: AbstractPDMDataset, feature: str, number_of_bins: int = 15):
    min_value = ds.get_features_of_life(0)[feature].min()
    max_value = ds.get_features_of_life(0)[feature].max()

    for life in iterate_over_features(ds):
        min_value = min(np.min(life[feature]), min_value)
        max_value = max(np.max(life[feature]), max_value)
    bins_to_use = np.linspace(min_value, max_value, number_of_bins + 1)
    return bins_to_use


def features_divergeces(
    ds: AbstractPDMDataset,
    number_of_bins: int = 15,
    columns: Optional[List[str]] = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Compute the divergence between features

    Parameters:
        ds: The dataset
        number_of_bins: Number of bins
        columns: Which columns to use

    Returns:
        A DataFrame in which each row contains the distances between a feature of two run-to-failure cycle with the following columns:

            - Cycle 1: Run-to-failure cycle 1
            - Cycle 2: Run-to-failure cycle 2
            - Wasserstein: Wasserstein
            - KL: KL Divergence
            - feature: The feature name
    """
    if columns is None:
        columns = ds.numeric_features()

    features_bins = {}
    iterator = tqdm(columns) if show_progress else columns

    for feature in iterator:
        features_bins[feature] = compute_bins(ds, feature, number_of_bins)

    histograms = {}
    for life in iterate_over_features(ds):
        for feature in columns:
            if feature not in histograms:
                histograms[feature] = []
            histograms[feature].append(
                histogram_per_cycle(life, feature, features_bins[feature])
            )

    df_data = []
    for feature in columns:
        data = {}
        for (i, h1), (j, h2) in itertools.combinations(
            enumerate(histograms[feature]), 2
        ):
            kl = (np.mean(kl_div(h1, h2)) + np.mean(kl_div(h2, h1))) / 2
            wd = wasserstein_distance(h1, h2)
            df_data.append(
                (
                    i,
                    j,
                    ds.get_features_of_life(i).shape[0],
                    ds.get_features_of_life(j).shape[0],
                    abs(ds.get_features_of_life(i).shape[0]-ds.get_features_of_life(j).shape[0]),
                    wd,
                    kl,
                    feature,
                )
            )
    df = pd.DataFrame(
        df_data,
        columns=[
            "Cycle 1",
            "Cycle 2",
            "Cycle 1 length",
            "Cycle 2 length",
            "Abs Length difference",           
            "Wasserstein",
            "KL",
            "feature",
        ],
    )

    return df
