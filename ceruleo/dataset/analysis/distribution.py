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


def histogram_per_life(
    life: pd.DataFrame,
    feature: str,
    bins_to_use: np.ndarray,
    normalize: bool = True,
) -> List[np.ndarray]:
    try:
        d = life[feature]
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

            - Life 1: Run-to-failure cycle 1
            - Life 2: Run-to-failure cycle 2
            - W: Wasserstein
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
                histogram_per_life(life, feature, features_bins[feature])
            )

    df_data = []
    for feature in columns:
        data = {}
        for (i, h1), (j, h2) in itertools.combinations(
            enumerate(histograms[feature]), 2
        ):
            kl = (np.mean(kl_div(h1, h2)) + np.mean(kl_div(h2, h1))) / 2
            wd = wasserstein_distance(h1, h2)
            df_data.append((i, j, wd, kl, feature))
    df = pd.DataFrame(df_data, columns=["Life 1", "Life 2", "W", "KL", "feature"])

    return df
