import logging
from itertools import product
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import kl_div
from scipy.stats import wasserstein_distance
from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset
from tqdm.auto import  tqdm 


logger = logging.getLogger(__name__)


def histogram_per_life(
    life:pd.DataFrame,
    feature:str,
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
        logger.info(
            f"Error {e} when computing the distribution for feature {feature}"
        )
    




def compute_bins(ds: AbstractTimeSeriesDataset, feature:str, number_of_bins: int = 15):
    min_value = ds[0][feature].min()
    max_value = ds[0][feature].max()
    for life in ds:
        min_value = min(np.min(life[feature]), min_value)
        max_value = max(np.max(life[feature]), max_value)
    bins_to_use = np.linspace(min_value, max_value, number_of_bins + 1)
    return bins_to_use

def features_divergeces(
    ds: AbstractTimeSeriesDataset, number_of_bins: int = 15, columns:Optional[List[str]] = None
) -> Tuple[pd.DataFrame, dict]:
    if columns is None :
        columns = ds.numeric_features()

    features_bins = {}    
    for feature in tqdm(columns):
        features_bins[feature] = compute_bins(ds, feature, number_of_bins)

    histograms = {}
    for life in ds:
        for feature in columns:
            if feature not in histograms:
                histograms[feature] = []
            histograms[feature].append(histogram_per_life(life, feature, features_bins[feature]))
    
    df_data = []
    for feature in columns:
        data = {}
        for (i, h1) in enumerate(histograms[feature]):     
            for  (j, h2) in enumerate(histograms[feature]):
                kl = (np.mean(kl_div(h1, h2)) + np.mean(kl_div(h2, h1)))/ 2
                wd = wasserstein_distance(h1, h2)
                df_data.append((i, j, wd, kl, feature))
    df = pd.DataFrame(df_data, columns=['Life 1', 'Life 2', 'W', 'KL', 'feature'])

    return df


def wasserstein_between_lives(
    life1: pd.DataFrame, life2: pd.DataFrame, columns: List[str], bins: int = 15
) -> pd.DataFrame:
    def dropna_values(x):
        return x[~np.isnan(x)]
        
    data = []
    for c in columns:
        v1 = dropna_values(life1[c].values )
        v2 = dropna_values(life2[c].values )
        n_points = min(len(v1), len(v2))
        v1 = np.random.choice(v1, n_points)
        v2 = np.random.choice(v2, n_points)
        _, b = np.histogram(np.hstack((v1, v2)), bins=bins)
        
        h1, _ = np.histogram(v1, bins=b)
        h2, _ = np.histogram(v2, bins=b)
        
        #h1 = h1 / (np.nansum(h1)+ 0.0001)
        #h2 = h2 / (np.nansum(h2) + 0.0001)
    
        
        
        
    return pd.DataFrame(data, columns=["Wasserstein"], index=columns)
