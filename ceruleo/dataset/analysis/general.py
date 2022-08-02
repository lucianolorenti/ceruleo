import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset
import antropy as ant
from itertools import combinations

logger = logging.getLogger(__name__)



def sample_rate(ds: AbstractTimeSeriesDataset, unit:Optional[str] = 's') -> np.ndarray:
    """Obtain an array of time difference between two consecutive samples

    If the index it's a timestamp, the time difference will be converted to the provided
    unit 

    Parameters
    ----------
    ds : AbstractTimeSeriesDataset
        The dataset
    unit : Optional[str], optional
        Unit to convert the timestamps differences, by default 's'

    Returns
    -------
    np.ndarray
        
    """
    time_diff = []
    for life in ds:
        diff =  np.diff(life.index.values)
        if pd.api.types.is_timedelta64_ns_dtype(diff.dtype):
            if unit is None:
                unit = 's'
            diff = diff / np.timedelta64(1, unit)
        time_diff.extend(diff)
    return np.array(time_diff)


def sample_rate_summary(ds: AbstractTimeSeriesDataset, unit:Optional[str] = 's') -> Tuple[float, float]:
    """Obtain the main and standar deviation of the sample rate of the dataset

    Parameters
    ----------
    ds : AbstractTimeSeriesDataset
        The dataset
    unit : Optional[str], optional
        Unit to convert the time differences, by default 's'

    Returns
    -------
    Tuple[float, float]
        Mean and standard deviation of the sample rate
    """
    sr = sample_rate(ds, unit)
    return np.mean(sr), np.std(sr)