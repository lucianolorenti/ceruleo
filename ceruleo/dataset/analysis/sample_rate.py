import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from ceruleo.dataset.ts_dataset import AbstractPDMDataset

logger = logging.getLogger(__name__)


def sample_rate(ds: AbstractPDMDataset, unit: str = "s") -> np.ndarray:
    """Obtain an array of time difference between two consecutive samples

    If the index it's a timestamp, the time difference will be converted to the provided unit

    Parameters:
        ds: The dataset
        unit: Unit to convert the timestamps differences

    Returns:
        Array of time differences

    """
    time_diff = []
    for life in ds:
        diff = np.diff(life.index.values)
        if pd.api.types.is_timedelta64_ns_dtype(diff.dtype):
            diff = diff / np.timedelta64(1, unit)
        time_diff.extend(diff)
    return np.array(time_diff)


def sample_rate_summary(
    ds: AbstractPDMDataset, unit: Optional[str] = "s"
) -> pd.DataFrame:
    """
    Obtain the mean, mode and standard deviation of the sample rate of the dataset

    Parameters:
        ds: The dataset
        unit: Unit to convert the time differences

    Returns:
        A Dataframe with the following columns: Mean sample rate, Std sample rate, Mode sample rate
    """
    sr = sample_rate(ds, unit)
    return pd.DataFrame(
        {
            "Mean sample rate": np.mean(sr),
            "Std sample rate": np.std(sr),
            "Mode sample rate": pd.Series(sr).mode().values[0],
        },
        index=["Dataset"],
    )
