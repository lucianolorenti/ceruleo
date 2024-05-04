import logging
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from ceruleo.dataset.ts_dataset import AbstractPDMDataset

logger = logging.getLogger(__name__)


class SampleRateAnalysis(BaseModel):
    mode: float
    mean: float
    std: float

    def to_pandas(self) -> pd.Series:
        return pd.Series(self.model_dump()).to_frame().T


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
) -> SampleRateAnalysis:
    """
    Obtain the mean, mode and standard deviation of the sample rate of the dataset

    Parameters:
        ds: The dataset
        unit: Unit to convert the time differences

    Returns:
        A SampleRateAnalysis with the following information: Mean sample rate, Std sample rate, Mode sample rate
    """
    sr = sample_rate(ds, unit)
    return SampleRateAnalysis(
        mean=np.mean(sr),
        std=np.std(sr),
        mode=pd.Series(sr).mode().values[0],
    )
