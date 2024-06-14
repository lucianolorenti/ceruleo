import logging
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import List 
from ceruleo.dataset.ts_dataset import AbstractPDMDataset
from ceruleo.utils import pydantic_to_dict

logger = logging.getLogger(__name__)


class SampleRateAnalysis(BaseModel):
    median: float
    mean: float
    std: float
    unit: str

    def to_pandas(self) -> pd.Series:
        return pd.Series(pydantic_to_dict(self)).to_frame().T

    def __repr__(self) -> str:
        return f"Median: {self.median} | {self.mean} +- {self.std} [{self.unit}]"
    

    def _repr_html_(self) -> str:
        return f"""<div> 
        <p> <span style="font-weight:bold"> Median: </span> {self.median} [{self.unit}]  </p>  
        <p> <span style="font-weight:bold">  Mean +- Std: </span> {self.mean:.3f} +- {self.std:.3f} [{self.unit}] </p>
        </div>
    """


def sample_rate(ds: AbstractPDMDataset, unit: str = "s") -> np.ndarray:
    """Obtain an array of time difference between two consecutive samples

    If the index it's a timestamp, the time difference will be converted to the provided unit

    Parameters:
        ds: The dataset
        unit: Unit to convert the timestamps differences

    Returns:
        Array of time differences

    """
    time_diff : List[float ]= []
    for life in ds:
        diff = np.diff(life.index.values)
        diff = diff[diff <= np.median(diff)]
        if pd.api.types.is_timedelta64_ns_dtype(diff.dtype):
            diff = diff / np.timedelta64(1, unit)
        time_diff.extend(diff)
    return np.array(time_diff)



def sample_rate_summary(
    ds: AbstractPDMDataset, unit: str = "s"
) -> SampleRateAnalysis:
    """
    Obtain the mean, median and standard deviation of the sample rate of the dataset

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
        median=np.median(sr),
        unit=unit
    )
