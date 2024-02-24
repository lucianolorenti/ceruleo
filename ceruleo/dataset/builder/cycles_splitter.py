from abc import ABC, abstractmethod
from typing import Iterator

import pandas as pd
from pathlib import Path

from ceruleo.dataset.builder.output import OutputMode


class CyclesSplitter(ABC):
    @abstractmethod
    def split(self, data: pd.DataFrame) -> Iterator[pd.DataFrame]:
        raise NotImplementedError


class IncreasingFeatureCycleSplitter(CyclesSplitter):
    """
    A splitter that divides a DataFrame into cycles based on changes in the value of an increasing feature.

    When the value of the increasing feature decreases, a new cycle is considered to start.

    """

    def __init__(self, increasing_feature: str):
        """Initializes the splitter with the name of the increasing feature.

        Parameters
        ----------
        increasing_feature : str
            The name of the increasing feature used for splitting.
        """
        self.increasing_feature = increasing_feature

    def split(self, data: pd.DataFrame) -> Iterator[pd.DataFrame]:
        """Splits the input DataFrame into cycles based on changes in the increasing feature.

        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame to be split.

        Yields
        ------
        Iterator[pd.DataFrame]
            An iterator of DataFrames, each containing a cycle of the input data.
        """
        restart_points = data[data[self.increasing_feature].diff() < 0].index.tolist()
        start_idx = 0
        i = 1
        for restart_idx in restart_points:
            subset = data.iloc[start_idx:restart_idx]
            yield subset.copy()
            start_idx = restart_idx
            i += 1
        yield subset.copy()


class LifeIdCycleSplitter(CyclesSplitter):
    """A splitter that divides a DataFrame into cycles based on unique life identifiers."""

    def __init__(self, life_id_feature: str):
        """Initializes the splitter with the name of the life id feature.

        Parameters
        ----------
        life_id_feature : str
            The name of the column representing the life identifier.
        """
        self.life_id_feature = life_id_feature

    def split(self, data: pd.DataFrame) -> Iterator[pd.DataFrame]:
        """Splits the input DataFrame into cycles based on unique life identifiers.

        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame to be split.

        Yields
        ------
        Iterator[pd.DataFrame]
            An iterator of DataFrames, each containing a cycle of the input data.
        """
        for life_id in data[self.life_id_feature].unique():
            subset = data[data[self.life_id_feature] == life_id]
            yield subset.copy()


class LifeEndIndicatorCycleSplitter(CyclesSplitter):
    """A splitter that divides a DataFrame into cycles based on a life end indicator feature."""

    def __init__(self, life_end_indicator_feature: str, end_value=1):
        """

        Parameters
        ----------
        life_end_indicator_feature : str
            The name of the column representing the life end indicator.
        end_value : int, optional
            The value indicating the end of a life cycle. by default 1

        """
        self.life_end_indicator_feature = life_end_indicator_feature
        self.end_value = end_value

    def split(self, data: pd.DataFrame) -> Iterator[pd.DataFrame]:
        """Splits the input DataFrame into cycles based on a life end indicator feature.

        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame to be split.

        Yields
        ------
        Iterator[pd.DataFrame]
            An iterator of DataFrames, each containing a cycle of the input data.
        """
        start_idx = 0
        for idx in data[
            data[self.life_end_indicator_feature] == self.end_value
        ].index.tolist():
            subset = data.iloc[start_idx : idx + 1]
            yield subset.copy()
            start_idx = idx + 1
        if start_idx < data.shape[0]:
            yield data.iloc[start_idx:].copy()


class FailureDataCycleSplitter(CyclesSplitter):
    """A splitter that divides a DataFrame into cycles based on a separate DataFrame containing failure data."""

    data_time_column: str
    fault_time_column: str

    def __init__(self, data_time_column: str, fault_time_column: str):
        self.data_time_column = data_time_column
        self.fault_time_column = fault_time_column

    def split(self, data: pd.DataFrame, fault: pd.DataFrame):
        data = self.merge_data_with_faults(data, fault)
        for life_index, life_data in data.groupby("fault_number"):
            if life_data.shape[0] == 0:
                continue
            yield life_data.copy()

    def merge_data_with_faults(self, data: pd.DataFrame, fault: pd.DataFrame):
        """Merge the raw sensor data with the fault information

        Parameters:

            data_file: Path where the raw sensor data is located
            fault_data_file: Path where the fault information is located

        Returns:

            df: Dataframe indexed by time with the raw sensors and faults
                The dataframe contains also a fault_number column
        """

        fault = fault.drop_duplicates(subset=[self.fault_time_column]).copy()
        fault["fault_number"] = range(fault.shape[0])
        return pd.merge_asof(
            data,
            fault,
            left_on=self.data_time_column,
            right_on=self.fault_time_column,

            suffixes=["_data", "_fault"],
            direction="forward",
        )
