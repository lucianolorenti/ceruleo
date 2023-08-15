from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path


class CyclesSplitter(ABC):
    @abstractmethod
    def split(self, data: pd.DataFrame):
        raise NotImplementedError


class IncreasingFeatureCycleSplitter:
    def __init__(self, increasing_feature: str):
        self.increasing_feature = increasing_feature

    def split(self, data: pd.DataFrame):
        restart_points = df[df[self.increasing_feature].diff() < 0].index.tolist()
        subsets = []
        start_idx = 0
        for restart_idx in restart_points:
            subset = df.iloc[start_idx:restart_idx]
            subsets.append(subset)
            start_idx = restart_idx
        subsets.append(df.iloc[start_idx:])


class LifeIdCycleSplitter:
    def __init__(self, life_id_feature: str):
        self.life_id_feature = life_id_feature

    def split(self, data: pd.DataFrame):
        subsets = []
        for life_id in data[self.life_id_feature].unique():
            subset = data[data[self.life_id_feature] == life_id]
            subsets.append(subset)
        return subsets


class LifeEndIndicatorCycleSplitter:
    def __init__(self, life_end_indicator_feature: str, end_value=1):
        self.life_end_indicator_feature = life_end_indicator_feature
        self.end_value = end_value

    def split(self, data: pd.DataFrame):
        subsets = []
        start_idx = 0
        for idx in data[
            data[self.life_end_indicator_feature] == self.end_value
        ].index.tolist():
            subset = data.iloc[start_idx:idx]
            subsets.append(subset)
            start_idx = idx
        subsets.append(data.iloc[start_idx:])
        return subsets
    

class FailureListCycleSplitter:
    def __init__(self, failure_list: pd.DataFrame):
        self.failure_list = failure_list
        

    def split(self, data: pd.DataFrame):
        subsets = []
        start_idx = 0
        for idx in self.failure_list.index.tolist():
            subset = data.iloc[start_idx:idx]
            subsets.append(subset)
            start_idx = idx
        subsets.append(data.iloc[start_idx:])
        return subsets


class DatasetBuilder:
    def __init__(self, splitter: CyclesSplitter):
        self.splitter = splitter

    def set_restarting_time_feature(self, name: str):
        self._restarting_time_feature = name
        return self

    def set_life_id_feature(self, name: str):
        self._life_id_feature = name
        return self

    def set_life_end_indicator_feature(self, name: str):
        self._life_end_indicator_feature = name
        return self

    def set_machine_id_feature(self, name: str):
        self._machine_type_feature = name
        return self

    def set_failure_list(self, failures: pd.DataFrame):
        self._failures = failures
        return self

    def build(self, data: pd.DataFrame, output_path: Path):
        pass


DatasetBuilder().restarting_time_feature("a").restarting_time_feature("b")


df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


# Three types of failure specification:
# Upload only one file
# One increasing feature that is the cumulated time of the piece being in place
# A column life id
# A column with a life en indicator
# Upload two files: data + list of failures
# A list of failures
# Upload multiple files
# Separated cycles
