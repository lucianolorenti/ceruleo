from abc import ABC, abstractmethod
import pandas as pd 
from pathlib import Path 

def merge_data_with_faults(
        data:pd.DataFrame,
        fault:pd.DataFrame
):
    """Merge the raw sensor data with the fault information

    Parameters:

        data_file: Path where the raw sensor data is located
        fault_data_file: Path where the fault information is located

    Returns:

        df: Dataframe indexed by time with the raw sensors and faults
            The dataframe contains also a fault_number column
    """
    data = data.set_index("time")
    fault = fault.drop_duplicates(subset=["time"]).set_index("time")
    fault["fault_number"] = range(fault.shape[0])
    return pd.merge_asof(data, fault, on="time", direction="forward").set_index(
        "time"
    )


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
    

class FailureDataCycleSplitter:
    def __init__(self, failure_list: pd.DataFrame):
        self.failure_list = failure_list
        

    def split(self, input_path:Path):
        files = list((input_path / "data").resolve().glob("*.csv"))
        faults_files = list(
            (input_path / "faults").resolve().glob("*.csv")
        )
        dataset_data = []
        for filename in tqdm(self.faults_files.keys(), "Processing files"):
            logger.info(f"Loading data file {files[tool]}")
            tool = filename[0:6]
            data_file = self.sensors_files[filename]
            fault_data_file = self.faults_files[filename]
            data = merge_data_with_faults(
                pd.read_csv(data_file), pd.read_csv(fault_data_file)
            )
            for life_index, life_data in data.groupby("fault_number"):
                if life_data.shape[0] == 0:
                    continue

                output_filename = f"Cycle_{int(life_index)}.pkl.gzip"
                dataset_data.append((tool, life_data.shape[0], output_filename))
                life = life_data.copy()
                life["RUL"] = self.RUL_column.get(life)
                self.save_cycle(life)
        df = pd.DataFrame(
            dataset_data,
            columns=["Tool", "Number of samples", "Failure Type", "Filename"],
        )
        df.to_csv(dataset_path / "processed" / "lives" / "lives_db.csv")
