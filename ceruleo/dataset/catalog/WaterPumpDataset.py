import os
from typing import Optional
from ceruleo import DATA_PATH
import pandas as pd
from ceruleo.dataset.ts_dataset import AbstractLivesDataset
import numpy as np


FEATURES_COLUMNS = [
                "Motor Casing Vibration",
            "Motor Frequency A",
            "Motor Frequency B",
            "Motor Frequency C",
            "Motor Speed",
            "Motor Current",
            "Motor Active Power",
            "Motor Apparent Power",
            "Motor Reactive Power",
            "Motor Shaft Power",
            "Motor Phase Current A",
            "Motor Phase Current B",
            "Motor Phase Current C",
            "Motor Coupling Vibration",
            "Motor Phase Voltage AB",
            "Sensor 15",
            "Motor Phase Voltage BC",
            "Motor Phase Voltage CA",
            "Pump Casing Vibration",
            "Pump Stage 1 Impeller Speed",
            "Pump Stage 1 Impeller Speed",
            "Pump Stage 1 Impeller Speed",
            "Pump Stage 1 Impeller Speed",
            "Pump Stage 1 Impeller Speed",
            "Pump Stage 1 Impeller Speed",
            "Pump Stage 2 Impeller Speed",
            "Pump Stage 2 Impeller Speed",
            "Pump Stage 2 Impeller Speed",
            "Pump Stage 2 Impeller Speed",
            "Pump Stage 2 Impeller Speed",
            "Pump Stage 2 Impeller Speed",
            "Pump Stage 2 Impeller Speed",
            "Pump Stage 2 Impeller Speed",
            "Pump Stage 2 Impeller Speed",
            "Pump Inlet Flow",
            "Pump Discharge Flow",
            "Pump UNKNOWN",
            "Pump Lube Oil Overhead Reservoir Level",
            "Pump Lube Oil Return Temp",
            "Pump Lube Oil Supply Temp",
            "Pump Thrust Bearing Active Temp",
            "Motor Non Drive End Radial Bearing Temp 1",
            "Motor Non Drive End Radial Bearing Temp 2",
            "Pump Thrust Bearing Inactive Temp",
            "Pump Drive End Radial Bearing Temp 1",
            "Pump non Drive End Radial Bearing Temp 1",
            "Pump Non Drive End Radial Bearing Temp 2",
            "Pump Drive End Radial Bearing Temp 2",
            "Pump Inlet Pressure",
            "Pump Temp Unknown",
            "Pump Discharge Pressure 1",
            "Pump Discharge Pressure 2",
]

class WaterPumpDataset(AbstractLivesDataset):
    """ """

    @property
    def n_time_series(self) -> int:
        return len(self.cycles)
    
    @property
    def rul_column(self) -> str:
        return "RUL"

    def get_time_series(self, i: int) -> pd.DataFrame:
        return self.cycles[i]
    
    def download(self, kaggle_username: str, kaggle_key: str):
        os.environ["KAGGLE_USERNAME"] = kaggle_username
        os.environ["KAGGLE_KEY"] = kaggle_key
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except:
            raise Exception("The kaggle api is needed for download the dataset")

        api = KaggleApi()

        api.authenticate()

        api.dataset_download_files(
            "nphantawee/pump-sensor-data",
            path=str(self.dataset_path / "raw"),
            unzip=True,
        )

    def split_in_cycles(self):
        q = self.df
        q = q[q["Machine status"] == "BROKEN"].copy()
        q["Life"] = list(range(1, 1 + q.shape[0]))
        q = q[["timestamp", "Life"]].copy()
        r = pd.merge_asof(
            self.df,
            q[["timestamp", "Life"]],
            on="timestamp",
            direction="forward",
            suffixes=("", "_broken"),
        )

        r["RUL"] = 0
        self.cycles = []
        for life_idx, life in r.groupby("Life"):
            life["RUL"] = life.shape[0] - np.arange(1, life.shape[0] + 1)
            life.set_index("timestamp", inplace=True)
            self.cycles.append(life)

    def __init__(self, *, kaggle_username: Optional[str], kaggle_key: Optional[str]):
        super().__init__()

        self.dataset_path = DATA_PATH / "water_pump"

        if not (self.dataset_path / "raw" / "sensor.csv").is_file():
            assert kaggle_key
            assert kaggle_username
            self.download(kaggle_username, kaggle_key)
        self.df = pd.read_csv(self.dataset_path / "raw" / "sensor.csv")
        self.df.drop(columns=["Unnamed: 0"], inplace=True)

        self.df.columns = [
            "timestamp",
            "Motor Casing Vibration",
            "Motor Frequency A",
            "Motor Frequency B",
            "Motor Frequency C",
            "Motor Speed",
            "Motor Current",
            "Motor Active Power",
            "Motor Apparent Power",
            "Motor Reactive Power",
            "Motor Shaft Power",
            "Motor Phase Current A",
            "Motor Phase Current B",
            "Motor Phase Current C",
            "Motor Coupling Vibration",
            "Motor Phase Voltage AB",
            "Sensor 15",
            "Motor Phase Voltage BC",
            "Motor Phase Voltage CA",
            "Pump Casing Vibration",
            "Pump Stage 1 Impeller Speed",
            "Pump Stage 1 Impeller Speed",
            "Pump Stage 1 Impeller Speed",
            "Pump Stage 1 Impeller Speed",
            "Pump Stage 1 Impeller Speed",
            "Pump Stage 1 Impeller Speed",
            "Pump Stage 2 Impeller Speed",
            "Pump Stage 2 Impeller Speed",
            "Pump Stage 2 Impeller Speed",
            "Pump Stage 2 Impeller Speed",
            "Pump Stage 2 Impeller Speed",
            "Pump Stage 2 Impeller Speed",
            "Pump Stage 2 Impeller Speed",
            "Pump Stage 2 Impeller Speed",
            "Pump Stage 2 Impeller Speed",
            "Pump Inlet Flow",
            "Pump Discharge Flow",
            "Pump UNKNOWN",
            "Pump Lube Oil Overhead Reservoir Level",
            "Pump Lube Oil Return Temp",
            "Pump Lube Oil Supply Temp",
            "Pump Thrust Bearing Active Temp",
            "Motor Non Drive End Radial Bearing Temp 1",
            "Motor Non Drive End Radial Bearing Temp 2",
            "Pump Thrust Bearing Inactive Temp",
            "Pump Drive End Radial Bearing Temp 1",
            "Pump non Drive End Radial Bearing Temp 1",
            "Pump Non Drive End Radial Bearing Temp 2",
            "Pump Drive End Radial Bearing Temp 2",
            "Pump Inlet Pressure",
            "Pump Temp Unknown",
            "Pump Discharge Pressure 1",
            "Pump Discharge Pressure 2",
            "Machine status",
        ]
        # Sensor 15 is NULL
        self.df.drop(columns=["Sensor 15"], inplace=True)
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
        self.split_in_cycles()
