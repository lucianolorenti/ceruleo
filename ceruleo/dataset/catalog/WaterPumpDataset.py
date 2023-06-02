import os
from ceruleo import DATA_PATH
import pandas as pd
from ceruleo.dataset.ts_dataset import AbstractLivesDataset

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except:
    raise Exception("Please install kaggle-api using pip install kaggle-api")


class WaterPumpDataset(AbstractLivesDataset):
    """ """

    def __init__(self, kaggle_username: str, kaggle_key: str):
        super().__init__()

        os.environ["KAGGLE_USERNAME"] = kaggle_username
        os.environ["KAGGLE_KEY"] = kaggle_key

        self.dataset_path = DATA_PATH / "water_pump"
        api = KaggleApi()
        api.authenticate()
        if not (self.dataset_path / "raw" / "sensor.csv").is_file():
            api.dataset_download_files(
                "nphantawee/pump-sensor-data",
                path=str(self.dataset_path / "raw"),
                unzip=True,
            )

        self.df = pd.read_csv(self.dataset_path / "raw" / "sensor.csv")
        self.df.drop(columns=["Unnamed: 0"], inplace=True)
        print(len(self.df.columns))
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


