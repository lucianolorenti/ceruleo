from pathlib import Path

PACKAGE_PATH = Path(__file__).resolve().parent
DATA_PATH = PACKAGE_PATH /'dataset' / 'data'

DATASET_PATH = Path.home() / '.rul_pm' / 'datasets'
DATASET_PATH.mkdir(parents=True, exist_ok=True)


__version__ = 0.5
