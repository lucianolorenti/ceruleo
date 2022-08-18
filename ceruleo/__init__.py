from pathlib import Path
import os

PACKAGE_PATH = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get('CERULEO_DATA_PATH', Path.home() / '.ceruleo' / 'data'))
DATA_PATH.mkdir(parents=True, exist_ok=True)

CACHE_PATH = Path(os.environ.get('CERULEO_CACHE_PATH', Path.home() / '.ceruleo' / 'cache'))
CACHE_PATH.mkdir(parents=True, exist_ok=True)


__version__ = "2.0.0"
