import functools
import gzip
import pickle
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from ceruleo.dataset.ts_dataset import AbstractTimeSeriesDataset
from ceruleo.transformation.functional.transformers import Transformer
from ceruleo.utils.lrucache import LRUDataCache
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm


def _transform(transformer, dataset, i: int):
    data = dataset[i]
    return (i, transformer.transform(data))


class TransformedDataset(AbstractTimeSeriesDataset):
    def __init__(
        self, dataset, transformer: Transformer, cache_size: Optional[int] = None
    ):
        super().__init__()
        self.transformer = transformer
        self.dataset = dataset
        if cache_size is None:
            cache_size = len(dataset)
        self.cache = LRUDataCache(cache_size)
        check_is_fitted(transformer)

    @property
    def n_time_series(self) -> int:
        return self.dataset.n_time_series

    def __call__(self, i: int):
        return self[i]

    def number_of_samples_of_time_series(self, i: int) -> int:
        _, y, _ = self[i]
        return y.shape[0]

    def preload(self):
        transform = functools.partial(_transform, self.transformer, self.dataset)
        with Pool(6) as p:
            values = list(
                tqdm(
                    p.imap(transform, range(self.n_time_series)),
                    total=self.n_time_series,
                    desc="Preloading",
                )
            )
        for i, (X, y, metadata) in values:
            self.cache.add(i, (X, y, metadata))

    def get_time_series(self, i: int) -> pd.DataFrame:
        if i not in self.cache.data:
            data = self.dataset[i]
            X, y, metadata = self.transformer.transform(data)
            self.cache.add(i, (X, y, metadata))
        X, y, metadata = self.cache.get(i)
        return X, y, metadata

    def get_features_of_life(
        self, i: int, pandas: bool = True
    ) -> Union[np.ndarray, pd.DataFrame]:
        X, _, _ = self[i]
        if pandas:
            return X
        else:
            return X.values

    def save(self, output_path: Path):
        TransformedSerializedDataset.save(self, output_path)


class TransformedSerializedDataset(TransformedDataset):
    @staticmethod
    def save(dataset: TransformedDataset, output_path: Path):
        if not output_path.is_dir():
            output_path.mkdir(parents=True, exist_ok=True)
        file_list = {}
        for i, life in enumerate(dataset):
            file_list[i] = f"ts_{i}.pkl.gz"
            with gzip.open(output_path / f"ts_{i}.pkl.gz", "wb") as file:
                pickle.dump(life, file)
        with open(output_path / "transformer.pkl", "wb") as file:
            pickle.dump(dataset.transformer, file)
        with open(output_path / "lives.pkl", "wb") as file:
            pickle.dump(file_list, file)

    def __init__(self, dataset_path: Path, cache_size: Optional[int] = None):
        self.dataset_path = dataset_path
        with open(dataset_path / "lives.pkl", "rb") as file:
            self.files = pickle.load(file)
        with open(dataset_path / "transformer.pkl", "rb") as file:
            self.transformer = pickle.load(file)
        if cache_size is None:
            cache_size = len(self.files)
        self.cache = LRUDataCache(cache_size)

    def _open_file(self, i: int):
        with gzip.open(self.dataset_path / self.files[i], "rb") as file:
            return pickle.load(file)

    def get_time_series(self, i: int) -> pd.DataFrame:
        if i not in self.cache.data:
            X, y, metadata = self._open_file(i)
            self.cache.add(i, (X, y, metadata))
        return self.cache.get(i)

    @property
    def n_time_series(self):
        return len(self.files)

    def __len__(self):
        return self.n_time_series
