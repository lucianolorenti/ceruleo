import pickle

import numpy as np
from rul_gcd.dataset.lives_dataset import AbstractLivesDataset
from rul_gcd.iterators.utils import (windowed_element_list,
                                     windowed_signal_generator)
from rul_gcd.transformation.transformers import Transformer
from rul_gcd.utils.lrucache import LRUDataCache
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

CACHE_SIZE = 20


class DatasetIterator:
    def __init__(self,
                 dataset: AbstractLivesDataset,
                 transformer: Transformer,
                 shuffle=False,
                 complete=False):
        self.dataset = dataset
        self.shuffle = shuffle
        self.transformer = transformer
        self.cache = LRUDataCache(CACHE_SIZE)
        self.complete = complete

        try:
            check_is_fitted(transformer)
        except NotFittedError:
            self.transformer.fit(dataset)

    def _load_data(self, life):
        if life not in self.cache.data:
            data = self.dataset[life]
            X, y = self.transformer.transform(data)
            self.cache.add(life, (X, y))
        return self.cache.get(life)


class LifeDatasetIterator(DatasetIterator):
    def __init__(self,
                 dataset: AbstractLivesDataset,
                 transformer: Transformer,
                 shuffle=False):
        super().__init__(dataset, transformer, shuffle)
        self.i = 0

    def __iter__(self):
        self.i = 0
        return self

    def __len__(self):
        return len(self.dataset)

    def __next__(self):
        if self.i == len(self):
            raise StopIteration
        current_life = self.i
        self.i += 1
        return self._load_data(current_life)


class WindowedDatasetIterator(DatasetIterator):
    def __init__(self,
                 dataset: AbstractLivesDataset,
                 window_size,
                 transformer: Transformer,
                 step=1,
                 shuffle=False):
        super().__init__(dataset, transformer, shuffle)
        self.window_size = window_size
        self.step = step
        self.shuffle = shuffle
        self.elements = windowed_element_list(self.dataset,
                                              self.window_size,
                                              step=self.step,
                                              shuffle=self.shuffle)
        self.i = 0

    def __iter__(self):
        self.i = 0
        return self

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, i):
        (life, timestamp) = self.elements[i]
        X, y = self._load_data(life)
        return windowed_signal_generator(X, y, timestamp, self.window_size)

    def at_end(self):
        return self.i == len(self.elements)

    def __next__(self):
        if self.at_end():
            raise StopIteration
        ret = self.__getitem__(self.i)
        self.i += 1
        return ret
