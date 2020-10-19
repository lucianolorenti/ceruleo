import pickle
import random

import numpy as np
import pandas as pd
from rul_gcd.dataset.lives_dataset import AbstractLivesDataset
from rul_gcd.iterators.utils import windowed_signal_generator
from rul_gcd.transformation.transformers import Transformer
from rul_gcd.utils.lrucache import LRUDataCache
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

CACHE_SIZE = 30


class DatasetIterator:
    def __init__(self,
                 dataset: AbstractLivesDataset,
                 transformer: Transformer,
                 shuffle=False,
                 cache_size=CACHE_SIZE):
        self.dataset = dataset
        self.shuffle = shuffle
        self.transformer = transformer
        self.cache = LRUDataCache(cache_size)        

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
                 window_size: int,
                 transformer: Transformer,
                 step: int = 1,
                 shuffle=False,
                 cache_size:int = CACHE_SIZE):
        super().__init__(dataset, transformer, shuffle, cache_size=cache_size)
        self.window_size = window_size
        self.step = step
        self.shuffle = shuffle
        self.orig_lifes, self.orig_elements = self._windowed_element_list()
        self.lifes, self.elements = self.orig_lifes, self.orig_elements
        self.i = 0

    def _windowed_element_list(self):
        olifes = []
        oelements = []
        for life in range(self.dataset.nlives):            
            X, _ = self._load_data(life)            
            list_ranges = list(range(0, X.shape[0], self.step))
            for i in list_ranges:
                if i - self.window_size >= 0:
                    olifes.append(life)
                    oelements.append(i)
        return olifes, oelements

    def _shuffle(self):
        if not self.shuffle:
            return
        valid_shuffle = ((self.shuffle == False)
                         or (self.shuffle
                             in ('signal', 'life', 'all', 'signal_life')))
        df = pd.DataFrame({
            'life': self.orig_lifes,
            'elements': self.orig_elements
        })
        if not valid_shuffle:
            raise ValueError(
                "shuffle parameter invalid. Valid values are: False, 'signal', 'life', 'all' 'signal_life'"
            )
        if self.shuffle == 'signal':
            groups = [d.sample(frac=1, axis=0) for _, d in df.groupby('life')]
            df = pd.concat(groups).reset_index(drop=True)
        elif self.shuffle == 'life':
            groups = [d for _, d in df.groupby('life')]
            random.shuffle(groups)
            df = pd.concat(groups).reset_index(drop=True)
        elif self.shuffle == 'signal_life':
            groups = [d.sample(frac=1, axis=0) for _, d in df.groupby('life')]
            random.shuffle(groups)
            df = pd.concat(groups).reset_index(drop=True)
        elif self.shuffle == 'all':
            df = df.sample(frac=1, axis=0)
        self.lifes = df['life'].values.tolist()
        self.elements = df['elements'].values.tolist()

    def __iter__(self):
        self.i = 0
        self._shuffle()
        return self

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, i: int):
        (life, timestamp) = (self.lifes[i], self.elements[i])
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
