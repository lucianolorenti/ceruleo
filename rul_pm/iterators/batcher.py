import math
from typing import Optional

import numpy as np
from rul_pm.dataset.lives_dataset import AbstractLivesDataset
from rul_pm.iterators.iterators import WindowedDatasetIterator
from rul_pm.transformation.transformers import Transformer


class Batcher:
    def __init__(self,
                 iterator: WindowedDatasetIterator,
                 batch_size: int,
                 restart_at_end: bool = True):
        self.iterator = iterator
        self.batch_size = batch_size
        self.restart_at_end = restart_at_end
        self.stop = False

    def __len__(self):
        return math.ceil(len(self.iterator) / self.batch_size)

    def __iter__(self):
        self.iterator.__iter__()
        return self

    def __next__(self):
        X = []
        y = []
        sample_weights = []
        if self.stop:
            raise StopIteration
        if self.iterator.at_end():
            if self.restart_at_end:
                self.__iter__()
            else:
                raise StopIteration
        try:
            for _ in range(self.batch_size):
                X_t, y_t, sample_weight = next(self.iterator)
                X.append(np.expand_dims(X_t, axis=0))
                y.append(np.expand_dims(y_t, axis=0))
                sample_weights.append(np.expand_dims(sample_weight, axis=0))

        except StopIteration:
            pass
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        sample_weights = np.concatenate(sample_weights, axis=0)
        return X.astype(np.float32), y.astype(np.float32), sample_weights


def get_batcher(dataset: AbstractLivesDataset, window: int, batch_size: int,
                transformer: Transformer, step: int, output_size: int = 1,
                shuffle: bool = False, restart_at_end: bool = True, cache_size: int = 20,
                evenly_spaced_points: Optional[int] = None,
                sample_weight: str = 'equal', add_last: bool = True) -> Batcher:
    """
    Utility function to create a batcher from a dataset
    """
    iterator = WindowedDatasetIterator(dataset,
                                       window,
                                       transformer,
                                       step=step,
                                       output_size=output_size,
                                       shuffle=shuffle,
                                       cache_size=cache_size,
                                       evenly_spaced_points=evenly_spaced_points,
                                       sample_weight=sample_weight,
                                       add_last=add_last)
    b = Batcher(iterator, batch_size, restart_at_end)
    return b
