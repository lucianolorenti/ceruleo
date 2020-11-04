import math
import numpy as np
from rul_pm.dataset.lives_dataset import AbstractLivesDataset
from rul_pm.transformation.transformers import Transformer
from rul_pm.iterators.iterators import WindowedDatasetIterator
from tqdm.auto import tqdm

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
        if self.stop:
            raise StopIteration
        if self.iterator.at_end():
            if self.restart_at_end:
                self.__iter__()
            else:
                raise StopIteration
        try:
            for _ in range(self.batch_size):
                X_t, y_t = next(self.iterator)
                X.append(np.expand_dims(X_t, axis=0))
                y.append(y_t)
        except StopIteration:
            pass
        X = np.concatenate(X, axis=0)
        y = np.array(y)
        return X.astype(np.float32), y.astype(np.float32)


def get_batcher(dataset: AbstractLivesDataset,
                window: int,
                batch_size: int,
                transformer: Transformer,
                step: int,
                output_size : int = 1,
                shuffle: bool = False,
                restart_at_end: bool = True,
                cache_size: int = 20) -> Batcher:
    iterator = WindowedDatasetIterator(dataset,
                                       window,
                                       transformer,
                                       step=step,
                                       output_size=output_size,
                                       shuffle=shuffle,
                                       cache_size=cache_size)
    b =  Batcher(iterator, batch_size)
    b.restart_at_end = restart_at_end
    return b


def dataset_map(fun, dataset, step, transformer, window):    
    batcher = get_batcher(dataset,
                          window,
                          512,
                          transformer,
                          step,
                          shuffle=False)
    batcher.restart_at_end = False    
    for X, y in tqdm(batcher):
        fun(X, y)



