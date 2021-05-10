import math
from typing import Optional, Tuple

import numpy as np
from rul_pm.dataset.lives_dataset import AbstractLivesDataset
from rul_pm.iterators.iterators import WindowedDatasetIterator
from rul_pm.transformation.transformers import Transformer


class Batcher:
    """WindowedIterator Batcher

    Parameters
    ----------
    iterator: WindowedDatasetIterator
              Dataset iterator
    batch_size: int
                Batch size to use
    restart_at_end: bool = True
                    Wether if the iterator is infinite or not
    """
    def __init__(self,
                 iterator: WindowedDatasetIterator,
                 batch_size: int,
                 restart_at_end: bool = True):
        self.iterator = iterator
        self.batch_size = batch_size
        self.restart_at_end = restart_at_end
        self.stop = False
        self.prefetch_size = None

    @staticmethod
    def new(dataset: AbstractLivesDataset,
            window: int,
            batch_size: int,
            transformer: Transformer,
            step: int,
            output_size: int = 1,
            shuffle: bool = False,
            restart_at_end: bool = True,
            cache_size: int = 20,
            evenly_spaced_points: Optional[int] = None,
            sample_weight: str = 'equal',
            add_last: bool = True,
            discard_threshold: Optional[float] = None):
        """Batcher constructor from a dataset

        The method constructs WindowedDatasetIterator from the dataset and
        then a Batcher from the iterator.
        Most of the parameters come from the WindowedDatasetIterator,

        Parameters
        ----------
        dataset : AbstractLivesDataset
            Dataset from which the batcher will be created
        batch_size : int
            Batch size
        restart_at_end : bool, optional
            [description], by default True

        Returns
        -------
        [type]
            [description]
        """
        iterator = WindowedDatasetIterator(
            dataset,
            window,
            transformer,
            step=step,
            output_size=output_size,
            shuffle=shuffle,
            cache_size=cache_size,
            evenly_spaced_points=evenly_spaced_points,
            sample_weight=sample_weight,
            add_last=add_last,
            discard_threshold=discard_threshold)
        b = Batcher(iterator, batch_size, restart_at_end)
        return b

    def __len__(self) -> int:
        """Number of batches

        Returns
        -------
        int
            Number of batches in the iterator
        """
        return math.ceil(len(self.iterator) / self.batch_size)

    def __iter__(self):
        self.iterator.__iter__()
        return self

    @property
    def n_features(self) -> int:
        """Number of features of the transformed dataset

        This is a helper method to obtain the transformed
        dataset information from the WindowedDatasetIterator

        Returns
        -------
        int
            Number of features of the transformed dataset
        """
        return self.iterator.transformer.n_features

    @property
    def window_size(self)->int:
        """Lookback window size

        This is a helper method to obtain the WindowedDatasetIterator
        information

        Returns
        -------
        int
            Lookback window size
        """
        return self.iterator.window_size

    @property
    def output_shape(self) -> int:
        """Number of values returned as target by each sample

        Returns
        -------
        int
            Number of values returned as target by each sample
        """
        return self.iterator.output_size

    @property
    def input_shape(self) -> Tuple[int, int]:
        """Tuple containing (window_size, n_features)

        Returns
        -------
        Tuple[int, int]
            Tuple containing (window_size, n_features)
        """
        return (self.window_size, self.n_features)

    @property
    def computed_step(self):
        if isinstance(self.step, int):
            return self.step
        elif isinstance(self.step, tuple):
            if self.step[0] == 'auto':
                return int(self.window / self.step[1])
        raise ValueError('Invalid step parameter')

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
