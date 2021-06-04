import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numpy.lib.arraysetops import isin
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

    def __init__(
        self,
        iterator: WindowedDatasetIterator,
        batch_size: int,
        restart_at_end: bool = True,
    ):
        self.iterator = iterator
        self.batch_size = batch_size
        self.restart_at_end = restart_at_end
        self.stop = False
        self.prefetch_size = None
        self.batch_data = None

    @staticmethod
    def new(
        dataset: AbstractLivesDataset,
        window: int,
        batch_size: int,
        transformer: Transformer,
        step: int,
        output_size: int = 1,
        shuffle: bool = False,
        restart_at_end: bool = True,
        cache_size: int = 20,
        evenly_spaced_points: Optional[int] = None,
        sample_weight: str = "equal",
        add_last: bool = True,
        discard_threshold: Optional[float] = None,
    ):
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
            Whether the Batcher is infinite or not, by default True

        Returns
        -------
        Batcher
            A new constructed batcher
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
            discard_threshold=discard_threshold,
        )
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
        return self.iterator.n_features

    @property
    def window_size(self) -> int:
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
            if self.step[0] == "auto":
                return int(self.window / self.step[1])
        raise ValueError("Invalid step parameter")

    def initialize_batch(self):
        def initialize_batch_element(elem):
            if isinstance(elem, tuple):
                for e in elem:
                    initialize_batch_element(e)
            else:
                elem.fill(0)

        if self.batch_data is None:
            return
        for i in range(len(self.batch_data)):
            initialize_batch_element(self.batch_data[i])

    def allocate_batch_data(self, d):
        def allocate_batch_data_element(d):
            if isinstance(d, tuple):
                return tuple(allocate_batch_data_element(q) for q in d)
            else:

                if isinstance(d, np.ndarray) or isinstance(d, pd.Series):
                    shape = d.shape
                elif isinstance(d, list):
                    shape = (len(d),)

                return np.zeros((self.batch_size, *shape))

        if self.batch_data is not None:
            return
        self.batch_data = []
        for i in range(len(d)):
            self.batch_data.append(allocate_batch_data_element(d[i]))

    def _assign_data(self, d, j):
        for i, elem in enumerate(d):
            if isinstance(elem, tuple):
                for k in range(len(elem)):
                    self.batch_data[i][k][j, :] = elem[k]
            else:
                self.batch_data[i][j, :] = elem

    def _slice_data(self, actual_batch_size):
        def slice_batch_data_element(d, actual_batch_size):
            if isinstance(d, tuple):
                return tuple(sliced_data(q, actual_batch_size) for q in d)
            else:
                return d[:actual_batch_size-1, :]

        if actual_batch_size == self.batch_size:
            return self.batch_data
        sliced_data = []
        for i in range(len(self.batch_data)):
            sliced_data.append(
                slice_batch_data_element(self.batch_data[i], actual_batch_size)
            )
        return sliced_data

    def __next__(self):

        self.initialize_batch()

        if self.stop:
            raise StopIteration
        if self.iterator.at_end():
            if self.restart_at_end:
                self.__iter__()
            else:
                raise StopIteration
        try:
            actual_batch_size = 0
            for j in range(self.batch_size):
                actual_batch_size += 1
                d = next(self.iterator)
                self.allocate_batch_data(d)
                self._assign_data(d, j)
        except StopIteration:
            pass

        return self._slice_data(actual_batch_size)
