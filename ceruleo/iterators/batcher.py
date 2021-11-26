"""Batching capabilities

Many machine learning models require data to be provided in the form of mini batches.
This module interacts with iterators to generate batches.
In particular, this module was created to be used with pytorch models.

Take a look at the pytorch example to see its usage.
"""
import math
from typing import Tuple

import numpy as np
import pandas as pd
from numpy.lib.arraysetops import isin
from ceruleo.dataset.ts_dataset import AbstractTimeSeriesDataset
from ceruleo.iterators.iterators import (
    NotWeighted,
    SampleWeight,
    WindowedDatasetIterator,
)
from ceruleo.iterators.shufflers import AbstractShuffler, NotShuffled


class Batcher:
    """WindowedIterator Batcher

    Example:

        ``` py 
        batcher = Batcher.new(transformed_dataset,
                              window=150,
                              batch_size=64,
                              step=1,
                              horizon=1)
        X, y, data = next(batcher)     
        X.shape

        (64, 150, n_features)       
        ```                  

    Parameters:

        iterator: Dataset iterator
        batch_size: int

    """

    def __init__(
        self,
        iterator: WindowedDatasetIterator,
        batch_size: int,
    ):
        self.iterator = iterator
        self.batch_size = batch_size
        self.stop = False
        self.batch_data = None

    @staticmethod
    def new(
        dataset: AbstractTimeSeriesDataset,
        window: int,
        batch_size: int,
        step: int,
        horizon: int = 1,
        shuffler: AbstractShuffler = NotShuffled(),
        sample_weight: SampleWeight = NotWeighted(),
        right_closed: bool = True,
        padding: bool = False,
    ):
        """Batcher constructor from a dataset

        The method constructs WindowedDatasetIterator from the dataset and
        then a Batcher from the iterator.
        Most of the parameters come from the WindowedDatasetIterator,


        Example:

            ``` py 
            batcher = Batcher.new(transformed_dataset,
                                window=150,
                                batch_size=64,
                                step=1,
                                horizon=1)
            X, y, data = next(batcher)     
            X.shape

            (64, 150, n_features)       
            ```                  

        Parameters:

            dataset: Dataset from which the batcher will be created
            batch_size: Batch size
            step: strides
            horizon: Size of the horizon to predict. By default 1
            shuffle: AbstractShuffler
            sample_weight: SampleWeight
            right_closed: bool
            padding: wheter to pad data if there are not enough points
                     to fill the window

        Returns:

            batcher: A new constructed batcher
        """
        iterator = WindowedDatasetIterator(
            dataset,
            window,
            step=step,
            horizon=horizon,
            shuffler=shuffler,
            sample_weight=sample_weight,
            right_closed=right_closed,
            padding=padding,
        )
        b = Batcher(iterator, batch_size)
        return b

    def __len__(self) -> int:
        """Number of batches

        Returns:

            batches: Number of batches in the iterator
        """
        if len(self.iterator) is None:
            return None
        q = math.ceil(len(self.iterator) / self.batch_size)
        return q

    def __iter__(self):
        self.stop = False
        self.iterator.__iter__()
        return self

    @property
    def n_features(self) -> int:
        """Number of features of the transformed dataset

        This is a helper method to obtain the transformed
        dataset information from the WindowedDatasetIterator

        Returns:

            features: Number of features of the transformed dataset
        """
        return self.iterator.n_features

    @property
    def window_size(self) -> int:
        """Lookback window size

        This is a helper method to obtain the WindowedDatasetIterator
        information

        Returns:

            window: Lookback window size
        """
        return self.iterator.window_size

    @property
    def output_shape(self) -> int:
        """Number of values returned as target by each sample

        Returns:

            output_size: Number of values returned as target by each sample
        """
        return self.iterator.output_size

    @property
    def input_shape(self) -> Tuple[int, int]:
        """Tuple containing (window_size, n_features)

        Returns:

            window_size, n_features
        """
        return self.iterator.input_shape

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
                return tuple(slice_batch_data_element(q, actual_batch_size) for q in d)
            else:
                return d[: actual_batch_size - 1, :]

        if actual_batch_size == self.batch_size:
            return self.batch_data
        sliced_data = []
        for i in range(len(self.batch_data)):
            sliced_data.append(
                slice_batch_data_element(self.batch_data[i], actual_batch_size)
            )
        return sliced_data

    def __next__(self):
        if self.stop:
            raise StopIteration
        try:
            actual_batch_size = 0
            for j in range(self.batch_size):
                actual_batch_size += 1
                d = next(self.iterator)
                self.allocate_batch_data(d)
                self._assign_data(d, j)
        except StopIteration:
            self.stop = True

        return self._slice_data(actual_batch_size)
