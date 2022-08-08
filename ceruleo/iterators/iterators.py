"""Windowed iteration capabilites

Usually time series data is divided in a contigous periods
of a determined window size, with some arbitray overlapping.
How the time series is iterated constitutes a very important
aspect when building a PdM model.

This module provides utilities to iterate the dataset



"""
from abc import abstractmethod
from enum import Enum
import logging
from signal import signal
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from ceruleo.dataset.transformed import TransformedDataset
from ceruleo.iterators.sample_weight import AbstractSampleWeights, NotWeighted, SampleWeight
from ceruleo.iterators.shufflers import AbstractShuffler, NotShuffled
from tqdm.auto import tqdm
import functools


logger = logging.getLogger(__name__)



def seq_to_seq_signal_generator(
    signal_X:np.ndarray,
    signal_Y:np.ndarray,
    i: int,
    window_size: int,
    output_size: int = 1,
    right_closed: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generator for sequence to sequence models

    Parameters:
        signal_X: The input signal
        signal_Y: The output signal
        i: Current index
        window_size: indow size
        output_size: Output sequence length, by default 1
        right_closed: Wether the lsat input of the windwo is included or not, by default True

    Returns:
        Input and ouput sequences

    """
    initial = max(i - window_size + 1, 0)
    is_df = isinstance(signal_X, pd.DataFrame)
    if is_df:
        signal_X_1 = signal_X.iloc[initial : i + (1 if right_closed else 0), :]
    else:
        signal_X_1 = signal_X[initial : i + (1 if right_closed else 0), :]

    if is_df:
        signal_y_1 = signal_Y.iloc[initial : i + (1 if right_closed else 0), :]
    else:
        signal_y_1 = signal_Y[initial : i + (1 if right_closed else 0), :]

    return (signal_X_1, signal_y_1)


def windowed_signal_generator(
    data: pd.DataFrame,
    target: pd.DataFrame,
    i: int,
    window_size: int,
    output_size: int = 1,
    right_closed: bool = True,
):
    """
    Return a lookback window and the value to predict.

    Parameters:

        data: Matrix of size (life_length, n_features) with the information of the life
        target: Target feature of size (life_length)
        i: Position of the value to predict
        window_size: Size of the lookback window
        output_size: Number of points of the target
        right_closed: Wether the las sample of the window should be included or not


    Returns:

        tuple (np.array, float)
    """
    initial = max(i - window_size + 1, 0)
    is_df = isinstance(data, pd.DataFrame)
    if is_df:
        signal_X_1 = data.iloc[initial : i + (1 if right_closed else 0), :].values
    else:
        signal_X_1 = data[initial : i + (1 if right_closed else 0), :]

    if len(target.shape) == 1:
        if is_df:
            signal_y_1 = target.iloc[i : min(i + output_size, target.shape[0])].values
        else:
            signal_y_1 = target[i : min(i + output_size, target.shape[0])]

        if signal_y_1.shape[0] < output_size:
            padding = np.zeros(output_size - signal_y_1.shape[0])
            signal_y_1 = np.hstack((signal_y_1, padding))
        signal_y_1 = np.expand_dims(signal_y_1, axis=1)
    else:
        if is_df:
            signal_y_1 = target.iloc[i : min(i + output_size, target.shape[0]), :].values
        else:
            signal_y_1 = target[i : min(i + output_size, target.shape[0]), :]

        if signal_y_1.shape[0] < output_size:

            padding = np.zeros(
                ((output_size - signal_y_1.shape[0]), signal_y_1.shape[1])
            )
            signal_y_1 = np.concatenate((signal_y_1, padding), axis=0)

    if signal_X_1.shape[0] < window_size:

        signal_X_1 = np.vstack(
            (
                np.zeros((window_size - signal_X_1.shape[0], signal_X_1.shape[1])),
                signal_X_1,
            )
        )

    return (signal_X_1, signal_y_1)


class IterationType(Enum):
    """Iteration type
    
    Possible values are

    - SEQ_TO_SEQ = 1:

        The seq to seq iterator will return as a target a window of a same size 
        as the input aligned with it
        

    - FORECAST = 2
    
        The forecast iterator produces as target the values of the Y transformers 
        that start where the X data ends.
        

    """
    SEQ_TO_SEQ = 1
    FORECAST = 2


def valid_sample(
    padding: int,
    window_size: int,
    current_sample: int,
    y: int,
):
    if not padding:
        return current_sample >= window_size - 1
    else:
        return True


class RelativePosition:
    """Relative position selector base class

    The relative position selectors allow specifying
    the iteration starts and end relative to the beginning
    or the end of the run-to-cycle failure
    """
    def __init__(self, i: int):
        self.i = i

    @abstractmethod
    def get(self, time_series_length: int):
        raise NotImplementedError


class RelativeToEnd(RelativePosition):
    """Specify positions relative to the end of the run-to-failure cycle

    Example:

        An iterator that iterate each run-to-failure cycle starting
        in the last 500 samples of each cycle.

        ``` py
        iterator = WindowedDatasetIterator(
                transformed_ds,
                window_size=3,
                step=1,
                start_index=RelativeToEnd(500),
                horizon=1)
        ```
    """
    def get(self, time_series_length: int):
        return max(time_series_length - self.i, 0)



class RelativeToStart(RelativePosition):
    """Specify positions relative to the start of the run-to-failure cycle

    Example:

        An iterator that iterate each run-to-failure cycle skipping the first
        200 samples of each cycle.

        ``` py
        iterator = WindowedDatasetIterator(
                transformed_ds,
                window_size=3,
                step=1,
                start_index=RelativeToStart(25),
                horizon=1)
        ```
    """
    def get(self, time_series_length: int):
        return self.i


class WindowedDatasetIterator:
    """Iterate a dataset using windows

    Parameters:

        dataset: The transformed dataset
        window_size: Size of the lookback window
        step: Separation between two consecutive size
            If step == window_size there are not overlapping
            between two consecutive windows
        horizon: Horizon to be predicted.
            If this value is 3, for each window, 3 elements
            of the target are expected to be predicted
        shuffler: How the data should be shuffled
        sample_weight: Which are the sample weight for each sample
        right_closed: Wether the last point of the window should be included or not
        padding: Wether to pad elements if the samples are not enough to fill the window
            Usually this happens at the beginning of the window
        iteration_type: Specify its the underlying model its a forecasting in which
             an scalar is predicted, or a sequence to sequence model similar
             to an autoencoder 
        start_index: Initial index of each run-tu-failure cycle
        end_index: Final index of each run-to-failure cycle
        valid_sample: A callable that returns wether a sample is valid or not
        last_point: Wether to add the last point
    """
    def __init__(
        self,
        dataset: TransformedDataset,
        window_size: int,
        step: int = 1,
        horizon: int = 1,
        shuffler: AbstractShuffler = NotShuffled(),
        sample_weight: SampleWeight = NotWeighted(),
        right_closed: bool = True,
        padding: bool = False,
        iteration_type: IterationType = IterationType.FORECAST,
        start_index: Union[int, RelativePosition] = 0,
        end_index: Optional[Union[int, RelativePosition]] = None,
        valid_sample: Callable[[int, int, int, int, int], bool] = valid_sample,
        last_point: bool = True

    ):
        self.last_point = last_point
        if isinstance(start_index, int):
            start_index = RelativeToStart(start_index)
        self.start_index = start_index
        if end_index is None:
            end_index = RelativeToEnd(0)
        elif isinstance(end_index, int):
            end_index = RelativeToStart(end_index)
        self.end_index = end_index
        self.dataset = dataset
        self.shuffler = shuffler
        self.window_size = window_size
        self.step = step
        self.shuffler.initialize(self)
        self.iteration_type = iteration_type




        if self.iteration_type == IterationType.FORECAST:
            self.slicing_function = windowed_signal_generator
        else:
            self.slicing_function = seq_to_seq_signal_generator

        if not isinstance(sample_weight, AbstractSampleWeights) or not callable(
            sample_weight
        ):
            raise ValueError(
                "sample_weight should be an AbstractSampleWeights or a callable"
            )

        self.sample_weight = sample_weight

        self.i = 0
        self.horizon = horizon
        self.right_closed = right_closed
        self.length = None
        self.padding = padding
        self.valid_sample = functools.partial(
            valid_sample, self.padding, self.window_size
        )



    def __len__(self):
        """
        Return the length of the iterator

        If it not was iterated once, it will compute the length by iterating
        from the entire dataset
        """
        if self.length is None:
            self.length = sum(1 for _ in self)
            self.__iter__()
        return self.length

    def __iter__(self):
        self.i = 0
        self.shuffler.start(self)
        return self

    def __next__(self):
        life, timestamp = self.shuffler.next_element()
        X, y, metadata = self.dataset[life]
        is_df = isinstance(y, pd.DataFrame)
        if is_df:
            valid = self.valid_sample(timestamp, y.iloc[timestamp])
        else:
            valid = self.valid_sample(timestamp, y[timestamp])
        while not valid:
            life, timestamp = self.shuffler.next_element()
            X, y, metadata = self.dataset[life]
            if is_df:
                valid = self.valid_sample(timestamp, y.iloc[timestamp])
            else:
                valid = self.valid_sample(timestamp, y[timestamp])

        curr_X, curr_y = self.slicing_function(
            X, y, timestamp, self.window_size, self.horizon, self.right_closed
        )
        return curr_X, curr_y, [self.sample_weight(y, timestamp, metadata)]

    def get_data(self, flatten: bool = True, show_progress: bool = False):
        """Obtain a 

        Parameters:
        
            flatten: Wether to flatten data
            show_progress: Wether to show progress

        Returns:
            X, y, sw: Data, target and sample weights
        """
        N_points = len(self)

        if flatten:
            dimension = self.window_size * self.n_features
            X = np.zeros((N_points, dimension), dtype=np.float32)
        else:
            X = np.zeros(
                (N_points, self.window_size, self.n_features), dtype=np.float32
            )
        if self.iteration_type == IterationType.FORECAST:
            y = np.zeros((N_points, self.horizon), dtype=np.float32)
        else:
            y = np.zeros((N_points, self.window_size, self.horizon), dtype=np.float32)
        sample_weight = np.zeros(N_points, dtype=np.float32)

        iterator = enumerate(self)
        if show_progress:
            iterator = tqdm(iterator, total=len(self))

        for i, (X_, y_, sample_weight_) in iterator:
            if flatten:
                X[i, :] = X_.flatten()
            else:
                X[i, :, :] = X_
            if self.iteration_type == IterationType.FORECAST:
                y[i, :] = y_.flatten()
            else:
                y[i, :, :] = y_
            sample_weight[i] = sample_weight_[0]
        return X, y, sample_weight

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
        return self.dataset.transformer.n_features

    @property
    def shape(self) -> Tuple[int, int]:
        """Tuple containing (window_size, n_features)

        Returns
        -------
        Tuple[int, int]
            Tuple containing (window_size, n_features)
        """
        return (self.window_size, self.n_features)
