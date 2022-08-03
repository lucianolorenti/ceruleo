from abc import abstractmethod
from enum import Enum
import logging
import random
from signal import signal
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from ceruleo.dataset.transformed import TransformedDataset
from ceruleo.iterators.shufflers import AbstractShuffler, NotShuffled
from tqdm.auto import tqdm
import functools


logger = logging.getLogger(__name__)


class AbstractSampleWeights:
    def __call__(self, y, i: int, metadata):
        raise NotImplementedError


class NotWeighted(AbstractSampleWeights):
    def __call__(self, y, i: int, metadata):
        return 1


SampleWeight = Union[AbstractSampleWeights, Callable[[np.ndarray, int, Any], float]]


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

    signal_X_1 = signal_X[initial : i + (1 if right_closed else 0), :]
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

    Parameters
    ----------
    data:
             Matrix of size (life_length, n_features) with the information of the life
    target:
             Target feature of size (life_length)
    i: int
       Position of the value to predict

    window_size: int
                 Size of the lookback window

    output_size: int
                 Number of points of the target

    right_closed: bool


    Returns
    -------
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
    def __init__(self, i: int):
        self.i = i

    @abstractmethod
    def get(self, time_series_length: int):
        raise NotImplementedError


class RelativeToEnd(RelativePosition):
    def get(self, time_series_length: int):
        
        return max(time_series_length - self.i, 0)
        


class RelativeToStart(RelativePosition):
    def get(self, time_series_length: int):
        return self.i


class WindowedDatasetIterator:
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

        if isinstance(start_index, int):
            start_index = RelativeToStart(start_index)
        self.start_index = start_index
        if end_index is None:
            end_index = RelativeToEnd(0)
        elif isinstance(end_index, int):
            end_index = RelativeToStart(end_index)
        self.last_point = last_point
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
