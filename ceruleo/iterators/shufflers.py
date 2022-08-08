"""Shufflers are helpers classes used by the iterator to change
the order of the run-to-failure cycles and the timestamps inside
each cycle

There are six types of shuffles

- NotShuffled: All the cycles are processed in order
- AllShuffled: 
- IntraTimeSeriesShuffler
- InverseOrder: 
- TimeSeriesOrderIntraSignalShuffling
- TimeSeriesOrderShuffling
"""
from typing import Callable, Optional, Tuple

import numpy as np
import math


class AbstractShuffler:
    """A Shuffler is used by the iterator to interleave samples of different run-to-fail cycles"""

    class Iterator:
        def __init__(self, shuffler, iterator):
            self.iterator = iterator
            self.shuffler = shuffler

        def __iter__(self):
            self.shuffler.initialize(self.iterator)
            return self

        def __next__(self):
            return self.shuffler.next_element()

    def iterator(self, iterator: "WindowedDatasetIterator"):
        return AbstractShuffler.Iterator(self, iterator)

    def at_end(self) -> bool:
        """Determines wether the iterator reached to its end

        Returns:

            bool: Wether the iterator is at its end
        """
        return self.current_time_series == self.wditerator.dataset.n_time_series

    def next_element(self) -> Tuple[int, int]:
        """Iterating function

        The method takes the current time serie, and the current time stamp
        and calls the advance method

        Returns:
            ts_index, timestamp: Index of the current run-to-failure cycle and
                          current timestamp of that cycle

        Raises:

            StopIteration: When the iteration reached to an end

        """

        if self.at_end():
            raise StopIteration
        ts_index = self.time_series()
        timestamp = self.timestamp()
        self.advance()

        return ts_index, timestamp

    def start(self, iterator: "WindowedDatasetIterator"):
        """start the shuffler given an iterator

        Parameters:

            iterator: An Windowed iterator
        """
        self.initialize(iterator)

    def time_series(self) -> int:
        """Current time series

        Returns:

            current_time_series: Current Time Series
        """
        return self.current_time_series

    def initialize(self, iterator: "WindowedDatasetIterator"):
        """Initialize the current shuffler

        This method is in charge of initializing everything to
        allow the correct iteration

        Parameters:

            iterator: WindowedDatasetIterator

        """
        self.wditerator = iterator
        self._samples_per_time_series = (
            np.ones(self.wditerator.dataset.n_time_series, dtype=np.int64)
            * np.iinfo(np.int64).max
        )
        self._time_series_sizes = (
            np.ones(self.wditerator.dataset.n_time_series, dtype=np.int64)
            * np.iinfo(np.int64).max
        )
        self.current_time_series = 0

    def load_time_series(self, time_series_index: int):
        total_length = self.wditerator.dataset.number_of_samples_of_time_series(
            time_series_index
        )
        start_index = self.wditerator.start_index.get(total_length)
        end_index = self.wditerator.end_index.get(total_length)

        N = end_index - start_index
        samples_per_step = N / self.wditerator.step
        self._samples_per_time_series[time_series_index] = math.ceil(samples_per_step)
        last_part_missing = (
            math.ceil((N - 1) / self.wditerator.step) - ((N - 1) / self.wditerator.step)
            > 0
        )
        if self.wditerator.last_point and last_part_missing:
            self._samples_per_time_series[time_series_index] += 1

        self._time_series_sizes[time_series_index] = total_length

    def number_samples_of_time_series(self, time_series_index: int) -> int:
        if self._samples_per_time_series[time_series_index] == np.iinfo(np.int64).max:
            self.load_time_series(time_series_index)
        return self._samples_per_time_series[time_series_index]

    def number_of_samples_of_current_time_series(self) -> int:
        """Obtain the number of samples for the current time series

        Returns:

            int: Number of samples

        """

        return self.number_samples_of_time_series(self.current_time_series)

    def timestamp(self) -> int:
        """Obtain a timestamp for the current run-to-failure cycle"""
        raise NotImplementedError

    def time_series_size(self, time_series_index: int):
        if self._time_series_sizes[time_series_index] == np.iinfo(np.int64).max:
            self.load_time_series(time_series_index)
        return self._time_series_sizes[time_series_index]

    def current_time_series_size(self):
        return self.time_series_size(self.current_time_series)

    def advance(self):
        """Abstract method for obtain the new pair (time series, timestamp)"""
        raise NotImplementedError


class IntraTimeSeriesShuffler(AbstractShuffler):
    """
    Each point of the time series is shuffled, but the TS are kept in order

        Iteration 1: | TS 1 | TS 1 | TS 1 | TS 2 | TS 2 | TS 2
                     |   3  |  1   |  2   |   2  |   3  |   1
        Iteration 2: | TS 1 | TS 1 | TS 1 | TS 2 | TS 2 | TS 2
                     |   1  |  3   |  2   |   3  |   2  |   1
    """

    def time_series_changed(self):
        self.current_timestamp_index = 0
        if self.current_time_series == self.wditerator.dataset.n_time_series:
            return
        self.timestamps = np.arange(
            start=0,
            stop=self.current_time_series_size(),
            step=self.wditerator.step,
            dtype=np.int64,
        )
        if (
            self.wditerator.last_point
            and self.timestamps[-1] != self.current_time_series_size() - 1
        ):
            self.timestamps = np.append(
                self.timestamps, self.current_time_series_size() - 1
            )
        np.random.shuffle(self.timestamps)

    def initialize(self, iterator: "WindowedDatasetIterator"):
        super().initialize(iterator)
        self.current_time_series = 0
        self.time_series_changed()
        self.n_time_series = iterator.dataset.n_time_series

    def timestamp(self) -> int:

        ret = self.timestamps[self.current_timestamp_index]
        return ret

    def advance(self):
        self.current_timestamp_index += 1

        if (
            self.current_timestamp_index
            >= self.number_of_samples_of_current_time_series()
        ):
            self.current_time_series += 1
            self.time_series_changed()


class TimeSeriesOrderShuffling(AbstractShuffler):
    """
    Time series are shuffled, but each point inside the time series kept  its order

        Iteration 1: | TS 1 | TS 1 | TS 1 | TS 2 | TS 2 | tS 2 |
                     |   1  | 2    |  3   |   1  |   2  |   3  |
        Iteration 2: | TS 2 | TS 2 | TS 2 | TS 1 | TS 1 | TS 1 |
                     |   1  |  2   |  3   |   1  |   2  |   3  |
    """

    def initialize(self, iterator: "WindowedDatasetIterator"):
        super().initialize(iterator)
        self.available_time_series = np.arange(
            iterator.dataset.n_time_series, dtype=np.int64
        )
        np.random.shuffle(self.available_time_series)
        self.available_time_series_index = 0
        self.current_timestamp = 0
        self.current_time_series = self.available_time_series[
            self.available_time_series_index
        ]

    def timestamp(self) -> int:
        ret = self.current_timestamp

        return ret

    def advance(self):

        if self.current_timestamp == self.current_time_series_size() - 1:
            self.time_series_changed()
        else:
            self.current_timestamp += self.wditerator.step
            self.current_timestamp = min(
                self.current_timestamp, self.current_time_series_size() - 1
            )

    def time_series_changed(self):
        self.current_timestamp = 0
        self.available_time_series_index += 1
        if self.available_time_series_index < len(self.available_time_series):
            self.current_time_series = self.available_time_series[
                self.available_time_series_index
            ]
        else:
            self.current_time_series = len(self.available_time_series)


class TimeSeriesOrderIntraSignalShuffling(AbstractShuffler):
    """
     Each point in the ts is shuffled, and the ts order are shuffled also.

    !!! note
    
        Iteration 1: | TS 1 | TS 1 | TS 1 | TS 2 | TS 2 | TS 2
                     |   3  | 2    |  1   |   1  |   3  |   2
        Iteration 2: | TS 2 | TS 2 | TS 2 | TS 1 | TS 1 | TS 1
                     |   3 |  1   |  2   |   3  |   1  |   2

    """

    def initialize(self, iterator: "WindowedDatasetIterator"):
        super().initialize(iterator)
        self.available_time_series = np.arange(
            iterator.dataset.n_time_series, dtype=np.int64
        )
        np.random.shuffle(self.available_time_series)
        self.available_time_series_index = 0
        self.current_time_series = self.available_time_series[
            self.available_time_series_index
        ]

        self.available_time_stamps = np.arange(
            start=0,
            stop=self.number_of_samples_of_current_time_series(),
            step=self.wditerator.step,
            dtype=np.int64,
        )
        self.available_time_stamps_index = 0

    def timestamp(self) -> int:
        ret = self.available_time_stamps[self.available_time_stamps_index]

        return ret

    def advance(self):
        self.available_time_stamps_index += 1
        if self.available_time_stamps_index == len(self.available_time_stamps):
            self.time_series_changed()

    def time_series_changed(self):
        self.available_time_stamps_index = 0
        self.available_time_series_index += 1
        if self.available_time_series_index < len(self.available_time_series):
            self.current_time_series = self.available_time_series[
                self.available_time_series_index
            ]
            self.available_time_stamps = np.arange(
                start=0,
                stop=self.current_time_series_size(),
                step=self.wditerator.step,
                dtype=np.int64,
            )

        else:
            self.current_time_series = len(self.available_time_series)


class InverseOrder(AbstractShuffler):
    """
    The data points will be fed in RUL decreasing order

        Iteration 1: | TS 2 | TS 1 | TS 2 | TS 1 | TS 2 | TS 1
                     |   4  | 3    |  3   |   2  |   2  |   1
    """

    def initialize(self, iterator: "WindowedDatasetIterator"):
        super().initialize(iterator)
        self.sizes = np.array(
            [self.time_series_size(i) for i in range(iterator.dataset.n_time_series)],
            dtype=np.int64,
        )

    def time_series(self) -> int:
        self.current_time_series = np.argmax(self.sizes)
        return self.current_time_series

    def timestamp(self):
        ret = self.sizes[self.current_time_series] - 1

        return ret

    def advance(self):
        self.sizes[self.current_time_series] = np.clip(
            np.int64(self.sizes[self.current_time_series]) - self.wditerator.step,
            0,
            np.inf,
        )
        if np.sum(self.sizes) == 0:
            self.current_time_series = len(self.sizes)


class AllShuffled(AbstractShuffler):
    """
    Everything is shuffled

        Iteration 1: | TS 1 | TS 2 | TS 2 | TS 1 | TS 1 | TS 2
                     |   3  | 2    |  1   |   1  |   2  |   3
    """

    def initialize(
        self, iterator: "WindowedDatasetIterator", evenly_sampled: bool = True
    ):
        super().initialize(iterator)
        self.evenly_sampled = evenly_sampled
        self.timestamps_per_ts = {
            i: None for i in range(self.wditerator.dataset.n_time_series)
        }
        self.timestamps_per_ts_indices = np.array(
            [0 for i in range(self.wditerator.dataset.n_time_series)], dtype=np.int64
        )

    def time_series(self) -> int:
        l = np.random.randint(self.wditerator.dataset.n_time_series)
        while (
            self.number_samples_of_time_series(l) == self.timestamps_per_ts_indices[l]
        ):

            l = np.random.randint(self.wditerator.dataset.n_time_series)
        self.current_time_series = l

        return self.current_time_series

    def at_end(self) -> bool:
        return (self._samples_per_time_series == self.timestamps_per_ts_indices).all()

    def timestamp(self) -> int:
        return self.timestamps_per_ts[self.current_time_series][
            self.timestamps_per_ts_indices[self.current_time_series]
        ]

    def advance(self):
        self.timestamps_per_ts_indices[self.current_time_series] += 1

    def load_time_series(self, time_series_index: int):
        super().load_time_series(time_series_index)
        if self.evenly_sampled:
            total_length = self.time_series_size(time_series_index)
            final_index = self.wditerator.end_index.get(total_length)

            self.timestamps_per_ts[time_series_index] = np.arange(
                start=self.wditerator.start_index.get(total_length),
                stop=final_index,
                step=self.wditerator.step,
                dtype=np.int64,
            )
            if self.timestamps_per_ts[time_series_index][-1] != final_index - 1:
                self.timestamps_per_ts[time_series_index] = np.append(
                    self.timestamps_per_ts[time_series_index], final_index - 1
                )

        else:
            N = self.time_series_size(time_series_index)
            step = self.wditerator.step
            self.timestamps_per_ts[time_series_index] = N - np.logspace(
                0, 1, num=math.ceil(N / step), base=N
            ).astype("int")
        np.random.shuffle(self.timestamps_per_ts[time_series_index])


class NotShuffled(AbstractShuffler):
    """
    Nothing is shuffled. Each sample of each run-to-failure cycle
    are iterated in order

        Iteration 1: | Life 1 | Life 1  | Life 1 | Life 2 | Life 3 | Life 3
                     |   1    | 2       |  3     |   1    |   2    |   1
        Iteration 2: | Life 1 | Life 1  | Life 1 | Life 2 | Life 3 | Life 3
                     |   1    | 2       |  3     |   1    |   2    |   1
    """

    def initialize(self, iterator: "WindowedDatasetIterator"):
        super().initialize(iterator)
        self.current_time_series = 0

        self.current_timestamp = self.wditerator.start_index.get(
            self.current_time_series_size()
        )

    def time_series_changed(self):
        self.current_time_series += 1
        if not self.at_end():
            self.current_timestamp = self.wditerator.start_index.get(
                self.current_time_series_size()
            )

    def timestamp(self) -> int:
        return self.current_timestamp

    def advance(self):
        final_timestamp = self.wditerator.end_index.get(self.current_time_series_size())
        if self.current_timestamp == final_timestamp - 1:
            self.time_series_changed()
        else:
            self.current_timestamp += self.wditerator.step
            self.current_timestamp = min(self.current_timestamp, final_timestamp - 1)
