import math

import numpy as np
from ceruleo.iterators.iterators import RelativeToEnd, RelativeToStart, WindowedDatasetIterator
from ceruleo.iterators.shufflers import (
    AbstractShuffler,
    AllShuffled,
    IntraTimeSeriesShuffler,
    InverseOrder,
    NotShuffled,
    TimeSeriesOrderIntraSignalShuffling,
    TimeSeriesOrderShuffling,
)


class MockDataFrame:
    def __init__(self, size):
        self.size = size

    @property
    def shape(self):
        return [self.size, None]


class MockDataset:
    def __init__(self):
        self.sizes = [5, 7, 3]

    def __getitem__(self, id: int):
        return MockDataFrame(self.sizes[id])

    def number_of_samples_of_time_series(self, id: int):
        return self.sizes[id]

    @property
    def n_time_series(self):
        return 3


class MockDatasetBig:
    def __init__(self, N:int = 6046):
        self.sizes = [55, 32, 59, 125, N]

    def __getitem__(self, id: int):
        return MockDataFrame(self.sizes[id])

    def number_of_samples_of_time_series(self, id: int):
        return self.sizes[id]

    @property
    def n_time_series(self):
        return 5


class MockDatasetMedium:
    def __init__(self):
        self.sizes = [155, 132, 159, 125, 604]

    def __getitem__(self, id: int):
        return MockDataFrame(self.sizes[id])

    def number_of_samples_of_time_series(self, id: int):
        return self.sizes[id]

    @property
    def n_time_series(self):
        return 5     


class MockIterator:
    def __init__(self,  step: int = 1, dataset=MockDataset(), last_point:bool = True):
        self.dataset = dataset
        self.step = step
        self.start_index = RelativeToStart(0)
        self.end_index = RelativeToEnd(0)
        self.last_point = last_point


class TestShufflers:
    def test_shufflers(self):

        x = NotShuffled()
        generated = [e for e in x.iterator(MockIterator())]
        expected = [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (2, 0),
            (2, 1),
            (2, 2),
        ]
        for g, e in zip(generated, expected):
            assert g == e

        x = NotShuffled()
        generated = [e for e in x.iterator(MockIterator(2))]
        expected = [
            (0, 0),
            (0, 2),
            (0, 4),
            (1, 0),
            (1, 2),
            (1, 4),
            (1, 6),
            (2, 0),
            (2, 2),
        ]
        for g, e in zip(generated, expected):
            assert g == e

    def test_intra_singal(self):
        np.random.seed(0)
        x = IntraTimeSeriesShuffler()
        iterator = MockIterator()
        generated = [e for e in x.iterator(iterator)]
        for i in range(3):
            samples_ts = [e for e in generated if e[0] == i]
            assert len(samples_ts) == iterator.dataset.sizes[i]
            ts_timestamps = [e[1] for e in samples_ts]
            assert ts_timestamps != list(range(0, iterator.dataset.sizes[i]))
            assert sorted(ts_timestamps) == list(range(0, iterator.dataset.sizes[i]))

        x = IntraTimeSeriesShuffler()
        step = 2
        iterator = MockIterator(step)
        generated = [e for e in x.iterator(iterator)]
        for i in range(3):
            samples_ts = [e for e in generated if e[0] == i]
            assert len(samples_ts) == math.ceil(iterator.dataset.sizes[i] / step)
            ts_timestamps = [e[1] for e in samples_ts]
            assert sorted(ts_timestamps) == list(
                np.arange(0, iterator.dataset.sizes[i], step=step)
            )

    def test_time_series_order(self):
        x = TimeSeriesOrderShuffling()
        it = MockIterator()
        generated = [e for e in x.iterator(it)]
        g = [g[0] for g in generated]
        assert len(np.where(np.diff(g))[0]) == 2
        i = [elem[1] for elem in generated if elem[0] == 2]
        assert i == list(range(0, it.dataset.sizes[2]))

        step = 2
        x = TimeSeriesOrderShuffling()
        it = MockIterator(step)
        generated = [e for e in x.iterator(it)]
        g = [g[0] for g in generated]
        assert len(np.where(np.diff(g))[0]) == 2
        i = [elem[1] for elem in generated if elem[0] == 2]
        assert i == np.arange(0, it.dataset.sizes[2], step=step).tolist()

    def test_TimeSeriesOrderIntraSignalShuffling(self):
        x = TimeSeriesOrderIntraSignalShuffling()
        it = MockIterator()
        generated = [e for e in x.iterator(it)]
        g = [g[0] for g in generated]
        assert len(np.where(np.diff(g))[0]) == 2

        step = 2
        x = TimeSeriesOrderIntraSignalShuffling()
        it = MockIterator(step)
        generated = [e for e in x.iterator(it)]
        g = [g[0] for g in generated]
        assert len(np.where(np.diff(g))[0]) == 2

    def test_InverseOrder(self):
        x = InverseOrder()
        it = MockIterator()
        generated = [e for e in x.iterator(it)]
        g = [g[1] for g in generated]
        assert np.all(np.diff(g) <= 0)

        step = 2
        x = InverseOrder()
        it = MockIterator(step)
        generated = [e for e in x.iterator(it)]
        g = [g[1] for g in generated]
        assert np.all(np.diff(g) <= 0)
        assert generated == [(1, 6), (0, 4), (1, 4), (0, 2), (1, 2), (2, 2), (0, 0), (1, 0), (2, 0)]

    def test_AllShuffled(self):
        x = AllShuffled()
        it = MockIterator()
        generated = [e for e in x.iterator(it)]
        assert len(generated) == np.sum(it.dataset.sizes)
        for i in range(3):
            g = [elem[0] for elem in generated if elem[0] == i]
            assert len(g) == it.dataset.sizes[i]

        step = 2
        x = AllShuffled()
        it = MockIterator(step)
        generated = [e for e in x.iterator(it)]
        assert len(generated) == np.sum(np.ceil(np.array(it.dataset.sizes) / step))
        for i in range(3):
            g = [elem[0] for elem in generated if elem[0] == i]
            assert len(g) == math.ceil(it.dataset.sizes[i] / step)


        step = 45
        x = AllShuffled()
        it = WindowedDatasetIterator(MockDatasetMedium(), window_size=5, step=step, shuffler=AllShuffled())
        q = [[e for e in x.iterator(it)] for i in range(250)]
        q1 = [len(b) for b in q]
        assert q1.count(q1[0]) == len(q1)


    def test_same_points(self):


        step = 5
        x = AllShuffled()
        it = WindowedDatasetIterator(MockDatasetBig(673), window_size=5, step=step)
        values1 = [e for e in x.iterator(it)]
        assert len(set(values1)) == len(values1)

        step = 5
        x = NotShuffled()
        it = WindowedDatasetIterator(MockDatasetBig(673), window_size=5, step=step)
        values2 = [e for e in x.iterator(it)]
        assert len(set(values2)) == len(values2)
        assert (sorted(values1, key=lambda x: (x[0], x[1])) == sorted(values2, key=lambda x: (x[0], x[1])))

        

