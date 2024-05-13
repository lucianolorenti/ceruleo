import numpy as np
import pandas as pd
from ceruleo.dataset.ts_dataset import AbstractPDMDataset
from ceruleo.iterators.batcher import Batcher
from ceruleo.iterators.iterators import WindowedDatasetIterator
from ceruleo.iterators.sample_weight import (
    ExponentialDecay,
    InverseToLengthWeighted,
    NotWeighted,
    RULInverseWeighted,
)
from ceruleo.transformation import Transformer
from ceruleo.transformation.features.scalers import MinMaxScaler
from ceruleo.transformation.features.selection import ByNameFeatureSelector


class SimpleDataset(AbstractPDMDataset):
    def __init__(self):
        self.lives = [
            pd.DataFrame(
                {"feature1": np.array(range(0, 100)), "RUL": np.array(range(0, 100))}
            )
        ]

    def get_time_series(self, i: int):
        return self.lives[i]

    @property
    def rul_column(self):
        return "RUL"

    @property
    def n_time_series(self):
        return len(self.lives)


class MockDataset(AbstractPDMDataset):
    def __init__(self, nlives: int):
        self.lives = [
            pd.DataFrame(
                {
                    "feature1": np.linspace(0, (i + 1) * 100, 50),
                    "feature2": np.linspace(-25, (i + 1) * 500, 50),
                    "RUL": np.linspace(100, 0, 50),
                }
            )
            for i in range(nlives - 1)
        ]

        self.lives.append(
            pd.DataFrame(
                {
                    "feature1": np.linspace(0, 5 * 100, 50),
                    "feature2": np.linspace(-25, 5 * 500, 50),
                    "feature3": np.linspace(-25, 5 * 500, 50),
                    "RUL": np.linspace(100, 0, 50),
                }
            )
        )

    def get_time_series(self, i: int):
        return self.lives[i]

    @property
    def rul_column(self):
        return "RUL"

    @property
    def n_time_series(self):
        return len(self.lives)


class TestIterators:
    def test_iterators(self):
        features = ["feature1", "feature2"]
        x = ByNameFeatureSelector(features=features)
        x = MinMaxScaler(range=(-1, 1))(x)

        y = ByNameFeatureSelector(features=["RUL"])
        transformer = Transformer(x, y)
        batch_size = 15
        window_size = 5
        ds = MockDataset(5)

        transformer.fit(ds)
        b = Batcher.new(ds.map(transformer), window_size, batch_size, 1)
        X, y, w = next(b)
        assert len(y.ravel()) == batch_size
        assert X.shape[0] == batch_size
        assert X.shape[1] == window_size
        assert X.shape[2] == 2

    def test_2(self):
        dataset = SimpleDataset()
        pipe = ByNameFeatureSelector(features=["feature1"])
        y_pipe = ByNameFeatureSelector(features=["RUL"])
        transformer_raw = Transformer(pipelineX=pipe, pipelineY=y_pipe)
        transformer_raw.fit(dataset)
        it = WindowedDatasetIterator(dataset.map(transformer_raw), 5)
        X, y, sw = next(it)
        assert np.all(X == np.array([[0, 1, 2, 3, 4]]).T)
        assert y[0][0] == 4


def build_elements():
    a = np.linspace(10, 0, 11)
    b = np.vstack((a, a)).T
    c = pd.DataFrame(a, columns=["RUL"])
    d = pd.DataFrame(b, columns=["RUL", "RUL2"])
    e = pd.Series(a)
    return [a, b, c, d, e]


class TestSampleWeight:
    def test_not_weighted(self):
        nw = NotWeighted()
        for el in build_elements():
            assert nw(el, 0, None) == 1
            assert nw(el, 5, None) == 1

    def test_rul_inverse_weighted(self):
        inverse = RULInverseWeighted()
        for el in build_elements():
            assert inverse(el, 0, None) == 1.0/(10 + 1)
            assert inverse(el, 5, None) == 1.0/(5 + 1)
            assert inverse(el, 10, None) == 1.0

    def test_InverseToLengthWeighted(self):
        inverse = InverseToLengthWeighted()
        for el in build_elements():
            assert inverse(el, 0, None) == 1/(10.0)
            assert inverse(el, 5, None) ==  1/(10.0)

    def test_ExponentialDecay(self):
        exp = ExponentialDecay(near_0_at=3)
        for el in build_elements():
            assert exp(el, 0, None) < 0.00001 
            assert exp(el, 9, None) > 0.21