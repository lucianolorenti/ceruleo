

import numpy as np
import pandas as pd
from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset
from temporis.iterators.batcher import Batcher
from temporis.transformation.features.scalers import MinMaxScaler
from temporis.transformation.features.selection import ByNameFeatureSelector
from temporis.transformation import (TemporisPipeline, Transformer)
import math

class MockDataset(AbstractTimeSeriesDataset):
    def __init__(self, nlives: int):

        self.lives = [
            pd.DataFrame({
                'feature1': np.linspace(0, (i+1)*100, 50),
                'feature2': np.linspace(-25, (i+1)*500, 50),
                'RUL': np.linspace(100, 0, 50)
            })
            for i in range(nlives-1)]

        self.lives.append(
            pd.DataFrame({
                'feature1': np.linspace(0, 5*100, 50),
                'feature2': np.linspace(-25, 5*500, 50),
                'feature3': np.linspace(-25, 5*500, 50),
                'RUL': np.linspace(100, 0, 50)
            })
        )

    def get_time_series(self, i: int):
        return self.lives[i]

    @property
    def rul_column(self):
        return 'RUL'

    @property
    def n_time_series(self):
        return len(self.lives)


class TestBatcher():
    def test_batcher(self):
        features = ['feature1', 'feature2']
        x = ByNameFeatureSelector(features=features)
        x = MinMaxScaler(range=(-1, 1))(x)

        y = ByNameFeatureSelector(features=['RUL'])
        transformer = Transformer(x, y)
        
        batch_size = 15
        window_size = 5
        ds = MockDataset(5)
        transformer.fit(ds)
        b = Batcher.new(ds.map(transformer), window_size, batch_size, 1)

        expected_size = math.ceil(((5*50) - (4*5)) / 15)
        assert len(b) == expected_size
        X, y, w = next(b)
        assert len(y.ravel()) == batch_size
        assert X.shape[0] == batch_size
        assert X.shape[1] == window_size
        assert X.shape[2] == 2


        features = ['feature1', 'feature2']
        x = ByNameFeatureSelector(features=features)
        x = MinMaxScaler(range=(-1, 1))(x)

        y = ByNameFeatureSelector(features=['RUL'])
        transformer = Transformer(x, y)
        
        batch_size = 15
        window_size = 5
        ds = MockDataset(5)
        transformer.fit(ds)
        b = Batcher.new(ds.map(transformer), window_size, batch_size, 1, padding=True)

        expected_size = math.ceil(((5*50)) / 15)
        assert len(b) == expected_size
        X, y, w = next(b)
        assert len(y.ravel()) == batch_size
        assert X.shape[0] == batch_size
        assert X.shape[1] == window_size
        assert X.shape[2] == 2