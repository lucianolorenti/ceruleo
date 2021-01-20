

import numpy as np
import pandas as pd
from rul_pm.dataset.lives_dataset import AbstractLivesDataset
from rul_pm.iterators.batcher import get_batcher
from rul_pm.transformation.features.scalers import PandasMinMaxScaler
from rul_pm.transformation.features.selection import ByNameFeatureSelector
from rul_pm.transformation.transformers import (LivesPipeline, Transformer,
                                                transformation_pipeline)


class MockDataset(AbstractLivesDataset):
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

    def get_life(self, i: int):
        return self.lives[i]

    @property
    def rul_column(self):
        return 'RUL'

    @property
    def nlives(self):
        return len(self.lives)


class TestBatcher():
    def test_batcher(self):
        features = ['feature1', 'feature2']
        transformer = Transformer(
            'RUL',
            transformation_pipeline(
                numericals_pipeline=LivesPipeline(
                    steps=[
                        ('ss', ByNameFeatureSelector(features)),
                        ('scaler', PandasMinMaxScaler((-1, 1)))
                    ]),
                output_df=False)
        )
        batch_size = 15
        window_size = 5
        ds = MockDataset(5)
        b = get_batcher(ds, window_size, batch_size,
                        transformer, 1, restart_at_end=False)
        X, y, w = next(b)
        assert len(y.ravel()) == batch_size
        assert X.shape[0] == batch_size
        assert X.shape[1] == window_size
        assert X.shape[2] == 2
