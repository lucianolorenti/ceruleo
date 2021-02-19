

import numpy as np
import pandas as pd
from rul_pm.dataset.lives_dataset import AbstractLivesDataset
from rul_pm.transformation.features.scalers import PandasMinMaxScaler
from rul_pm.transformation.features.selection import ByNameFeatureSelector
from rul_pm.transformation.features.transformation import MeanCentering
from rul_pm.transformation.transformers import Transformer
from rul_pm.transformation.transformerstep import \
    Concatenate as TransformationConcatenate


class MockDataset(AbstractLivesDataset):
    def __init__(self):

        self.lives = [
            pd.DataFrame({
                'a': [1, 2, 3, 4],
                'b': [2, 4, 6, 8],
                'RUL':[4, 3, 2, 1]
            }),

            pd.DataFrame({
                'a': [150, 5, 14, 24],
                'b': [-52, -14, -36, 8],
                'RUL':[4, 3, 2, 1]
            })
        ]

    def get_life(self, i: int):
        return self.lives[i]

    @property
    def rul_column(self):
        return 'RUL'

    @property
    def nlives(self):
        return len(self.lives)


class MockDataset1(AbstractLivesDataset):
    def __init__(self):

        self.lives = [
            pd.DataFrame({
                'a': [1, 2, 3, 4],
                'b': [1, 2, 3, 4],
                'RUL':[4, 3, 2, 1]
            }),

            pd.DataFrame({
                'a': [2, 4, 6, 8],
                'b': [2, 4, 6, 8],
                'RUL':[4, 3, 2, 1]
            })
        ]

    def get_life(self, i: int):
        return self.lives[i]

    @property
    def rul_column(self):
        return 'RUL'

    @property
    def nlives(self):
        return len(self.lives)


class TestPipeline():
    def test_FitOrder(self):

        dataset = MockDataset()

        pipe = ByNameFeatureSelector(['a', 'b'])
        pipe = MeanCentering()(pipe)
        pipe = PandasMinMaxScaler((-1, 1))(pipe)

        target_pipe = ByNameFeatureSelector(['RUL'])

        test_transformer = Transformer(
            transformerX=pipe.build(),
            transformerY=target_pipe.build()
        )
        test_transformer.fit(dataset)

        X, y, sw = test_transformer.transform(dataset[0])

        assert(X.shape[1] == 2)
        df_dataset = dataset.toPandas()

        centered_df = df_dataset[['a', 'b']] - df_dataset[['a', 'b']].mean()
        scaler = test_transformer.transformerX.steps[2][1]
        assert(scaler.data_min.equals(centered_df.min(axis=0)))
        assert(scaler.data_max.equals(centered_df.max(axis=0)))

    def test_PandasConcatenate(self):
        dataset = MockDataset1()

        pipe = ByNameFeatureSelector(['a'])
        pipe = PandasMinMaxScaler((-1, 1))(pipe)

        pipe2 = ByNameFeatureSelector(['b'])
        pipe2 = PandasMinMaxScaler((-5, 0))(pipe2)

        pipe = TransformationConcatenate()([pipe, pipe2])
        pipe = MeanCentering()(pipe)

        target_pipe = ByNameFeatureSelector(['RUL'])

        test_transformer = Transformer(
            transformerX=pipe.build(),
            transformerY=target_pipe.build()
        )
        test_transformer.fit(dataset)

        df = dataset.toPandas()[['a', 'b']]

        data_min = df.min()
        data_max = df.max()

        gt = ((df-data_min)/(data_max-data_min))
        gt['a'] = gt['a'] * (1-(-1)) + (-1)
        gt['b'] = gt['b'] * (0-(-5)) + (-5)
        gt = gt-gt.mean()

        X, y, sw = test_transformer.transform(dataset[0])

        assert((np.mean((gt.iloc[:4, :].values - X)**2)) < 0.0001)
        X, y, sw = test_transformer.transform(dataset[1])
        assert((np.mean((gt.iloc[4:, :].values - X)**2)) < 0.0001)

        assert(X.shape[1] == 2)
