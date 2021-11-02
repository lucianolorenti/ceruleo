import random

import numpy as np
import pandas as pd
from temporis.iterators.shufflers import AllShuffled

from rul_pm.models.baseline import BaselineModel


from sklearn.linear_model import ElasticNet
from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset

from temporis.iterators.utils import true_values
from temporis.models.keras import tf_regression_dataset
from temporis.transformation import Transformer
from temporis.transformation.features.scalers import MinMaxScaler
from temporis.transformation.features.selection import ByNameFeatureSelector



random.seed(42)


import numpy as np

np.random.seed(42)


from tensorflow.python.framework import random_seed

random_seed.set_seed(42)


class MockDataset(AbstractTimeSeriesDataset):
    def __init__(self, nlives: int):

        self.lives = [
            pd.DataFrame(
                {
                    "feature1": np.linspace(0, 100, 50),
                    "feature2": np.linspace(-25, 500, 50),
                    "RUL": np.linspace(100, 0, 50),
                }
            )
            for i in range(nlives - 1)
        ]

        self.lives.append(
            pd.DataFrame(
                {
                    "feature1": np.linspace(0, 100, 50),
                    "feature2": np.linspace(-25, 500, 50),
                    "feature3": np.linspace(-25, 500, 50),
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


class MockDataset1(AbstractTimeSeriesDataset):
    def __init__(self, nlives: int):

        self.lives = [
            pd.DataFrame(
                {
                    "feature1": np.linspace(0, 100, 50),
                    "feature2": np.linspace(-25, 500, 50),
                    "RUL": np.linspace(100, 0, 50),
                }
            )
            for i in range(nlives - 1)
        ]

        self.lives.append(
            pd.DataFrame(
                {
                    "feature1": np.linspace(0, 100, 50),
                    "feature2": np.linspace(-25, 500, 50),
                    "feature3": np.linspace(-25, 500, 50),
                    "RUL": np.linspace(5000, 0, 50),
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


class TestModels:


    def test_baselines(self):
        features = ["feature1", "feature2"]
        pipe = ByNameFeatureSelector(features)
        pipe = MinMaxScaler((-1, 1))(pipe)
        rul_pipe = ByNameFeatureSelector(["RUL"])
        transformer = Transformer(pipe, rul_pipe)
        ds = MockDataset(5)
        transformer.fit(ds)
        ds_transformed = ds.map(transformer)
        model = BaselineModel(mode="mean")
        model.fit(ds_transformed)

        y_pred = model.predict(ds_transformed[[0]])
        assert np.all(np.diff(y_pred) < 0)

        y_pred = model.predict(ds_transformed)
        y_true = true_values(ds_transformed)

        assert y_pred.shape[0] == y_true.shape[0]

        assert np.mean(np.abs(np.squeeze(y_pred) - np.squeeze(y_true))) < 0.001

        model = BaselineModel(mode="median")
        model.fit(ds_transformed)

        y_pred = model.predict(ds_transformed)
        y_true = true_values(ds_transformed)

        assert y_pred.shape[0] == y_true.shape[0]

        ds = MockDataset1(5)
        transformer.fit(ds)
        transformed_ds = ds.map(transformer)
        model = BaselineModel(mode="mean")
        model.fit(transformed_ds)

        assert model.fitted_RUL > 100

        model = BaselineModel(mode="median")
        model.fit(transformed_ds)

        assert model.fitted_RUL == 100




