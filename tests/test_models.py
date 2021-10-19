import random

import numpy as np
import pandas as pd
from temporis.iterators.shufflers import AllShuffled
from rul_pm.models.baseline import BaselineModel
from rul_pm.models.keras.models.simple import (
    build_convolutional,
    build_FCN,
    build_recurrent,
)
from rul_pm.models.sklearn import predict, train_model
from sklearn.linear_model import ElasticNet
from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset
from temporis.iterators.batcher import Batcher
from temporis.iterators.iterators import WindowedDatasetIterator
from temporis.iterators.utils import true_values
from temporis.models.keras import keras_autoencoder_batcher, tf_regression_dataset
from temporis.transformation import Transformer
from temporis.transformation.features.scalers import PandasMinMaxScaler
from temporis.transformation.features.selection import ByNameFeatureSelector
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten
from xgboost import XGBRegressor

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


class TestKeras:
    def test_keras(self):

        features = ["feature1", "feature2"]
        pipe = ByNameFeatureSelector(features)
        pipe = PandasMinMaxScaler((-1, 1))(pipe)
        rul_pipe = ByNameFeatureSelector(["RUL"])
        transformer = Transformer(pipe, rul_pipe)
        ds = MockDataset(5)
        transformer.fit(ds)
        train_dataset = ds[range(0, 4)]
        val_dataset = ds[range(4, 5)]

        train_batcher = WindowedDatasetIterator(
            train_dataset.map(transformer),
            window_size=1,
            step=1,
            shuffler=AllShuffled(),
        )

        val_batcher = WindowedDatasetIterator(
            val_dataset.map(transformer), window_size=1, step=1
        )

        input = Input(shape=train_batcher.input_shape)
        x = input
        x = Flatten()(x)
        x = Dense(5, activation="relu")(x)
        x = Dense(1)(x)
        model = Model(inputs=[input], outputs=[x])
        model.compile(loss="mae", optimizer="Adam")
        model.fit(
            tf_regression_dataset(train_batcher).batch(2),
            validation_data=tf_regression_dataset(val_batcher).batch(64),
            epochs=50,
        )
        y_pred = model.predict(tf_regression_dataset(val_batcher).batch(64))
        y_true = true_values(tf_regression_dataset(val_batcher).batch(64))

        mse = np.mean((y_pred.ravel() - y_true.ravel()) ** 2)

        assert mse < 3

    def test_baselines(self):
        features = ["feature1", "feature2"]
        pipe = ByNameFeatureSelector(features)
        pipe = PandasMinMaxScaler((-1, 1))(pipe)
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

    def test_models(self):
        features = ["feature1", "feature2"]
        pipe = ByNameFeatureSelector(features)
        pipe = PandasMinMaxScaler((-1, 1))(pipe)
        rul_pipe = ByNameFeatureSelector(["RUL"])
        transformer = Transformer(pipe, rul_pipe)
        ds = MockDataset(5)
        transformer.fit(ds)
        train_dataset = ds[range(0, 4)]
        val_dataset = ds[range(4, 5)]

        train_batcher = WindowedDatasetIterator(
            train_dataset.map(transformer),
            window_size=1,
            step=1,
            shuffler=AllShuffled(),
        )

        val_batcher = WindowedDatasetIterator(
            val_dataset.map(transformer), window_size=1, step=1
        )

        model = build_FCN(
            input_shape=train_batcher.input_shape,
            layers_sizes=[16, 8],
            dropout=0.01,
            l2=0,
            batch_normalization=False,
        )
        train_batcher = tf_regression_dataset(train_batcher).batch(2)
        val_batcher = tf_regression_dataset(val_batcher).batch(64)
        model.compile(optimizer="adam", loss="mae")
        model.fit(train_batcher, validation_data=val_batcher, epochs=35)
        y_pred = model.predict(val_batcher)
        y_true = true_values(val_batcher)

        mse = np.mean((y_pred.ravel() - y_true.ravel()) ** 2)

        assert mse < 3


class TestSKLearn:
    def test_sklearn(self):
        features = ["feature1", "feature2"]

        x = ByNameFeatureSelector(features)
        x = PandasMinMaxScaler((-1, 1))(x)

        y = ByNameFeatureSelector(["RUL"])
        transformer = Transformer(x, y)

        ds = MockDataset(5)
        transformer.fit(ds)
        ds_iterator = WindowedDatasetIterator(
            ds.map(transformer), window_size=1, step=1
        )
        model = ElasticNet(alpha=0.1, l1_ratio=1, tol=0.00001, max_iter=10000000)

        train_model(model, ds_iterator)
        y_pred = predict(model, ds_iterator)
        y_true = true_values(ds_iterator)

        rmse = np.sqrt(np.mean((y_pred.ravel() - y_true.ravel()) ** 2))
        assert rmse < 0.5


class TestXGBoost:
    def test_xgboost(self):
        features = ["feature1", "feature2"]

        x = ByNameFeatureSelector(features)
        x = PandasMinMaxScaler((-1, 1))(x)

        y = ByNameFeatureSelector(["RUL"])
        transformer = Transformer(x, y)

        ds = MockDataset(5)
        transformer.fit(ds)
        transformed_ds = ds.map(transformer)
        ds_iterator = WindowedDatasetIterator(
            transformed_ds, window_size=1, step=2, shuffler=AllShuffled()
        )
        model = XGBRegressor()
        train_model(model, ds_iterator)

        y_pred = predict(model, ds_iterator)
        y_true = true_values(ds_iterator)

        assert np.sum(y_pred - y_true) < 0.001
