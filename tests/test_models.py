import numpy as np
import pandas as pd
from numpy.random import seed
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from ceruleo.dataset.ts_dataset import AbstractTimeSeriesDataset
from ceruleo.iterators.iterators import WindowedDatasetIterator
from ceruleo.iterators.shufflers import AllShuffled
from ceruleo.iterators.utils import true_values
from ceruleo.models.keras.keras import tf_regression_dataset
from ceruleo.models.scikitlearn import (
    EstimatorWrapper,
    SKLearnTimeSeriesWindowTransformer,
    predict,
    train_model,
)
from ceruleo.transformation import Transformer
from ceruleo.transformation.features.scalers import MinMaxScaler
from ceruleo.transformation.features.selection import ByNameFeatureSelector
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from xgboost import XGBRegressor

seed(1)
import tensorflow as tf

tf.random.set_seed(2)


class SimpleDataset(AbstractTimeSeriesDataset):
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


class MockDataset(AbstractTimeSeriesDataset):
    def __init__(self, nlives: int):

        self.lives = [
            pd.DataFrame(
                {
                    "feature1": np.linspace(0, 5 * 100, 50),
                    "feature2": np.linspace(-25, 5 * 500, 50),
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


class TestModels:
    def test_models(self):
        features = ["feature1", "feature2"]
        x = ByNameFeatureSelector(features=features)
        x = MinMaxScaler(range=(-1, 1))(x)

        y = ByNameFeatureSelector(features=["RUL"])
        transformer = Transformer(x, y)
        batch_size = 15
        window_size = 5
        ds = MockDataset(5)
        transformer.fit(ds)
        iterator = WindowedDatasetIterator(
            ds.map(transformer),
            window_size,
            step=1,
            horizon=1,
        )

        b1 = tf_regression_dataset(iterator).batch(15)
        assert b1.take(1)

    def test_sklearn(self):
        features = ["feature1", "feature2"]
        x = ByNameFeatureSelector(features=features)
        x = MinMaxScaler(range=(-1, 1))(x)

        y = ByNameFeatureSelector(features=["RUL"])
        transformer = lambda: Transformer(x, y)
        ds = MockDataset(5)

        transformer = SKLearnTimeSeriesWindowTransformer(transformer, window_size=5)
        pipe = make_pipeline(transformer, EstimatorWrapper(LinearRegression()))
        pipe.fit(ds)

        y_pred = pipe.predict(ds)
        y_true = transformer.true_values(ds)

        mse = np.sum((y_pred - y_true) ** 2)
        assert mse < 0.01

    def test_keras(self):

        features = ["feature1", "feature2"]
        pipe = ByNameFeatureSelector(features=features)
        pipe = MinMaxScaler(range=(-1, 1))(pipe)
        rul_pipe = ByNameFeatureSelector(features=["RUL"])
        transformer = Transformer(pipe, rul_pipe)
        ds = MockDataset(5)
        transformer.fit(ds)
        train_dataset = ds[range(0, 4)]
        val_dataset = ds[range(4, 5)]
        window_size = 6
        train_iterator = WindowedDatasetIterator(
            train_dataset.map(transformer),
            window_size=window_size,
            step=1,
            shuffler=AllShuffled(),
        )

        val_iterator = WindowedDatasetIterator(
            val_dataset.map(transformer), window_size=window_size, step=1
        )

        input = Input(shape=train_iterator.input_shape)
        x = input
        x = Flatten()(x)
        x = Dense(8, activation="relu")(x)
        x = Dense(5, activation="relu")(x)
        x = Dense(1)(x)
        model = Model(inputs=[input], outputs=[x])
        model.compile(loss="mae", optimizer=tf.keras.optimizers.SGD(0.01))
        y_true = true_values(val_iterator)
        y_pred_before_fit = model.predict(tf_regression_dataset(val_iterator).batch(64))
        mae_before_fit = np.mean(np.abs(y_pred_before_fit.ravel() - y_true.ravel()))
        model.fit(
            tf_regression_dataset(train_iterator).batch(4),
            validation_data=tf_regression_dataset(val_iterator).batch(64),
            epochs=15,
        )
        y_pred = model.predict(tf_regression_dataset(val_iterator).batch(64))
        

        mae = np.mean(np.abs(y_pred.ravel() - y_true.ravel()))

        assert mae < mae_before_fit

    def test_xgboost(self):
        features = ["feature1", "feature2"]

        x = ByNameFeatureSelector(features=features)
        x = MinMaxScaler(range=(-1, 1))(x)

        y = ByNameFeatureSelector(features=["RUL"])
        transformer = Transformer(x, y)

        ds = MockDataset(5)
        transformer.fit(ds)
        transformed_ds = ds.map(transformer)
        ds_iterator = WindowedDatasetIterator(
            transformed_ds, window_size=5, step=2, shuffler=AllShuffled()
        )
        model = XGBRegressor(n_estimators=500)
        train_model(model, ds_iterator)

        y_pred = predict(model, ds_iterator)
        y_true = true_values(ds_iterator)

        assert np.sum(y_pred - y_true) < 0.001
