import numpy as np
import pandas as pd
from rul_pm.dataset.lives_dataset import AbstractLivesDataset
from rul_pm.iterators.batcher import Batcher
from rul_pm.iterators.iterators import WindowedDatasetIterator
from rul_pm.iterators.utils import true_values
from rul_pm.models.baseline import BaselineModel
from rul_pm.models.keras import KerasTrainableModel
from rul_pm.models.keras.models.simple import FCN
from rul_pm.models.sklearn import SKLearnModel
from rul_pm.models.xgboost_ import XGBoostModel
from rul_pm.transformation.features.scalers import PandasMinMaxScaler
from rul_pm.transformation.features.selection import ByNameFeatureSelector
from rul_pm.transformation.transformers import (LivesPipeline, Transformer)
from sklearn.linear_model import ElasticNet
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten


import random
random.seed(42)


import numpy as np
np.random.seed(42)


from tensorflow.python.framework import random_seed
random_seed.set_seed(42)


class MockDataset(AbstractLivesDataset):
    def __init__(self, nlives: int):

        self.lives = [
            pd.DataFrame({
                'feature1': np.linspace(0, 100, 50),
                'feature2': np.linspace(-25, 500, 50),
                'RUL': np.linspace(100, 0, 50)
            }) for i in range(nlives - 1)
        ]

        self.lives.append(
            pd.DataFrame({
                'feature1': np.linspace(0, 100, 50),
                'feature2': np.linspace(-25, 500, 50),
                'feature3': np.linspace(-25, 500, 50),
                'RUL': np.linspace(100, 0, 50)
            }))

    def get_life(self, i: int):
        return self.lives[i]

    @property
    def rul_column(self):
        return 'RUL'

    @property
    def nlives(self):
        return len(self.lives)



class MockDataset1(AbstractLivesDataset):
    def __init__(self, nlives: int):

        self.lives = [
            pd.DataFrame({
                'feature1': np.linspace(0, 100, 50),
                'feature2': np.linspace(-25, 500, 50),
                'RUL': np.linspace(100, 0, 50)
            }) for i in range(nlives - 1)
        ]

        self.lives.append(
            pd.DataFrame({
                'feature1': np.linspace(0, 100, 50),
                'feature2': np.linspace(-25, 500, 50),
                'feature3': np.linspace(-25, 500, 50),
                'RUL': np.linspace(5000, 0, 50)
            }))

    def get_life(self, i: int):
        return self.lives[i]

    @property
    def rul_column(self):
        return 'RUL'

    @property
    def nlives(self):
        return len(self.lives)

class TestKeras():
    def test_keras(self):

        class MockModel(KerasTrainableModel):
            
            def build_model(self, input_shape):

                input = Input(shape=input_shape)
                x = input  
                x = Flatten()(x)   
                x = Dense(5, activation='relu')(x)
                x = Dense(1)(x)        
                return Model(inputs=[input], outputs=[x])
        
        features = ['feature1', 'feature2']
        pipe = ByNameFeatureSelector(features)
        pipe = PandasMinMaxScaler((-1, 1))(pipe)
        rul_pipe =  ByNameFeatureSelector(['RUL'])
        transformer = Transformer(
            pipe.build(),
            rul_pipe.build()
        )
        ds = MockDataset(5)
        transformer.fit(ds)
        train_dataset = ds[range(0,4)]
        val_dataset = ds[range(4,5)]


        train_batcher = Batcher.new(train_dataset,
                                        window=1,
                                        step=1,                                        
                                        batch_size=64,
                                        transformer=transformer,
                                        shuffle='all')

        val_batcher = Batcher.new(val_dataset,
                                window=1,
                                step=1,
                                batch_size=64,
                                transformer=transformer,
                                shuffle=False,
                                restart_at_end=False)



        model = MockModel(learning_rate=0.8)
        model.fit(train_batcher, val_batcher, epochs=50)
        y_pred = model.predict(val_batcher)
        y_true = true_values(val_batcher)


        mse = np.mean((y_pred.ravel() - y_true.ravel())**2)

        assert mse < 2

    def test_baselines(self):
        features = ['feature1', 'feature2']
        pipe = ByNameFeatureSelector(features)
        pipe = PandasMinMaxScaler((-1, 1))(pipe)
        rul_pipe =  ByNameFeatureSelector(['RUL'])
        transformer = Transformer(
            pipe.build(),
            rul_pipe.build()
        )
        ds = MockDataset(5)
        transformer.fit(ds)
        model = BaselineModel(transformer, mode='mean')
        model.fit(ds)
        
        y_pred = model.predict(ds[[0]])
        assert np.all(np.diff(y_pred) < 0)
        
        
        y_pred = model.predict(ds)
        y_true = model.true_values(ds)   
        
        assert y_pred.shape[0] == y_true.shape[0]

        assert np.mean(np.abs(y_pred - y_true)) < 0.001

        model = BaselineModel(transformer, mode='median')
        model.fit(ds)

        y_pred = model.predict(ds)
        y_true = model.true_values(ds)

        assert y_pred.shape[0] == y_true.shape[0]

        ds = MockDataset1(5)
        transformer.fit(ds)
        model = BaselineModel(transformer, mode='mean')
        model.fit(ds)

        assert model.fitted_RUL > 100


        model = BaselineModel(transformer, mode='median')
        model.fit(ds)

        assert model.fitted_RUL == 100

    def test_models(self):
        features = ['feature1', 'feature2']
        pipe = ByNameFeatureSelector(features)
        pipe = PandasMinMaxScaler((-1, 1))(pipe)
        rul_pipe =  ByNameFeatureSelector(['RUL'])
        transformer = Transformer(
            pipe.build(),
            rul_pipe.build()
        )
        ds = MockDataset(5)
        transformer.fit(ds)
        train_dataset = ds[range(0,4)]
        val_dataset = ds[range(4,5)]


        train_batcher = Batcher.new(train_dataset,
                                        window=1,
                                        step=1,                                        
                                        batch_size=4,
                                        transformer=transformer,
                                        shuffle='all')

        val_batcher = Batcher.new(val_dataset,
                                window=1,
                                step=1,
                                batch_size=64,
                                transformer=transformer,
                                shuffle=False,
                                restart_at_end=False)



        model = FCN(learning_rate=0.01, 
                    layers_sizes=[16, 8], 
                    dropout=0.01, 
                    l2=0,
                    batch_normalization=False)
        model.fit(train_batcher, val_batcher, epochs=35)
        y_pred = model.predict(val_batcher)
        y_true = true_values(val_batcher)


        mse = np.mean((y_pred.ravel() - y_true.ravel())**2)

        assert mse < 2




class TestSKLearn():
    def test_sklearn(self):
        features = ['feature1', 'feature2']
        transformer = Transformer(
            LivesPipeline(steps =[
                    ('ss', ByNameFeatureSelector(features)), 
                    ('scaler', PandasMinMaxScaler((-1, 1)))
            ]),
            ByNameFeatureSelector(['RUL']).build())

        ds = MockDataset(5)
        transformer.fit(ds)
        ds_iterator = WindowedDatasetIterator(ds,
                                 window_size=1,
                                 step=1,
                                 transformer=transformer,
                                 shuffle=False)
        model = SKLearnModel(model=ElasticNet(alpha=0.1,
                                              l1_ratio=1,
                                              tol=0.00001, 
                                              max_iter=10000000), )
        
        model.fit(ds_iterator)
        y_pred = model.predict(ds_iterator)
        y_true = true_values(ds_iterator)


        rmse = np.sqrt(np.mean((y_pred.ravel() - y_true.ravel())**2))
        assert rmse < 0.5


class TestXGBoost():
    def test_xgboost(self):
        features = ['feature1', 'feature2']
        transformer = Transformer(
            LivesPipeline(steps =[
                    ('ss', ByNameFeatureSelector(features)), 
                    ('scaler', PandasMinMaxScaler((-1, 1)))
            ]),
            ByNameFeatureSelector(['RUL']).build())
        ds = MockDataset(5)
        transformer.fit(ds)
        ds_iterator = WindowedDatasetIterator(ds,
                                             window_size=1,
                                             step=2,
                                             transformer=transformer,
                                             shuffle='all')
        model = XGBoostModel()

        model.fit(ds_iterator)

        y_pred = model.predict(ds_iterator)
        y_true = true_values(ds_iterator)

        assert np.sum(y_pred - y_true) < 0.001
