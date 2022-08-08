from pathlib import Path

import numpy as np
import pandas as pd
from ceruleo.dataset.catalog.CMAPSS import CMAPSSDataset, sensor_indices
from ceruleo.dataset.transformed import TransformedSerializedDataset
from ceruleo.dataset.ts_dataset import (AbstractTimeSeriesDataset,
                                         FoldedDataset)
from ceruleo.transformation import Transformer
from ceruleo.transformation.features.scalers import MinMaxScaler
from ceruleo.transformation.features.selection import ByNameFeatureSelector
from sklearn.model_selection import train_test_split

class MockDataset(AbstractTimeSeriesDataset):
    def __init__(self, nlives: int):
        super().__init__()
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





class TestDataset:
    def test_dataset(self):
        ds = MockDataset(5)
        columns = [set(["feature1", "feature2", "RUL"]) for i in range(4)]
        columns.append(set(["feature1", "feature2", "feature3", "RUL"]))
        for life, columns in zip(ds, columns):
            assert set(life.columns) == columns
        assert ds.n_time_series == 5

        p = ds.to_pandas()
        assert p.shape[0] == 50 * 5
        assert set(ds.common_features()) == set(["feature1", "feature2", "RUL"])

        folded = ds[[3, 2, 1]]
        assert isinstance(folded, FoldedDataset)
        assert folded[0][["feature1", "feature2"]].equals(
            ds[3][["feature1", "feature2"]]
        )
        assert not folded[1][["feature1", "feature2"]].equals(
            ds[3][["feature1", "feature2"]]
        )

        two_folds = ds[[3, 2, 1]][[1, 0]]
        assert two_folds.original_indices() == [2, 3 ]

    def test_CMAPSSDataset(self):
        ds = CMAPSSDataset()
        assert len(ds) == 709
        life = ds[0]
        assert len(life.columns) == 28

    def test_transformed_dataset(self):
        dataset = MockDataset(nlives=5)

        pipe = ByNameFeatureSelector(features=["feature1"])
        pipe = MinMaxScaler(range=(-1, 1))(pipe)

        target_pipe = ByNameFeatureSelector(features=["RUL"])

        transformer = Transformer(pipelineX=pipe, pipelineY=target_pipe)

        transformer.fit(dataset)

        transformed_dataset = dataset.map(transformer)
        transformed_dataset.preload()

        X = transformed_dataset.get_features_of_life(0, pandas=True)
        assert isinstance(X, pd.DataFrame)


        X = transformed_dataset.get_features_of_life(0, pandas=False)
        assert isinstance(X, np.ndarray)

        path = Path('./saved_dataset').resolve()
        transformed_dataset.save(path)

        transformed_serialized_dataset = TransformedSerializedDataset(path)
        assert len(transformed_serialized_dataset) == len(transformed_dataset)

        assert np.all(transformed_serialized_dataset[0][0] == transformed_dataset[0][0])
        assert np.all(transformed_serialized_dataset[1][0] == transformed_dataset[1][0])
        
        dataset = MockDataset(nlives=30)
        train_ds, test_ds = train_test_split(dataset, train_size=0.8)
        train_ds, val_ds = train_test_split(train_ds, train_size=0.8)

        len_train_ds = len(train_ds)
        len_val_ds = len(val_ds)
        len_test_ds = len(test_ds)

        pipe = ByNameFeatureSelector(features=["feature1"])
        pipe = MinMaxScaler(range=(-1, 1))(pipe)

        target_pipe = ByNameFeatureSelector(features=["RUL"])

        transformer = Transformer(pipelineX=pipe, pipelineY=target_pipe)

        transformer.fit(train_ds)

        train_path = Path('./saved_dataset/train/').resolve()
        val_path = Path('./saved_dataset/val/').resolve()
        test_path = Path('./saved_dataset/test/').resolve()
        train_ds.map(transformer).save(train_path)
        val_ds.map(transformer).save(val_path)
        test_ds.map(transformer).save(test_path)

        transformed_serialized_dataset = TransformedSerializedDataset(train_path)
        n = []
        for X, y, sw in transformed_serialized_dataset:
            n.append(y.shape[0])

        assert len(n) == len_train_ds
        assert len(transformed_serialized_dataset) == len_train_ds

        transformed_serialized_dataset = TransformedSerializedDataset(val_path)
        assert len(transformed_serialized_dataset) == len_val_ds

        transformed_serialized_dataset = TransformedSerializedDataset(test_path)
        assert len(transformed_serialized_dataset) == len_test_ds





class TestAnalysis:
    def test_analysis(self):
        class MockCorruptedDataset(AbstractTimeSeriesDataset):
            def __init__(self):
                super().__init__()
                self.lives = [
                    pd.DataFrame(
                        {
                            "feature1": [np.nan, np.nan, 3, 4],
                            "feature2": [np.nan, 2, 3, 4],
                            "RUL": [1, 2, 3, 4],
                        }
                    ),
                    pd.DataFrame(
                        {
                            "feature1": [0, np.nan, 3, 4],
                            "feature2": [2, 2, 2, 2],
                            "RUL": [1, 2, 3, 4],
                        }
                    ),
                ]

            def get_time_series(self, i: int):
                return self.lives[i]

            @property
            def rul_column(self):
                return "RUL"

            @property
            def n_time_series(self):
                return len(self.lives)

        #ds = MockCorruptedDataset()
        #df, null_per_life = null_proportion(ds)
        #assert null_per_life['feature1'][0] == 0.5
        #assert null_per_life['feature2'][1] == 0


        #df, var_per_life = variance_information(ds)
        #assert var_per_life['feature2'][1] == 0



class TestCMAPSS:
    def test_CMAPSS(self):
        train_dataset = CMAPSSDataset(train=True, models=['FD001'])    
        sensors_from_article = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17,20, 21]

        assert len(train_dataset) == 100

        features = np.array([train_dataset[0].columns[i] for i in sensor_indices])

        labels_true = np.array([f'SensorMeasure{f}' for f in sensors_from_article])

        assert (features == labels_true).all()