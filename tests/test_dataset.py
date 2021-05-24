from rul_pm.dataset.analysis import null_proportion, variance_information
import numpy as np
import pandas as pd
from rul_pm.dataset.lives_dataset import AbstractLivesDataset, FoldedDataset
from rul_pm.dataset.CMAPSS import CMAPSSDataset, sensor_indices

class MockDataset(AbstractLivesDataset):
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

    def get_life(self, i: int):
        return self.lives[i]

    @property
    def rul_column(self):
        return "RUL"

    @property
    def nlives(self):
        return len(self.lives)





class TestDataset:
    def test_dataset(self):
        ds = MockDataset(5)
        columns = [set(["feature1", "feature2", "RUL", "life"]) for i in range(4)]
        columns.append(set(["feature1", "feature2", "feature3", "RUL", "life"]))
        for life, columns in zip(ds, columns):
            assert set(life.columns) == columns
        assert ds.nlives == 5

        p = ds.toPandas()
        assert p.shape[0] == 50 * 5
        assert set(ds.commonFeatures()) == set(["feature1", "feature2", "RUL", "life"])

        folded = ds[[3, 2, 1]]
        assert isinstance(folded, FoldedDataset)
        assert folded[0][["feature1", "feature2"]].equals(
            ds[3][["feature1", "feature2"]]
        )
        assert not folded[1][["feature1", "feature2"]].equals(
            ds[3][["feature1", "feature2"]]
        )

    def test_CMAPSSDataset(self):
        ds = CMAPSSDataset()
        assert len(ds) == 709
        life = ds[0]
        assert len(life.columns) == 29


class TestAnalysis:
    def test_analysis(self):
        class MockCorruptedDataset(AbstractLivesDataset):
            def __init__(self,):

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

            def get_life(self, i: int):
                return self.lives[i]

            @property
            def rul_column(self):
                return "RUL"

            @property
            def nlives(self):
                return len(self.lives)

        ds = MockCorruptedDataset()
        df, null_per_life = null_proportion(ds)
        assert null_per_life['feature1'][0] == 0.5
        assert null_per_life['feature2'][1] == 0


        df, var_per_life = variance_information(ds)
        assert var_per_life['feature2'][1] == 0



class TestCMAPSS:
    def test_CMAPSS(self):
        train_dataset = CMAPSSDataset(train=True, models=['FD001'])    
        sensors_from_article = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17,20, 21]

        assert len(train_dataset) == 100

        features = np.array([train_dataset[0].columns[i] for i in sensor_indices])

        labels_true = np.array([f'SensorMeasure{f}' for f in sensors_from_article])

        assert (features == labels_true).all()