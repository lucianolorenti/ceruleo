

from random import sample
import numpy as np
import pandas as pd
from ceruleo.dataset.analysis.numerical_features import analysis

from ceruleo.dataset.ts_dataset import AbstractTimeSeriesDataset
from ceruleo.dataset.analysis.correlation import correlation_analysis
from ceruleo.transformation.features.selection import ByNameFeatureSelector, ByTypeFeatureSelector
from ceruleo.transformation.functional.transformers import Transformer
from ceruleo.dataset.analysis.sample_rate import sample_rate, sample_rate_summary
from ceruleo.dataset.analysis.distribution import features_divergeces

class MockDataset(AbstractTimeSeriesDataset):
    def __init__(self, nlives: int):
        super().__init__()
        self.lives = [
            pd.DataFrame(
                {
                    "feature1": np.linspace(0, 100, 50),
                    "feature2": np.random.randint(2, size=(50,)),
                    "RUL": np.linspace(100, 0, 50),
                }
            )
            for i in range(nlives)
        ]

    def get_time_series(self, i: int):
        return self.lives[i]

    @property
    def rul_column(self):
        return "RUL"

    @property
    def n_time_series(self):
        return len(self.lives)


class TestAnalysis:
    def test_correlation(self):
        ds = MockDataset(5)
        assert isinstance(correlation_analysis(ds), pd.DataFrame)

        transformer = Transformer(
            pipelineX=ByNameFeatureSelector(features=['feature1', 'feature2']), 
            pipelineY=ByNameFeatureSelector(features=['RUL'])
        )

        transformed_dataset = transformer.fit_map(ds)
        assert isinstance(correlation_analysis(transformed_dataset), pd.DataFrame)

    def test_samplerate(self):
        dataset = MockDataset(5)
        sample_rates = sample_rate(dataset)
        assert isinstance(sample_rates, np.ndarray)

        assert isinstance(sample_rate_summary(dataset), pd.DataFrame)

    def test_distribution(self):
        dataset = MockDataset(5)
        assert isinstance(features_divergeces(dataset), pd.DataFrame)

    def test_analysis(self):
        dataset = MockDataset(5)
        df = analysis(dataset)
        assert isinstance(df, pd.DataFrame)
