import numpy as np
import pandas as pd
from pyexpat import features
from scipy.stats import entropy
from ceruleo.dataset.ts_dataset import AbstractTimeSeriesDataset
from ceruleo.transformation import Concatenate as TransformationConcatenate
from ceruleo.transformation import Transformer
from ceruleo.transformation.features.imputers import PerColumnImputer
from ceruleo.transformation.features.outliers import IQROutlierRemover
from ceruleo.transformation.features.scalers import MinMaxScaler
from ceruleo.transformation.features.selection import ByNameFeatureSelector
from ceruleo.transformation.features.split import SplitByCategory
from ceruleo.transformation.features.transformation import MeanCentering
from ceruleo.transformation.functional.concatenate import Concatenate
from ceruleo.transformation.functional.graph_utils import root_nodes
from ceruleo.transformation.functional.pipeline.pipeline import make_pipeline


def gaussian(N: int, mean: float = 50, std: float = 10):
    return np.random.randn(N) * std + mean


class MockDatasetCategorical(AbstractTimeSeriesDataset):
    def build_df(self):
        N = 50
        return pd.DataFrame(
            {
                "Categorical": ["a"] * N + ["b"] * N,
                "feature1": np.hstack(
                    (gaussian(N, self.mean_a_f1), gaussian(N, self.mean_b_f1))
                ),
                "feature2": np.hstack(
                    (gaussian(N, self.mean_a_f2), gaussian(N, self.mean_b_f2))
                ),
            }
        )

    def __init__(self, N: int = 5):
        super().__init__()
        self.mean_a_f1 = 50
        self.mean_b_f1 = -16

        self.mean_a_f2 = 90
        self.mean_b_f2 = 250

        self.lives = [self.build_df() for i in range(N)]
        life_4 = self.lives[4]
        life_4.loc[life_4.index[50], "feature1"] = 591212
        life_4.loc[life_4.index[21], "feature2"] = 591212

        life_3 = self.lives[3]
        life_3.loc[life_4.index[88], "feature1"] = 591212
        life_3.loc[life_4.index[25], "feature2"] = 591212

    def get_time_series(self, i: int):
        return self.lives[i]

    @property
    def rul_column(self):
        return "RUL"

    @property
    def n_time_series(self):
        return len(self.lives)


class MockDataset(AbstractTimeSeriesDataset):
    def __init__(self):
        super().__init__()
        self.lives = [
            pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "RUL": [4, 3, 2, 1]}),
            pd.DataFrame(
                {"a": [150, 5, 14, 24], "b": [-52, -14, -36, 8], "RUL": [4, 3, 2, 1]}
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


class MockDataset1(AbstractTimeSeriesDataset):
    def __init__(self):
        super().__init__()
        self.lives = [
            pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4], "RUL": [4, 3, 2, 1]}),
            pd.DataFrame({"a": [2, 4, 6, 8], "b": [2, 4, 6, 8], "RUL": [4, 3, 2, 1]}),
        ]

    def get_time_series(self, i: int):
        return self.lives[i]

    @property
    def rul_column(self):
        return "RUL"

    @property
    def n_time_series(self):
        return len(self.lives)


class MockDataset2(AbstractTimeSeriesDataset):
    def __init__(self, n: int = 5):
        super().__init__()
        self.lives = [
            pd.DataFrame(
                {"a": ["A", "A", "A", "A"], "b": [1, 2, 3, 4], "RUL": [4, 3, 2, 1]}
            )
            for i in range(n)
        ]

    def get_time_series(self, i: int):
        return self.lives[i]

    @property
    def rul_column(self):
        return "RUL"

    @property
    def n_time_series(self):
        return len(self.lives)


class TestPipeline:
    def test_FitOrder(self):

        dataset = MockDataset()

        pipe = ByNameFeatureSelector(features=["a", "b"])
        pipe = MeanCentering()(pipe)
        pipe = MinMaxScaler(range=(-1, 1), name="Scaler")(pipe)

        target_pipe = ByNameFeatureSelector(features=["RUL"])

        test_transformer = Transformer(pipelineX=pipe, pipelineY=target_pipe)

        test_transformer.fit(dataset)

        X, y, sw = test_transformer.transform(dataset[0])

        assert X.shape[1] == 2
        df_dataset = dataset.to_pandas()

        centered_df = df_dataset[["a", "b"]] - df_dataset[["a", "b"]].mean()
        scaler = test_transformer.pipelineX.find_node("Scaler")
        assert scaler.data_min.equals(centered_df.min(axis=0))
        assert scaler.data_max.equals(centered_df.max(axis=0))

    def test_FitOrder2(self):
        dataset = MockDataset()

        pipe_a = ByNameFeatureSelector(features=["a"])
        pipe_a = MeanCentering()(pipe_a)
        scaler_a = MinMaxScaler(range=(-1, 1), name="a")
        pipe_a = scaler_a(pipe_a)

        pipe_b = ByNameFeatureSelector(features=["b"])
        pipe_b = MeanCentering()(pipe_b)
        scaler_b = MinMaxScaler(range=(-1, 1), name="b")
        pipe_b = scaler_b(pipe_b)

        pipe = TransformationConcatenate()([pipe_a, pipe_b])

        target_pipe = ByNameFeatureSelector(features=["RUL"])

        test_transformer = Transformer(pipelineX=pipe, pipelineY=target_pipe)

        test_transformer.fit(dataset)

        X, y, sw = test_transformer.transform(dataset[0])

        assert X.shape[1] == 2
        df_dataset = dataset.to_pandas()
        centered_df = df_dataset[["a", "b"]] - df_dataset[["a", "b"]].mean()

        assert scaler_a.data_min.equals(centered_df.min(axis=0)[["a"]])
        assert scaler_b.data_max.equals(centered_df.max(axis=0)[["b"]])

    def test_Concatenate(self):
        dataset = MockDataset1()

        pipe = ByNameFeatureSelector(features=["a"])
        pipe = MinMaxScaler(range=(-1, 1))(pipe)

        pipe2 = ByNameFeatureSelector(features=["b"])
        pipe2 = MinMaxScaler(range=(-5, 0))(pipe2)

        pipe = TransformationConcatenate()([pipe, pipe2])
        pipe = MeanCentering()(pipe)

        target_pipe = ByNameFeatureSelector(features=["RUL"])

        test_transformer = Transformer(pipelineX=pipe, pipelineY=target_pipe)

        test_transformer.fit(dataset)

        df = dataset.to_pandas()[["a", "b"]]

        data_min = df.min()
        data_max = df.max()

        gt = (df - data_min) / (data_max - data_min)
        gt["a"] = gt["a"] * (1 - (-1)) + (-1)
        gt["b"] = gt["b"] * (0 - (-5)) + (-5)
        gt = gt - gt.mean()

        X, y, sw = test_transformer.transform(dataset[0])

        assert (np.mean((gt.iloc[:4, :].values - X.values) ** 2)) < 0.0001
        X, y, sw = test_transformer.transform(dataset[1])
        assert (np.mean((gt.iloc[4:, :].values - X.values) ** 2)) < 0.0001

        assert X.shape[1] == 2

    def test_subpipeline(self):
        dataset = MockDatasetCategorical()
        pipe = ByNameFeatureSelector(features=["Categorical", "feature1", "feature2"])
        bb = make_pipeline(
            IQROutlierRemover(lower_quantile=0.05, upper_quantile=0.95, clip=True),
            MinMaxScaler(range=(-1, 1)),
            PerColumnImputer(),
        )
        pipe = SplitByCategory(features="Categorical", pipeline=bb)(pipe)

        target_pipe = ByNameFeatureSelector(features=["RUL"])

        test_transformer = Transformer(pipelineX=pipe)
        test_transformer.fit(dataset)

        q = np.hstack([d[d["Categorical"] == "a"]["feature1"] for d in dataset])
        approx_cat_a_feature1_1_quantile = np.quantile(q, 0.05)
        approx_cat_a_feature1_3_quantile = np.quantile(q, 0.95)
        r = root_nodes(pipe)[0]
        IQR_Node = r.next[0].next[1].next[0]
        real_cat_a_feature1_1_quantile = IQR_Node.Q1["feature1"]
        real_cat_a_feature1_3_quantile = IQR_Node.Q3["feature1"]

        assert approx_cat_a_feature1_1_quantile - real_cat_a_feature1_1_quantile < 5
        assert approx_cat_a_feature1_3_quantile - real_cat_a_feature1_3_quantile < 5

        assert test_transformer.transform(dataset[4])[0]["feature1"].iloc[50] - 1 < 0.01
        assert test_transformer.transform(dataset[4])[0]["feature2"].iloc[21] - 1 < 0.01

        d = dataset[4]
        aa = d[d["Categorical"] == "a"]["feature1"]
        counts_before_transformation, _ = np.histogram(aa)
        counts_before_transformation = counts_before_transformation / np.sum(
            counts_before_transformation
        )

        bb = test_transformer.transform(dataset[4])[0]["feature1"]
        counts_after_transformation, _ = np.histogram(bb[:50])
        counts_after_transformation = counts_after_transformation / np.sum(
            counts_after_transformation
        )
        assert entropy(counts_before_transformation, counts_after_transformation) < 0.01

    def test_split_one_category(self):
        dataset_orig = MockDataset2()
        dataset = dataset_orig[0:5]

        pipe = ByNameFeatureSelector(features=["a", "b"])
        scaler_pipe = make_pipeline(MinMaxScaler(range=(-1, 1), name="Scaler"))
        pipe = SplitByCategory(
            features="a", pipeline=scaler_pipe, add_default=False
        )(pipe)
        pipe = MeanCentering()(pipe)

        pipe = MinMaxScaler(range=(-1, 1), name="Scaler2")(pipe)

        target_pipe = ByNameFeatureSelector(features=["RUL"])

        test_transformer = Transformer(pipelineX=pipe, pipelineY=target_pipe)

        test_transformer.fit(dataset)

        X, y, sw = test_transformer.transform(dataset[0])

        assert X.shape[1] == 1