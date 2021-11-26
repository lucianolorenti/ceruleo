from curses import window
from typing import List, Optional

import numpy as np
import pandas as pd
import pytest
import scipy.stats
from ceruleo.dataset.ts_dataset import AbstractTimeSeriesDataset
from ceruleo.transformation.features.entropy import LocalEntropyMeasures
from ceruleo.transformation.features.extraction import (
    EMD,
    ChangesDetector,
    Difference,
    ExpandingStatistics,
    OneHotCategorical,
    RollingStatistics,
    SimpleEncodingCategorical,
    SlidingNonOverlappingEMD,
)
from ceruleo.transformation.features.outliers import (
    EWMAOutOfRange,
    IQROutlierRemover,
    ZScoreOutlierRemover,
    IsolationForestOutlierRemover,
)
from ceruleo.transformation.features.resamplers import IntegerIndexResamplerTransformer
from ceruleo.transformation.features.selection import (
    ByNameFeatureSelector,
    NullProportionSelector,
)
from ceruleo.transformation.features.transformation import Accumulate
from ceruleo.transformation.functional.pipeline.pipeline import Pipeline
from ceruleo.transformation.utils import QuantileEstimator


def manual_expanding(df: pd.DataFrame, min_points: int = 1):
    to_compute = [
        "kurtosis",
        "skewness",
        "max",
        "min",
        "std",
        "peak",
        "impulse",
        "clearance",
        "rms",
        "shape",
        "crest",
    ]
    dfs = []
    for c in df.columns:
        d = []
        for i in range(min_points - 1):
            d.append([np.nan for f in to_compute])
        for end in range(min_points, df.shape[0] + 1):
            data = df[c].iloc[:end]
            row = [manual_features(data, f) for f in to_compute]
            d.append(row)
        dfs.append(pd.DataFrame(d, columns=[f"{c}_{f}" for f in to_compute]))
    return pd.concat(dfs, axis=1)


def manual_rolling(df: pd.DataFrame, min_points: int = 1, window_size: int = 5):
    to_compute = [
        "kurtosis",
        "skewness",
        "max",
        "min",
        "std",
        "peak",
        "impulse",
        "clearance",
        "rms",
        "shape",
        "crest",
    ]
    dfs = []
    for c in df.columns:
        d = []
        for i in range(min_points - 1):
            d.append([np.nan for f in to_compute])
        for end in range(min_points, df.shape[0] + 1):
            data = df[c].iloc[max(end - window_size, 0) : end]
            row = [manual_features(data, f) for f in to_compute]
            d.append(row)
        dfs.append(pd.DataFrame(d, columns=[f"{c}_{f}" for f in to_compute]))
    return pd.concat(dfs, axis=1)


def kurtosis(s: pd.Series) -> float:
    return scipy.stats.kurtosis(s.values, bias=False)


def skewness(s: pd.Series) -> float:
    return scipy.stats.skew(s.values, bias=False)


def _max(s: pd.Series) -> float:
    return np.max(s.values)


def _min(s: pd.Series) -> float:
    return np.min(s.values)


def std(s: pd.Series) -> float:
    return np.std(s.values, ddof=1)


def peak(s: pd.Series) -> float:
    return max(s) - min(s)


def impulse(s: pd.Series) -> float:
    return peak(s) / np.mean(np.abs(s))


def clearance(s: pd.Series) -> float:
    return peak(s) / (np.mean(np.sqrt(np.abs(s))) ** 2)


def rms(s: pd.Series) -> float:
    return np.sqrt(np.mean(s**2))


def shape(s: pd.Series) -> float:
    return rms(s) / np.mean(np.abs(s))


def crest(s: pd.Series) -> float:
    return peak(s) / rms(s)


feature_functions = {
    "crest": crest,
    "shape": shape,
    "rms": rms,
    "clearance": clearance,
    "impulse": impulse,
    "peak": peak,
    "std": std,
    "min": _min,
    "max": _max,
    "skewness": skewness,
    "kurtosis": kurtosis,
}


def manual_features(s: pd.Series, name: str):
    return feature_functions[name](s)


class DatasetFromPandas(AbstractTimeSeriesDataset):
    def __init__(self, lives: List[pd.DataFrame]):

        self.lives = lives

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


class MockDataset2(AbstractTimeSeriesDataset):
    def __init__(self, nlives: int):
        N = 500
        self.lives = [
            pd.DataFrame(
                {
                    "feature1": np.random.randn(N) * 0.5 + 2,
                    "feature2": np.random.randn(N) * 0.5 + 5,
                    "RUL": np.linspace(100, 0, N),
                }
            )
            for i in range(nlives - 1)
        ]

        self.lives.append(
            pd.DataFrame(
                {
                    "feature1": np.random.randn(N) * 0.5 + 1,
                    "feature2": np.random.randn(N) * 0.5 + 5,
                    "feature3": np.random.randn(N) * 0.5 + 2,
                    "feature4": np.random.randn(N) * 0.5 + 4,
                    "RUL": np.linspace(100, 0, N),
                }
            )
        )

        for j in range(self.nlives - 1):
            for k in range(5):
                p = np.random.randint(N)
                self.lives[j]["feature1"][p] = 5000
                self.lives[j]["feature2"][p] = 5000

        for k in range(5):
            p = np.random.randint(N)
            self.lives[-1]["feature1"][p] = 5000
            self.lives[-1]["feature2"][p] = 5000
            self.lives[-1]["feature3"][p] = 5000
            self.lives[-1]["feature4"][p] = 5000

    def get_life(self, i: int):
        return self.lives[i]

    @property
    def rul_column(self):
        return "RUL"

    @property
    def nlives(self):
        return len(self.lives)


class TestTransformers:
    def test_IQROutlierRemover(self):

        remover = IQROutlierRemover(clip=False)
        df = pd.DataFrame(
            {
                "a": [0, 0.5, 0.2, 0.1, 0.9, 15, 0.5, 0.3, 0.5],
                "b": [5, 6, 7, 5, 9, 5, 6, 5, -45],
            }
        )
        df_new = remover.fit_transform(df)
        assert np.isposinf(df_new["a"][5])
        assert np.isneginf(df_new["b"][8])

        remover = IQROutlierRemover(clip=False, prefer_partial_fit=True)
        df = pd.DataFrame(
            {
                "a": [0, 0.5, 0.2, 0.1, 0.9, 15, 0.5, 0.3, 0.5],
                "b": [5, 6, 7, 5, 9, 5, 6, 5, -45],
            }
        )
        df_new = remover.fit_transform(df)
        assert np.isposinf(df_new["a"][5])
        assert np.isneginf(df_new["b"][8])

    def test_ZScoreOutlierRemover(self):

        remover = ZScoreOutlierRemover(number_of_std_allowed=2)
        df = pd.DataFrame(
            {
                "a": [0, 0.5, 0.2, 0.1, 0.9, 15, 0.5, 0.3, 0.5],
                "b": [5, 6, 7, 5, 9, 5, 6, 5, 45],
            }
        )
        df_new = remover.fit_transform(df)
        assert pd.isnull(df_new["a"][5])
        assert pd.isnull(df_new["b"][8])

        remover = ZScoreOutlierRemover(number_of_std_allowed=2, prefer_partial_fit=True)
        df = pd.DataFrame(
            {
                "a": [0, 0.5, 0.2, 0.1, 0.9, 15, 0.5, 0.3, 0.5],
                "b": [5, 6, 7, 5, 9, 5, 6, 5, 45],
            }
        )
        df_new = remover.fit_transform(df)
        assert pd.isnull(df_new["a"][5])
        assert pd.isnull(df_new["b"][8])


class TestResamplers:
    def test_resampler(self):

        remover = SubSampleTransformer(2)
        df = pd.DataFrame(
            {
                "a": [0, 0.5, 0.2, 0.1, 0.9, 15, 0.5, 0.3, 0.5, 12],
                "b": [5, 6, 7, 5, 9, 5, 6, 5, 45, 12],
            }
        )
        df_new = remover.fit_transform(df)

        assert df_new.shape[0] == int(df.shape[0] / 2)


class TestSelection:
    def test_selection(self):
        df = pd.DataFrame(
            {
                "a": [0, 0.5, None, None],  # 0.5
                "b": [5, None, None, None],  # 0.75
                "c": [5, 2, 3, None],  # 0.25
            }
        )

        selector = NullProportionSelector(max_null_proportion=0.3)
        df_new = selector.fit_transform(df)
        assert set(df_new.columns) == set(["c"])

        selector = NullProportionSelector(max_null_proportion=0.55)
        df_new = selector.fit_transform(df)
        assert set(df_new.columns) == set(["a", "c"])

        selector = NullProportionSelector(max_null_proportion=0.8)
        df_new = selector.fit_transform(df)
        assert set(df_new.columns) == set(["a", "b", "c"])


class TestGenerators:
    def test_EMD(self):
        emd = EMD(n=3)
        df = pd.DataFrame(
            {
                "feature1": np.linspace(-50, 50, 500),
                "feature2": np.cos(np.linspace(-1, 50, 500)),
            }
        )
        df_t = emd.fit_transform(df)
        assert len(df_t["feature1_1"].dropna()) == 0
        assert df_t.shape[1] == 6

    def test_ChangesCounter(self):

        df = pd.DataFrame(
            {
                "feature1": ["a", "a", "a", "b", "b", "a", "a", "c"],
                "feature2": ["a", "a", "b", "b", "c", "c", "c", "c"],
            }
        )
        df_gt = pd.DataFrame(
            {"feature1": [1, 0, 0, 1, 0, 1, 0, 1], "feature2": [1, 0, 1, 0, 1, 0, 0, 0]}
        )
        t = ChangesDetector()
        df1 = t.fit_transform(df).astype("int")
        assert df1.equals(df_gt)

    def test_Accumulate(self):
        df = pd.DataFrame(
            {"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [2, 2, 2, 2], "d": [1, 0, 1, 0]}
        )
        transformer = Accumulate()
        df_new = transformer.fit_transform(df)

        assert df_new["a"].iloc[-1] == 10
        assert df_new["b"].iloc[-1] == 20
        assert df_new["c"].iloc[-1] == 8
        assert (df_new["d"].values == np.array([1, 1, 2, 2])).all()

        ds = MockDataset(5)
        transformer = Accumulate()
        transformer.fit(ds)

        life_0 = transformer.transform(ds.lives[0])
        life_1 = transformer.transform(ds.lives[1])

        assert (life_0["feature1"] == ds.lives[0]["feature1"].cumsum()).all()
        assert (life_0["feature2"] == ds.lives[0]["feature2"].cumsum()).all()

        assert (life_1["feature1"] == ds.lives[1]["feature1"].cumsum()).all()
        assert (life_1["feature2"] == ds.lives[1]["feature2"].cumsum()).all()

    def test_expanding(self):

        lives = [
            pd.DataFrame(
                {
                    "a": np.random.rand(50) * 100 * np.random.rand(50) ** 3,
                    "b": np.random.rand(50) * 100 * np.random.rand(50) ** 2,
                }
            )
        ]
        to_compute = [
            "kurtosis",
            "skewness",
            "max",
            "min",
            "std",
            "peak",
            "impulse",
            "clearance",
            "rms",
            "shape",
            "crest",
        ]
        expanding = ExpandingStatistics(to_compute=to_compute)

        ds_train = DatasetFromPandas(lives[0:2])
        ds_test = DatasetFromPandas(lives[2:2])

        expanding.fit(ds_train)

        pandas_t = expanding.transform(ds_train[0][["a", "b"]])
        fixed_t = manual_expanding(ds_train[0][["a", "b"]], 2)

        assert (pandas_t - fixed_t).mean().mean() < 1e-15

    def test_rolling(self):
        lives = [
            pd.DataFrame(
                {
                    "a": np.random.rand(50) * 100 * np.random.rand(50) ** 3,
                    "b": np.random.rand(50) * 100 * np.random.rand(50) ** 2,
                }
            )
        ]
        to_compute = [
            "kurtosis",
            "skewness",
            "max",
            "min",
            "std",
            "peak",
            "impulse",
            "clearance",
            "rms",
            "shape",
            "crest",
        ]
        rolling = RollingStatistics(to_compute=to_compute, min_points=2, window=5)

        ds_train = DatasetFromPandas(lives[0:2])
        ds_test = DatasetFromPandas(lives[2:2])

        rolling.fit(ds_train)

        pandas_t = rolling.transform(ds_train[0][["a", "b"]])
        fixed_t = manual_rolling(ds_train[0][["a", "b"]], 2, 5)

        assert (pandas_t - fixed_t).mean().mean() < 1e-10

        rolling = RollingStatistics(
            min_points=2,
            window=5,
            specific={"a": ["mean", "kurtosis"], "b": ["peak", "impulse"]},
        )
        rolling.fit(ds_train)
        pandas_t = rolling.transform(ds_train[0][["a", "b"]])
        assert sorted(pandas_t.columns) == sorted(['a_mean', 'a_kurtosis', 'b_peak', 'b_impulse'])

    def test_EWMAOutOfRange(self):
        a = np.random.randn(500) * 0.5 + 2
        b = np.random.randn(500) * 0.5 + 5
        a[120] = 1500
        a[320] = 5000

        b[120] = 1500
        b[320] = 5000
        b[215] = 1500

        df = pd.DataFrame(
            {
                "a": a,
                "b": b,
            }
        )
        transformer = ByNameFeatureSelector(features=["a", "b"])
        transformer = EWMAOutOfRange(return_mask=True)(transformer)
        transformer = Accumulate()(transformer)
        df_new = Pipeline(transformer).fit_transform(df)
        assert df_new["a"].iloc[-1] == 2
        assert df_new["b"].iloc[-1] == 3

        transformer = ByNameFeatureSelector(features=["a", "b"])
        transformer = EWMAOutOfRange(return_mask=False)(transformer)
        df_new = Pipeline(transformer).fit_transform(df)

        assert np.isnan(df_new["a"].iloc[320])
        assert np.isnan(df_new["b"].iloc[215])

    def test_IsolationForestOutlierRemover(self):
        a = np.random.randn(500) * 0.5 + 2
        b = np.random.randn(500) * 0.5 + 5
        a[120] = 1500
        a[320] = 5000

        b[120] = 1500
        b[320] = 5000
        b[215] = 1500

        df = pd.DataFrame(
            {
                "a": a,
                "b": b,
            }
        )
        transformer = ByNameFeatureSelector(features=["a", "b"])
        transformer = IsolationForestOutlierRemover()(transformer)
        df_new = Pipeline(transformer).fit_transform(df)

        assert np.isnan(df_new["a"].iloc[320])
        assert np.isnan(df_new["b"].iloc[215])

    def test_encodings(self):
        df = pd.DataFrame(
            {
                "a": ["c1", "c2", "c3", "c1", "c3"],
                "b": [1, 1, 1, 1, 1],
            }
        )
        df1 = pd.DataFrame(
            {
                "a": ["c1", "c1", "c2", "c1", "c4"],
                "b": [1, 1, 1, 1, 1],
            }
        )
        df2 = pd.DataFrame(
            {
                "a": ["c1", "c1", "c1", "c3", "c2"],
                "b": [1, 1, 1, 1, 1],
            }
        )
        df3 = pd.DataFrame(
            {
                "a": ["c1", "c1", "c1", "c3", "c5"],
                "b": [1, 1, 1, 1, 1],
            }
        )
        transformer = OneHotCategorical(feature="a")
        transformer.partial_fit(df)
        transformer.partial_fit(df1)

        df_t = transformer.transform(df2)
        df_true = pd.DataFrame(
            {
                "c1": [1, 1, 1, 0, 0],
                "c2": [0, 0, 0, 0, 1],
                "c3": [0, 0, 0, 1, 0],
                "c4": [0, 0, 0, 0, 0],
            }
        )

        assert (df_t == df_true).all().all()

        df_t = transformer.transform(df3)
        df_true = pd.DataFrame(
            {
                "c1": [1, 1, 1, 0, 0],
                "c2": [0, 0, 0, 0, 0],
                "c3": [0, 0, 0, 1, 0],
                "c4": [0, 0, 0, 0, 0],
            }
        )

        assert (df_t == df_true).all().all()

        transformer = SimpleEncodingCategorical(feature="a")
        transformer.partial_fit(df)
        transformer.partial_fit(df1)

        df_t = transformer.transform(df2)
        df_true = pd.DataFrame([0, 0, 0, 2, 1])

        assert np.all(df_true.values == df_t.values)

        df_t = transformer.transform(df3)
        df_true = pd.DataFrame([0, 0, 0, 2, -1])
        assert np.all(df_true.values == df_t.values)

    def test_Difference(self):

        df = pd.DataFrame(
            {"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [2, 2, 2, 2], "d": [1, 1, 1, 1]}
        )
        with pytest.raises(ValueError):
            transformer = Difference(feature_set1=["a", "b"], feature_set2=["d"])

        transformer = Difference(feature_set1=["a", "b"], feature_set2=["c", "d"])
        df_new = transformer.fit_transform(df)
        assert (df_new["a"].values == np.array([1 - 2, 2 - 2, 3 - 2, 4 - 2])).all()
        assert (df_new["b"].values == np.array([2 - 1, 4 - 1, 6 - 1, 8 - 1])).all()
        assert df_new.shape[1] == 2


class TestEntropy:
    def test_entropy(self):
        df = pd.DataFrame({"a": [0, 0, 1, 1, 1, 1, 0, 0, 0]})

        selector = LocalEntropyMeasures(2)
        df_new = selector.fit_transform(df)

        LOCAL_BLOCK_ENTROPY = np.array(
            [
                np.nan,
                1.4150375,
                3.0,
                1.4150375,
                1.4150375,
                1.4150375,
                3.0,
                1.4150375,
                1.4150375,
            ]
        )
        LOCAL_ENTROPY_RATE = np.array(
            [np.nan, np.nan, 1, 0.0, 0.5849625, 0.5849625, 1.5849625, 0.0, 1.0]
        )
        LOCAL_ACTIVE_INFORMATION = np.array(
            [
                np.nan,
                np.nan,
                -0.19264508,
                0.80735492,
                0.22239242,
                0.22239242,
                -0.36257008,
                1.22239242,
                0.22239242,
            ]
        )
        assert np.nansum(df_new["a_local_entropy_rate"] - LOCAL_ENTROPY_RATE) < 1e-6

        assert np.nansum(df_new["a_local_block_entropy"] - LOCAL_BLOCK_ENTROPY) < 1e-6

        assert (
            np.nansum(df_new["a_local_active_information"] - LOCAL_ACTIVE_INFORMATION)
            < 1e-6
        )


class TestQuantileEstimator:
    def test_quantile(self):

        A = pd.DataFrame({"A": np.random.randn(15000)*10, "B": np.random.randn(15000)*10})
        q = QuantileEstimator(tdigest_size=200)
        q.update(A)
        s = q.estimate_quantile(0.5)
        assert s.index.tolist() == ["A", "B"]
        assert (np.abs(s.A - A.quantile(0.5)["A"])) < 0.1
        assert (np.abs(s.B - A.quantile(0.5)["B"])) < 0.1

        s = q.estimate_quantile(0.1)

        assert (np.abs(s.A - A.quantile(0.1)["A"])) < 0.1
        assert (np.abs(s.B - A.quantile(0.1)["B"])) < 0.1

        B = pd.DataFrame({"A": np.random.randn(15000) + 5, "B": np.random.randn(15000)})
        q.update(B)
        s = q.estimate_quantile(0.5)
        assert s.index.tolist() == ["A", "B"]
        assert np.abs(s.A) - 5 < 0.1
        assert np.abs(s.B) < 0.1
        assert s.index.tolist() == ["A", "B"]

        assert q.estimate_quantile(0.5, "A") - 5 < 0.1


class TestEMD:
    def test_emd(self):
        A = pd.DataFrame({"A": np.random.randn(15000), "B": np.random.randn(15000)})
        q = SlidingNonOverlappingEMD(window_size=300, max_imfs=5)
        r = q.transform(A)
        assert r.shape[0] == A.shape[0]
        assert r.shape[1] == A.shape[1] * 5


class TestResamplers:
    def test_resampler(self):

        A = pd.DataFrame(
            {
                "time": np.linspace(500, 0, 25).astype(int),
                "B": np.linspace(0, 500, 25).astype(int),
            }
        )
        A.loc[5, "time"] = A.loc[4, "time"] - 5
        A.loc[10, "time"] = A.loc[9, "time"] - 12
        A.loc[17, "time"] = A.loc[16, "time"] - 35

        resampler = IntegerIndexResamplerTransformer(
            time_feature="time", steps=15, drop_time_feature=True
        )

        resampler.partial_fit(A)
        q = resampler.transform(A)
        assert q.columns.values.tolist() == ["B"]
        assert np.all(np.diff(q) == 15)

        resampler = IntegerIndexResamplerTransformer(
            time_feature="time", steps=15, drop_time_feature=False
        )

        resampler.partial_fit(A)
        q = resampler.transform(A)
        assert q.columns.values.tolist() == ["time", "B"]

        A = pd.DataFrame({"time": [5, 3, 0], "B": [10, 6, 0]})
        resampler = IntegerIndexResamplerTransformer(
            time_feature="time", steps=2, drop_time_feature=False
        )

        resampler.partial_fit(A)
        q = resampler.transform(A)
        assert np.sum(q["B"] - np.array([0, 4, 8])) < 0.000005
