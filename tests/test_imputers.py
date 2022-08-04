

import numpy as np
import pandas as pd
import pytest
from ceruleo.transformation.features.imputers import (ForwardFillImputer,
                                                       MeanImputer, ApplyRollingImputer,
                                                       MedianImputer, NaNtoInf)


class TestImputers():
    def test_ApplyRollingImputer(self):
        def finite_max(x):
            return np.max(x[np.isfinite(x)])
        f = ApplyRollingImputer(window_size=3, func=finite_max)
        df = pd.DataFrame({
            'a': [0, np.inf, 12, -np.inf,  0.9, 15, 0.5, 0.3, 0.5],
            'b': [5, 6,   7,   5,   np.inf,   5,  6,   5,   45],
        })
        assert not pd.isnull(df['a'][1])
        assert not pd.isnull(df['a'][2])
        assert not pd.isnull(df['b'][4])
        f.partial_fit(df)
        df_new = f.transform(df)

        assert np.all(np.isfinite(df_new))
        assert np.all(df_new['a'] == np.array([0, 12, 12, 15, 0.9, 15, 0.5, 0.3, 0.5]))

    def test_PandasRemoveInf(self):

        remover = NaNtoInf()
        df = pd.DataFrame({
            'a': [0, np.inf, -np.inf, 0.1, 0.9, 15, 0.5, 0.3, 0.5],
            'b': [5, 6,   7,   5,   np.inf,   5,  6,   5,   45],
        })
        assert not pd.isnull(df['a'][1])
        assert not pd.isnull(df['a'][2])
        assert not pd.isnull(df['b'][4])

        df_new = remover.fit_transform(df)

        assert pd.isnull(df_new['a'][1])
        assert pd.isnull(df_new['a'][2])
        assert pd.isnull(df_new['b'][4])

    def test_PandasMedianImputer(self):

        remover = MedianImputer()
        df = pd.DataFrame({
            'a': [0, np.nan, np.nan, 0.1, 0.9, 15, 0.5, 0.3, 0.5],
            'b': [5, 6,   np.nan,   5,   9,   5,  6,   5,   45],
        })
        a_median = np.median([0, 0.1, 0.9, 15, 0.5, 0.3, 0.5])
        b_median = np.median([5, 6,   5,   9,   5,  6,   5,   45])

        df_new = remover.fit_transform(df)

        assert not (pd.isnull(df_new).any().any())
        assert df_new['a'][1] ==  pytest.approx(a_median)
        assert df_new['a'][2] == pytest.approx(a_median)
        assert df_new['b'][2] == pytest.approx(b_median)

    def test_PandasMeanImputer(self):

        remover = MeanImputer()
        df = pd.DataFrame({
            'a': [0, np.nan, np.nan, 0.1, 0.9, 15, 0.5, 0.3, 0.5],
            'b': [5, 6,   np.nan,   5,   9,   5,  6,   5,   45],
        })
        a_mean = np.mean([0, 0.1, 0.9, 15, 0.5, 0.3, 0.5])
        b_mean = np.mean([5, 6,   5,   9,   5,  6,   5,   45])

        df_new = remover.fit_transform(df)
        assert not (pd.isnull(df_new).any().any())

        assert df_new['a'][1] == pytest.approx(a_mean)
        assert df_new['a'][2] == pytest.approx(a_mean)

        assert df_new['b'][2] == pytest.approx(b_mean)

    def test_ForwardFillImputer(self):

        remover = ForwardFillImputer()
        df = pd.DataFrame({
            'a': [0, 0.5, np.nan, np.nan, 0.9, 15, 0.5, 0.3, 0.5],
            'b': [5, 6,   7,   5,   np.nan,   5,  6,   5,   45],
        })
        df_new = remover.fit_transform(df)
        assert not (pd.isnull(df_new).any().any())

        assert df_new['a'][1] == pytest.approx(0.5)
        assert df_new['a'][2] == pytest.approx(0.5)

        assert df_new['b'][4] == pytest.approx(5)
