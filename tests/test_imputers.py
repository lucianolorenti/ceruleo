

import numpy as np
import pandas as pd
import pytest
from rul_pm.transformation.features.imputers import (ForwardFillImputer,
                                                     PandasMeanImputer,
                                                     PandasMedianImputer,
                                                     PandasRemoveInf)


class TestImputers():

    def test_PandasRemoveInf(self):

        remover = PandasRemoveInf()
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

        remover = PandasMedianImputer()
        df = pd.DataFrame({
            'a': [0, np.nan, np.nan, 0.1, 0.9, 15, 0.5, 0.3, 0.5],
            'b': [5, 6,   np.nan,   5,   9,   5,  6,   5,   45],
        })
        a_median = np.median([0, 0.1, 0.9, 15, 0.5, 0.3, 0.5])
        b_median = np.median([5, 6,   5,   9,   5,  6,   5,   45])

        df_new = remover.fit_transform(df)

        assert not (pd.isnull(df_new).any().any())
        assert pytest.approx(df_new['a'][1], a_median)
        assert pytest.approx(df_new['a'][2], a_median)
        assert pytest.approx(df_new['b'][2], b_median)

    def test_PandasMeanImputer(self):

        remover = PandasMeanImputer()
        df = pd.DataFrame({
            'a': [0, np.nan, np.nan, 0.1, 0.9, 15, 0.5, 0.3, 0.5],
            'b': [5, 6,   np.nan,   5,   9,   5,  6,   5,   45],
        })
        a_mean = np.mean([0, 0.1, 0.9, 15, 0.5, 0.3, 0.5])
        b_mean = np.mean([5, 6,   5,   9,   5,  6,   5,   45])

        df_new = remover.fit_transform(df)
        assert not (pd.isnull(df_new).any().any())

        assert pytest.approx(df_new['a'][1], a_mean)
        assert pytest.approx(df_new['a'][2], a_mean)

        assert pytest.approx(df_new['b'][2], b_mean)

    def test_ForwardFillImputer(self):

        remover = ForwardFillImputer()
        df = pd.DataFrame({
            'a': [0, 0.5, np.nan, np.nan, 0.9, 15, 0.5, 0.3, 0.5],
            'b': [5, 6,   7,   5,   np.nan,   5,  6,   5,   45],
        })
        df_new = remover.fit_transform(df)
        assert not (pd.isnull(df_new).any().any())

        assert pytest.approx(df_new['a'][1], 0.5)
        assert pytest.approx(df_new['a'][2], 0.5)

        assert pytest.approx(df_new['b'][4], 5)
