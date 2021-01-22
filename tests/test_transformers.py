

import numpy as np
import pandas as pd
from rul_pm.transformation.features.selection import NullProportionSelector
from rul_pm.transformation.outliers import (EWMAOutlierRemover,
                                            IQROutlierRemover,
                                            ZScoreOutlierRemover)
from rul_pm.transformation.resamplers import SubSampleTransformer
from rul_pm.transformation.target import PicewiseRULThreshold


class TestTargetTransformers():
    def test_PicewiseRUL(self):
        t = PicewiseRULThreshold(26)
        d = np.vstack(
            (np.linspace(0, 30, 50),
             np.linspace(0, 30, 50)))

        d1 = t.fit_transform(d)
        assert d1.max() == 26


class TestTransformers():

    def test_IQROutlierRemover(self):

        remover = IQROutlierRemover(1.5, 1.0)
        df = pd.DataFrame({
            'a': [0, 0.5, 0.2, 0.1, 0.9, 15, 0.5, 0.3, 0.5],
            'b': [5, 6,   7,   5,   9,   5,  6,   5,   45],
        })
        df_new = remover.fit_transform(df)
        assert pd.isnull(df_new['a'][5])
        assert pd.isnull(df_new['b'][8])

    def test_ZScoreOutlierRemover(self):

        remover = ZScoreOutlierRemover(2)
        df = pd.DataFrame({
            'a': [0, 0.5, 0.2, 0.1, 0.9, 15, 0.5, 0.3, 0.5],
            'b': [5, 6,   7,   5,   9,   5,  6,   5,   45],
        })
        df_new = remover.fit_transform(df)
        assert pd.isnull(df_new['a'][5])
        assert pd.isnull(df_new['b'][8])

    def test_EWMAOutlierRemover(self):

        remover = EWMAOutlierRemover(0.5)
        df = pd.DataFrame({
            'a': [0, 0.5, 0.2, 0.1, 0.9, 15, 0.5, 0.3, 0.5],
            'b': [5, 6,   7,   5,   9,   5,  6,   5,   45],
        })
        df_new = remover.fit_transform(df)
        assert pd.isnull(df_new['a'][5])
        assert pd.isnull(df_new['b'][8])


class TestResamplers():
    def test_resampler(self):

        remover = SubSampleTransformer(2)
        df = pd.DataFrame({
            'a': [0, 0.5, 0.2, 0.1, 0.9, 15, 0.5, 0.3, 0.5, 12],
            'b': [5, 6,   7,   5,   9,   5,  6,   5,   45, 12],
        })
        df_new = remover.fit_transform(df)

        assert df_new.shape[0] == int(df.shape[0]/2)


class TestSelection():
    def test_selection(self):
        df = pd.DataFrame({
            'a': [0, 0.5, None, None],  # 0.5
            'b': [5, None, None, None],  # 0.75
            'c': [5, 2, 3, None],  # 0.25
        })

        selector = NullProportionSelector(0.3)
        df_new = selector.fit_transform(df)
        assert set(df_new.columns) == set(['a', 'c'])

        selector = NullProportionSelector(0.55)
        df_new = selector.fit_transform(df)
        assert set(df_new.columns) == set(['c'])
