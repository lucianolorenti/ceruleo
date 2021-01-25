

import numpy as np
import pandas as pd
from rul_pm.dataset.lives_dataset import AbstractLivesDataset
from rul_pm.transformation.features.generation import (
    Accumulate, AccumulateEWMAOutOfRange)
from rul_pm.transformation.features.selection import NullProportionSelector
from rul_pm.transformation.outliers import (EWMAOutlierRemover,
                                            IQROutlierRemover,
                                            ZScoreOutlierRemover)
from rul_pm.transformation.resamplers import SubSampleTransformer
from rul_pm.transformation.target import PicewiseRULThreshold


class MockDataset(AbstractLivesDataset):
    def __init__(self, nlives: int):

        self.lives = [
            pd.DataFrame({
                'feature1': np.linspace(0, 100, 50),
                'feature2': np.random.randint(2, size=(50,)),
                'RUL': np.linspace(100, 0, 50)
            })
            for i in range(nlives)]

    def get_life(self, i: int):
        return self.lives[i]

    @property
    def rul_column(self):
        return 'RUL'

    @property
    def nlives(self):
        return len(self.lives)


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


class TestGenerators:
    def test_Accumulate(self):
        df = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': [2, 4, 6, 8],
            'c': [1, 0, 1, 0],
        })
        transformer = Accumulate()
        df_new = transformer.fit_transform(df)

        assert df_new['a'].iloc[-1] == 10
        assert df_new['b'].iloc[-1] == 20
        assert df_new['c'].iloc[-1] == 2
        assert (df_new['c'].values == np.array([1, 1, 2, 2])).all()

        ds = MockDataset(5)
        transformer = Accumulate()
        transformer.fit(ds)

        life_0 = transformer.transform(ds.lives[0])
        life_1 = transformer.transform(ds.lives[1])

        assert (life_0['feature1'] == ds.lives[0]['feature1'].cumsum()).all()
        assert (life_0['feature2'] == ds.lives[0]['feature2'].cumsum()).all()

        assert (life_1['feature1'] == ds.lives[1]['feature1'].cumsum()).all()
        assert (life_1['feature2'] == ds.lives[1]['feature2'].cumsum()).all()

    def test_AccumulateEWMAOutOfRange(self):
        a = np.random.randn(500)*0.5 + 2
        b = np.random.randn(500)*0.5 + 5
        a[120] = 1500
        a[320] = 5000

        b[120] = 1500
        b[320] = 5000
        b[215] = 1500

        df = pd.DataFrame({
            'a': a,
            'b': b,
        })

        transformer = AccumulateEWMAOutOfRange()
        df_new = transformer.fit_transform(df)

        assert df_new['a'].iloc[-1] == 2
        assert df_new['b'].iloc[-1] == 3
