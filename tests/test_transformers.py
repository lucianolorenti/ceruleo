

import numpy as np
import pandas as pd
import pytest
from rul_pm.dataset.lives_dataset import AbstractLivesDataset
from rul_pm.transformation.features.generation import (
    EMD, Accumulate, AccumulateEWMAOutOfRange, ChangesCounter, Difference)
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


class MockDataset2(AbstractLivesDataset):
    def __init__(self, nlives: int):
        N = 500
        self.lives = [
            pd.DataFrame({
                'feature1': np.random.randn(N)*0.5 + 2,
                'feature2': np.random.randn(N)*0.5 + 5,
                'RUL': np.linspace(100, 0, N)
            })
            for i in range(nlives-1)]

        self.lives.append(
            pd.DataFrame({
                'feature1': np.random.randn(N)*0.5 + 1,
                'feature2': np.random.randn(N)*0.5 + 5,
                'feature3': np.random.randn(N)*0.5 + 2,
                'feature4': np.random.randn(N)*0.5 + 4,
                'RUL': np.linspace(100, 0, N)
            })
        )

        for j in range(self.nlives-1):
            for k in range(5):
                p = np.random.randint(N)
                self.lives[j]['feature1'][p] = 5000
                self.lives[j]['feature2'][p] = 5000

        for k in range(5):
            p = np.random.randint(N)
            self.lives[-1]['feature1'][p] = 5000
            self.lives[-1]['feature2'][p] = 5000
            self.lives[-1]['feature3'][p] = 5000
            self.lives[-1]['feature4'][p] = 5000

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
    def test_EMD(self):
        emd = EMD(3)
        df = pd.DataFrame({
            'feature1': np.linspace(-50, 50, 500),
            'feature2': np.cos(np.linspace(-1, 50, 500)),
        })
        df_t = emd.fit_transform(df)
        assert len(df_t['feature1_1'].dropna()) == 0
        assert(df_t.shape[1] == 6)

    def test_ChangesCounter(self):

        df = pd.DataFrame({
            'feature1': ['a', 'a', 'a', 'b', 'b', 'a', 'a', 'c'],
            'feature2': ['a', 'a', 'b', 'b', 'c', 'c', 'c', 'c']
        })
        df_gt = pd.DataFrame({
            'feature1': [1, 1, 1, 2, 2, 3, 3, 4],
            'feature2': [1, 1, 2, 2, 3, 3, 3, 3]
        })
        t = ChangesCounter()
        df1 = t.fit_transform(df)
        assert (df1.equals(df_gt))

    def test_Accumulate(self):
        df = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': [2, 4, 6, 8],
            'c': [2, 2, 2, 2],
            'd': [1, 0, 1, 0]
        })
        transformer = Accumulate()
        df_new = transformer.fit_transform(df)

        assert df_new['a'].iloc[-1] == 10
        assert df_new['b'].iloc[-1] == 20
        assert df_new['c'].iloc[-1] == 8
        assert (df_new['d'].values == np.array([1, 1, 2, 2])).all()

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

        ds = MockDataset2(5)
        transformer = AccumulateEWMAOutOfRange()
        for life in ds:
            transformer.partial_fit(life)
        new_life = transformer.transform(ds[-1])

    def test_Difference(self):
        df = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': [2, 4, 6, 8],
            'c': [2, 2, 2, 2],
            'd': [1, 1, 1, 1]
        })
        with pytest.raises(ValueError):
            transformer = Difference(['a', 'b'], ['d'])

        transformer = Difference(['a', 'b'], ['c', 'd'])
        df_new = transformer.fit_transform(df)
        print(df_new.columns)
        assert (df_new['a'].values == np.array([1-2, 2-2, 3-2, 4-2])).all()
        assert (df_new['b'].values == np.array([2-1, 4-1, 6-1, 8-1])).all()
        assert (df_new.shape[1] == 2)
