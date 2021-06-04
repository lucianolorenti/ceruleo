
from typing import List

import numpy as np
import pandas as pd
import pytest
import scipy.stats
from rul_pm.dataset.lives_dataset import AbstractLivesDataset
from rul_pm.transformation.features.transformation import Accumulate
from rul_pm.transformation.features.extraction import (EMD, 
                                                       ChangesDetector,
                                                       Difference,
                                                       ExpandingStatistics, OneHotCategoricalPandas, SimpleEncodingCategorical)
from rul_pm.transformation.features.outliers import (EWMAOutlierRemover,
                                                     EWMAOutOfRange,
                                                     IQROutlierRemover,
                                                     ZScoreOutlierRemover)
from rul_pm.transformation.features.resamplers import SubSampleTransformer
from rul_pm.transformation.features.selection import (ByNameFeatureSelector,
                                                      NullProportionSelector)
from rul_pm.transformation.target import PicewiseRUL


def manual_expanding(df: pd.DataFrame, min_points:int= 1):
    to_compute = ['kurtosis', 'skewness', 'max', 'min', 'std', 'peak', 'impulse',
            'clearance', 'rms', 'shape', 'crest']
    dfs = []
    for c in df.columns:
        d = []
        for i in range(min_points-1):
            d.append([np.nan for f in to_compute])
        for end in range(min_points, df.shape[0]+1):
            data = df[c].iloc[:end]
            row = [manual_features(data, f) for f in to_compute]
            d.append(row)
        dfs.append(pd.DataFrame(d, columns=[f'{c}_{f}' for f in to_compute]))
    return pd.concat(dfs, axis=1)



def kurtosis(s: pd.Series) -> float:
    return scipy.stats.kurtosis(s.values, bias=False)


def skewness(s: pd.Series) -> float:
    return scipy.stats.skew(s.values, bias=False)


def max(s: pd.Series) -> float:
    return np.max(s.values)


def min(s: pd.Series) -> float:
    return np.min(s.values)


def std(s: pd.Series) -> float:
    return np.std(s.values, ddof=1)


def peak(s: pd.Series) -> float:
    return max(s) - min(s)


def impulse(s: pd.Series) -> float:
    return peak(s) / np.mean(np.abs(s))


def clearance(s: pd.Series) -> float:
    return peak(s) / (np.mean(np.sqrt(np.abs(s)))**2)


def rms(s: pd.Series) -> float:
    return np.sqrt(np.mean(s**2))


def shape(s: pd.Series) -> float:
    return rms(s) / np.mean(np.abs(s))


def crest(s: pd.Series) -> float:
    return peak(s) / rms(s)


feature_functions = {
    'crest': crest,
    'shape': shape,
    'rms': rms,
    'clearance': clearance,
    'impulse': impulse,
    'peak': peak,
    'std': std,
    'min': min,
    'max': max,
    'skewness': skewness,
    'kurtosis': kurtosis
}


def manual_features(s: pd.Series, name: str):
    return feature_functions[name](s)



class DatasetFromPandas(AbstractLivesDataset):
    def __init__(self,  lives: List[pd.DataFrame]):

        self.lives = lives

    def get_life(self, i: int):
        return self.lives[i]

    @property
    def rul_column(self):
        return 'RUL'

    @property
    def nlives(self):
        return len(self.lives)

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
        t = PicewiseRUL(26)
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
            'feature1': [1, 0, 0, 1, 0, 1, 0, 1],
            'feature2': [1, 0, 1, 0, 1, 0, 0, 0]
        })
        t = ChangesDetector()
        df1 = t.fit_transform(df).astype('int')
        print(df1)
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

    def test_expanding(self):

        lives = [
            pd.DataFrame({
            'a': np.random.rand(50) * 100 * np.random.rand(50)**3,
            'b': np.random.rand(50) * 100 * np.random.rand(50)**2,
          
        })
        ]

        expanding = ExpandingStatistics()

        ds_train = DatasetFromPandas(lives[0:2])
        ds_test = DatasetFromPandas(lives[2:2])

        expanding.fit(ds_train)

        pandas_t = expanding.transform(ds_train[0][['a', 'b']])
        fixed_t = manual_expanding(ds_train[0][['a', 'b']], 2)

        return (pandas_t-fixed_t).mean().mean() < 1e-17


        

    def test_EWMAOutOfRange(self):
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
        transformer = ByNameFeatureSelector(['a', 'b'])
        transformer = EWMAOutOfRange()(transformer)
        transformer = Accumulate()(transformer)
        df_new = transformer.build().fit_transform(df)
        assert df_new['a'].iloc[-1] == 2
        assert df_new['b'].iloc[-1] == 3

        # TODO improve test
        #ds = MockDataset2(5)
        #transformer = ByNameFeatureSelector(['feature1', 'feature2'])
        #transformer = EWMAOutOfRange()(transformer)
        #transformer = Accumulate()(transformer)
        #transformer = transformer.build()
        
        #transformer.fit(ds)
        #new_life = transformer.transform(ds[-1])
        

    def test_encodings(self):
        df = pd.DataFrame({
            'a': ['c1', 'c2', 'c3', 'c1', 'c3'],
            'b': [1,1,1,1,1],            
        })
        df1 = pd.DataFrame({
            'a': ['c1', 'c1', 'c2', 'c1', 'c4'],
            'b': [1,1,1,1,1],             
        })
        df2 = pd.DataFrame({
            'a': ['c1', 'c1', 'c1', 'c3', 'c2'],
            'b': [1,1,1,1,1],             
        })
        df3 = pd.DataFrame({
            'a': ['c1', 'c1', 'c1', 'c3', 'c5'],
            'b': [1,1,1,1,1],             
        })
        transformer = OneHotCategoricalPandas('a')
        transformer.partial_fit(df)
        transformer.partial_fit(df1)

        df_t = transformer.transform(df2)
        df_true = pd.DataFrame({
            'c1':[1,1,1,0,0],
            'c2':[0,0,0,0,1],
            'c3':[0,0,0,1,0],
            'c4':[0,0,0,0,0]
        })

        assert (df_t == df_true).all().all()

        df_t = transformer.transform(df3)
        df_true = pd.DataFrame({
            'c1':[1,1,1,0,0],
            'c2':[0,0,0,0,0],
            'c3':[0,0,0,1,0],
            'c4':[0,0,0,0,0]
        })

        assert (df_t == df_true).all().all()


        transformer = SimpleEncodingCategorical('a')
        transformer.partial_fit(df)
        transformer.partial_fit(df1)

        df_t = transformer.transform(df2)
        df_true = pd.DataFrame([0,0,0,2,1])
 
        assert np.all(df_true.values == df_t.values)

        df_t = transformer.transform(df3)
        df_true = pd.DataFrame([0,0,0,2,-1])
        assert np.all(df_true.values == df_t.values)




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
        assert (df_new['b'].values ==  np.array([2-1, 4-1, 6-1, 8-1])).all()
        assert (df_new.shape[1] == 2)
