

import pandas as pd
from rul_pm.transformation.outliers import (EWMAOutlierRemover,
                                            IQROutlierRemover,
                                            ZScoreOutlierRemover)


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
