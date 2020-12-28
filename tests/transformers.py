import unittest

import pandas as pd
from rul_pm.transformation.outliers import IQROutlierRemover
from rul_pm.utils.lrucache import LRUDataCache


class Transformers(unittest.TestCase):

    def test_IQROutlierRemover(self):

        remover = IQROutlierRemover(1.5, 1.0)
        df = pd.DataFrame({
            'a': [0, 0.5, 0.2, 0.1, 0.9, 15, 0.5, 0.3, 0.5],
            'b': [5, 6,   7,   5,   9,   5,  6,   5,   45],
        })
        df_new = remover.fit_transform(df)
        self.assertTrue(pd.isnull(df_new['a'][5]))
        self.assertTrue(pd.isnull(df_new['b'][8]))
