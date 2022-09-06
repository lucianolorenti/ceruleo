import numpy as np
from ceruleo.transformation.target import PicewiseRUL, RULBinarizer, TargetToClasses
import pandas as pd

class TestTargetTransformers():
    def test_PicewiseRUL(self):
        t = PicewiseRUL(max_life=26)
        d = np.linspace(0, 30, 50)
        d = pd.DataFrame({'RUL': d})
        d1 = t.fit_transform(d)
        assert d1.max().RUL == 26

    def test_classes(self):
        t = TargetToClasses(bins=[15, 50, 150, 200])
        d = pd.DataFrame({'RUL': np.linspace(0, 500, 50)})
        d1 = t.fit_transform(d)

        assert np.all(((d <=15 ).values == (d1 == 0).values))
        assert np.all(((((d >15)  & (d <= 50))).values == (d1 == 1).values))

    def test_binarizer(self):
        t = RULBinarizer(50)
        d = pd.DataFrame({'RUL': np.linspace(0, 500, 50)})
        d1 = t.fit_transform(d)
        assert np.all(((d > 50 ).values == (d1 == 1).values))