import numpy as np
from ceruleo.transformation.target import PicewiseRUL


class TestTargetTransformers():
    def test_PicewiseRUL(self):
        t = PicewiseRUL(max_life=26)
        d = np.vstack(
            (np.linspace(0, 30, 50),
             np.linspace(0, 30, 50)))

        d1 = t.fit_transform(d)
        assert d1.max() == 26