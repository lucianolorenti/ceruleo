

import numpy as np
from ceruleo.iterators.iterators import windowed_signal_generator


class TestWindowGeneratorTest:
    def test_signal_generator(self):
        N = 15
        X = np.zeros((N, 3))
        for j in range(3):
            X[:, j] = np.linspace(0, 25, num=N)

        y = np.linspace(0, 25, num=N)

        X_w, y_w = windowed_signal_generator(X, y, 3, 6, 1)

        assert y[3] == y_w
        assert X_w.shape[0] == 6
        assert X_w.shape[1] == 3
        assert X_w[-1, 0] == X[3, 0]

        X_w, y_w = windowed_signal_generator(X, y, 3, 6, 2)

        assert (np.squeeze(y[[3, 4]]) == np.squeeze(y_w)).all()

        X_w, y_w = windowed_signal_generator(X, y, 3, 6, 2, right_closed=False)

        assert X_w[-1, 0] == X[2, 0]
        assert (np.squeeze(y[[3, 4]]) == np.squeeze(y_w)).all()
