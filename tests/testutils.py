import unittest
from rul_gcd.iterators.utils import windowed_signal_generator
import numpy as np

class WindowGeneratorTest(unittest.TestCase):
    def test_signal_generator(self):
        N = 15        
        X = np.zeros((N, 3))
        for j in range(3):
            X[:, j] = np.linspace(0, 50, num=N)

        y = np.linspace(0, 25, num=N)
        
        X_w, y_true = windowed_signal_generator(X, y, 3, 6)
        self.assertEqual(y[3], y_true)
        self.assertEqual(X_w.shape[0], 6)
        self.assertEqual(X_w.shape[1], 3)
        self.assertEqual(X_w[-1, 0], X[2, 0])

