import numpy as np
from rul_gcd.dataset.lives_dataset import AbstractLivesDataset


def windowed_signal_generator(signal_X, signal_y, i, window_size):
    initial = max(i - window_size, 0)
    signal_X_1 = signal_X[initial:i, :]
    signal_y_1 = signal_y[i]
    if signal_X_1.shape[0] < window_size:
        signal_X_1 = np.vstack((
            np.zeros((
                window_size - signal_X_1.shape[0],
                signal_X_1.shape[1])), 
            signal_X_1))
    return (signal_X_1, signal_y_1)


