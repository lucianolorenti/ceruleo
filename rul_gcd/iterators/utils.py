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


def windowed_element_list(dataset:AbstractLivesDataset, window_size, step=1, shuffle=False):
    def list_signal_windows(data, window_size, step, shuffle):
        list_ranges = list(range(0, data.shape[0], step))
        oelements_ = []
        for i in list_ranges:
            if i - window_size >= 0:
                oelements_.append((life, i))
        if shuffle == 'signal':
            np.random.shuffle(oelements_)
        return oelements_

    oelements = []
    for life in range(dataset.nlives):
        data = dataset.get_life(life)
        oelements.extend(list_signal_windows(data, window_size, step, shuffle))
    if shuffle == 'all':
        np.random.shuffle(oelements)
    return oelements
