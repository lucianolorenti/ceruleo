import numpy as np
from rul_pm.dataset.lives_dataset import AbstractLivesDataset


def windowed_signal_generator(signal_X, signal_y, i:int, window_size:int, output_size:int=1):
    """
    Return a lookback window and the value to predict.

    Parameters
    ----------
    signal_X:
             Matrix of size (life_length, n_features) with the information of the life
    signal_y:
             Target feature of size (life_length)
    i: int
       Position of the value to predict
    window_size: int
                 Size of the lookback window
    Returns
    -------
    tuple (np.array, float)
    """
    initial = max(i - window_size, 0)
    signal_X_1 = signal_X[initial:i, :]
    if len(signal_y.shape) == 1:
        signal_y_1 = signal_y[i:min(i+output_size, signal_y.shape[0])]

        if signal_y_1.shape[0] < output_size:
            padding = np.array([signal_y_1[-1]]*(output_size - signal_y_1.shape[0]))
            signal_y_1 = np.hstack((signal_y_1, padding))
    else:
        signal_y_1 = signal_y[i:min(i+output_size, signal_y.shape[0]), :]        
        if signal_y_1.shape[0] < output_size:
            padding = signal_y_1[-1,:].tolist()*(output_size - signal_y_1.shape[0])
            signal_y_1 = np.hstack((signal_y_1, padding))
            


    if signal_X_1.shape[0] < window_size:
        signal_X_1 = np.vstack((
            np.zeros((
                window_size - signal_X_1.shape[0],
                signal_X_1.shape[1])), 
            signal_X_1))
    return (signal_X_1, signal_y_1)

