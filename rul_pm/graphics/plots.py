import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rul_pm.dataset.lives_dataset import AbstractLivesDataset
from rul_pm.iterators.iterators import LifeDatasetIterator


def plot_lives(ds: AbstractLivesDataset):
    """
    Plot each life
    """
    fig, ax = plt.subplots()
    it = LifeDatasetIterator(ds)
    for _, y in it:
        ax.plot(y)
    return fig, ax


def plot_errors_wrt_RUL(val_rul, pred_cont, treshhold=0, bins=15, **kwargs):
    """
    Plot errors with respect to the RUL

    Parameters
    ----------
    val_rul: np.array   
             Array of true RUL

    pred_cont: np.array
             Array of predicted RUL

    threshold: float
             Threshold to use for clipping the RUL

    bins: int
          Number of bins to partitionate the range of possible RUL

    Returns
    -------
    fig: pyplot.Plot
    ax: pyplot.Axis
    """
    indices = np.where(val_rul <= treshhold)
    _, bin_edges = np.histogram(val_rul[indices], bins=bins)
    heights = []
    labels = []
    xs = []
    errs = []
    fig, ax = plt.subplots(1, 1, **kwargs)
    for i in range(len(bin_edges)-1):
        if i < len(bin_edges)-2:
            hist_indices = (val_rul >= bin_edges[i]) & (
                val_rul < bin_edges[i+1])
            labels.append(f'[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f})')
        else:
            hist_indices = (val_rul >= bin_edges[i]) & (
                val_rul <= bin_edges[i+1])
            labels.append(f'[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}]')
        error = (val_rul[hist_indices] - pred_cont[hist_indices])**2
        height = np.sqrt(np.mean(error))
        variance = np.std(error)

        heights.append(height)
        errs.append(variance)
        xs.append(i)
    ax.bar(height=heights, x=xs, tick_label=labels)
    ax.set_xlabel('RUL')
    ax.set_ylabel('RMSE')
    return fig, ax


def plot_true_vs_predicted(y_true, y_predicted, **kwargs):
    fig, ax = plt.subplots(1, 1, **kwargs)
    ax.plot(y_predicted, 'o', label='Predicted', markersize=0.7)
    ax.plot(y_true, label='True')
    ax.legend()
    return fig, ax
