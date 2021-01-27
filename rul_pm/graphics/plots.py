from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
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


def plot_errors_wrt_RUL(val_rul, pred_cont, treshhold=np.inf, bins=15, **kwargs):
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


def plot_true_vs_predicted(y_true, y_predicted, ylabel: Optional[str] = None, **kwargs):
    fig, ax = plt.subplots(1, 1, **kwargs)
    ax.plot(y_predicted, 'o', label='Predicted', markersize=0.7)
    ax.plot(y_true, label='True')
    ax.set_ylabel('Time [h]' if ylabel is None else ylabel)
    ax.legend()
    return fig, ax


def cv_plot_errors_wrt_RUL(bin_edges, error_histogram, **kwargs):
    """
    """
    fig, ax = plt.subplots(**kwargs)
    labels = []
    heights = []
    xs = []
    yerr = []

    for i in range(len(error_histogram)):
        xs.append(i)
        heights.append(np.mean(error_histogram[i]))
        yerr.append(np.std(error_histogram[i]))
        labels.append(f'[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f})')

    ax.bar(height=heights, x=xs, yerr=yerr, tick_label=labels)
    ax.set_xlabel('RUL')
    ax.set_ylabel('RMSE')

    return fig


def compute_bars(error_histogram):

    heights = []
    xs = []
    yerr = []
    for i in range(len(error_histogram)):
        xs.append(i)
        heights.append(np.mean(error_histogram[i]))
        yerr.append(np.std(error_histogram[i]))
    return heights, xs, yerr


def cv_plot_errors_wrt_RUL_multiple_models(bin_edges, error_histograms, model_names, width=0.5, **kwargs):
    """
    """
    fig, ax = plt.subplots(**kwargs)
    labels = []
    bars = [compute_bars(e) for e in error_histograms]

    deltax = (width) / len(bars)
    for i in range(len(error_histograms[0])):
        labels.append(f'[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f})')

    for i, (heights, xs, yerr) in enumerate(bars):
        xx = np.array(xs) + (deltax*(i+1)) - width

        ax.bar(height=heights, width=(width / len(bars))-0.01, x=xx, yerr=yerr,
               label=model_names[i])
    ax.set_xlabel('RUL')
    ax.set_ylabel('RMSE')
    ax.set_xticklabels(labels)
    ax.set_xticks(list(range(len(labels))))
    ax.legend()

    return fig
