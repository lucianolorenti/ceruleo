from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rul_pm.dataset.lives_dataset import AbstractLivesDataset
from rul_pm.graphics.curly_brace import curlyBrace
from rul_pm.iterators.iterators import LifeDatasetIterator
from rul_pm.results.results import cv_error_histogram


def plot_lives(ds: AbstractLivesDataset):
    """
    Plot each life
    """
    fig, ax = plt.subplots()
    it = LifeDatasetIterator(ds)
    for _, y in it:
        ax.plot(y)
    return fig, ax


def plot_errors_wrt_RUL(val_rul,
                        pred_cont,
                        treshhold=np.inf,
                        bins=15,
                        **kwargs):
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
    for i in range(len(bin_edges) - 1):
        if i < len(bin_edges) - 2:
            hist_indices = (val_rul >= bin_edges[i]) & (val_rul <
                                                        bin_edges[i + 1])
            labels.append(f'[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f})')
        else:
            hist_indices = (val_rul >= bin_edges[i]) & (val_rul <=
                                                        bin_edges[i + 1])
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


def plot_true_vs_predicted(y_true,
                           y_predicted,
                           ylabel: Optional[str] = None,
                           **kwargs):
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

    return fig, ax


def compute_bars(error_histogram):

    heights = []
    xs = []
    yerr = []
    for i in range(len(error_histogram)):
        xs.append(i)
        heights.append(np.mean(error_histogram[i]))
        yerr.append(np.std(error_histogram[i]))
    return heights, xs, yerr


def _cv_boxplot_errors_wrt_RUL_multiple_models(bin_edge,
                                               error_histograms,
                                               model_names,
                                               fig=None,
                                               ax=None,
                                               y_axis_label=None,
                                               x_axis_label=None,
                                               **kwargs):
    """Plot a error bar for each model

    Args:
        bin_edge ([type]): [description]
        error_histograms ([type]): [description]
        model_names ([type]): [description]
        width (float, optional): [description]. Defaults to 0.5.
        ax ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    if fig is None:
        fig, ax = plt.subplots(**kwargs)
    labels = []
    n_models = len(model_names)
    #bars = [compute_bars(e) for e in error_histograms]

    #deltax = (width) / len(bars)
    for i in range(len(error_histograms[0])):
        labels.append(f'[{bin_edge[i]:.1f}, {bin_edge[i+1]:.1f})')

    #for i, (heights, xs, yerr) in enumerate(bars):
    #    xx = np.array(xs) + (deltax*(i+1)) - width
    #    ax.bar(height=heights, width=(width / len(bars))-0.01, x=xx, yerr=yerr,
    #           label=model_names[i])

    max_value = -np.inf
    min_value = np.inf
    colors = sns.color_palette("hls", n_models)
    for model_number, (model_name, model_data) in enumerate(
            zip(model_names, error_histograms)):
        positions = []
        data_list = []
        for i, bins in enumerate(model_data):
            data = np.concatenate((*bins, ))
            max_value = max(np.max(data), max_value)
            min_value = min(np.min(data), min_value)
            positions.append((model_number * 0.5) + (i * n_models))
            data_list.append(data)
        box = ax.boxplot(np.array(data_list, dtype=object),
                         positions=positions,
                         widths=0.2)
        set_box_color(box, colors[model_number])
        ax.plot([], c=colors[model_number], label=model_name)

    ticks = []
    for i, _ in enumerate(error_histograms[0]):
        x = np.mean([(model_number * 0.5) + (i * n_models)
                     for model_number in range(n_models)])
        ticks.append(x)

    max_x = np.max(ticks) + 1
    ax.set_xlabel('RUL' + ('' if x_axis_label is None else x_axis_label))
    ax.set_ylabel('$y - \hat{y}$' +
                  ('' if y_axis_label is None else y_axis_label))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.legend()
    ax2 = ax.twinx()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_ylim(ax.get_ylim())
    curlyBrace(fig,
               ax2, (max_x, 0), (max_x, min_value),
               str_text='Over estim.',
               c="#000")
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    curlyBrace(fig,
               ax2, (max_x, max_value), (max_x, 0),
               str_text='Under estim.',
               c="#000")

    return fig, ax


def preprocess_results(results_dict: dict, nbins:int):
    """Boxplots of difference between true and predicted RUL
    The format of the input should be:
    {
        'Model Name': [
            {
                'true': np.array,
                'predicted': np.array
            },
            {
                'true': np.array,
                'predicted': np.array
            },
            ...
        'Model Name 2': [
             {
                'true': np.array,
                'predicted': np.array
            },
            {
                'true': np.array,
                'predicted': np.array
            },
            ...
        ]
    }

    Parameters
    ----------
    nbins: int
           Number of boxplots
    """
    
    max_y_value = np.max([
        r['true'].max() for model_name in results_dict.keys()
        for r in results_dict[model_name]
    ])
    bin_edges = np.linspace(0, max_y_value, nbins + 1)

    model_names = []
    error_histogram_list = []
    for model_name in results_dict.keys():
        model_names.append(model_name)
        trues = []
        predicted = []
        for results in results_dict[model_name]:
            trues.append(results['true'])
            predicted.append(results['predicted'])
        _, error_histogram = cv_error_histogram(trues,
                                                predicted,
                                                nbins=nbins,
                                                bin_edges=bin_edges)
        error_histogram_list.append(error_histogram)
    return model_names, bin_edges, error_histogram_list


def cv_boxplot_errors_wrt_RUL_multiple_models(results_dict: dict,
                                              nbins: int,
                                              y_axis_label=None,
                                              x_axis_label=None,
                                              fig=None,
                                              ax=None,
                                              **kwargs):
    """Boxplots of difference between true and predicted RUL
    The format of the input should be:
    {
        'Model Name': [
            {
                'true': np.array,
                'predicted': np.array
            },
            {
                'true': np.array,
                'predicted': np.array
            },
            ...
        'Model Name 2': [
             {
                'true': np.array,
                'predicted': np.array
            },
            {
                'true': np.array,
                'predicted': np.array
            },
            ...
        ]
    }

    Parameters
    ----------
    nbins: int
           Number of boxplots
    """
    if fig is None:
        fig, ax = plt.subplots(**kwargs)
    

    model_names, bin_edges, error_histogram_list = preprocess_results(results_dict, nbins)
    return _cv_boxplot_errors_wrt_RUL_multiple_models(
        bin_edges,
        error_histogram_list,
        model_names,
        fig=fig,
        ax=ax,
        bins=nbins,
        y_axis_label=y_axis_label,
        x_axis_label=x_axis_label)




def _cv_barplot_errors_wrt_RUL_multiple_models(bin_edge,
                                               error_histograms,
                                               model_names,
                                               fig=None,
                                               ax=None,
                                               y_axis_label=None,
                                               x_axis_label=None,
                                               **kwargs):
    """Plot a error bar for each model

    Args:
        bin_edge ([type]): [description]
        error_histograms ([type]): [description]
        model_names ([type]): [description]
        width (float, optional): [description]. Defaults to 0.5.
        ax ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    if fig is None:
        fig, ax = plt.subplots(**kwargs)
    labels = []
    n_models = len(model_names)
    #bars = [compute_bars(e) for e in error_histograms]

    #deltax = (width) / len(bars)
    for i in range(len(error_histograms[0])):
        labels.append(f'[{bin_edge[i]:.1f}, {bin_edge[i+1]:.1f})')

    #for i, (heights, xs, yerr) in enumerate(bars):
    #    xx = np.array(xs) + (deltax*(i+1)) - width
    #    ax.bar(height=heights, width=(width / len(bars))-0.01, x=xx, yerr=yerr,
    #           label=model_names[i])

    max_value = -np.inf
    min_value = np.inf
    colors = sns.color_palette("hls", n_models)
    for model_number, (model_name, model_data) in enumerate(
            zip(model_names, error_histograms)):
        positions = []
        data_list = []
        for i, bins in enumerate(model_data):
            data = np.concatenate((*bins, ))
            max_value = max(np.max(data), max_value)
            min_value = min(np.min(data), min_value)
            positions.append((model_number * 0.5) + (i * n_models))
            data_list.append(data)
        box = ax.boxplot(np.array(data_list, dtype=object),
                         positions=positions,
                         widths=0.2)
        set_box_color(box, colors[model_number])
        ax.plot([], c=colors[model_number], label=model_name)

    ticks = []
    for i, _ in enumerate(error_histograms[0]):
        x = np.mean([(model_number * 0.5) + (i * n_models)
                     for model_number in range(n_models)])
        ticks.append(x)

    max_x = np.max(ticks) + 1
    ax.set_xlabel('RUL' + ('' if x_axis_label is None else x_axis_label))
    ax.set_ylabel('$y - \hat{y}$' +
                  ('' if y_axis_label is None else y_axis_label))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.legend()
    ax2 = ax.twinx()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_ylim(ax.get_ylim())
    curlyBrace(fig,
               ax2, (max_x, 0), (max_x, min_value),
               str_text='Over estim.',
               c="#000")
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    curlyBrace(fig,
               ax2, (max_x, max_value), (max_x, 0),
               str_text='Under estim.',
               c="#000")

    return fig, ax


def cv_barplot_errors_wrt_RUL_multiple_models(results_dict: dict,
                                              nbins: int,
                                              y_axis_label=None,
                                              x_axis_label=None,
                                              fig=None,
                                              ax=None,
                                              **kwargs):
    """Boxplots of difference between true and predicted RUL
    The format of the input should be:
    {
        'Model Name': [
            {
                'true': np.array,
                'predicted': np.array
            },
            {
                'true': np.array,
                'predicted': np.array
            },
            ...
        'Model Name 2': [
             {
                'true': np.array,
                'predicted': np.array
            },
            {
                'true': np.array,
                'predicted': np.array
            },
            ...
        ]
    }

    Parameters
    ----------
    nbins: int
           Number of boxplots
    """
    if fig is None:
        fig, ax = plt.subplots(**kwargs)
    

    model_names, bin_edges, error_histogram_list = preprocess_results(results_dict, nbins)
    return _cv_barplot_errors_wrt_RUL_multiple_models(
        bin_edges,
        error_histogram_list,
        model_names,
        fig=fig,
        ax=ax,
        bins=nbins,
        y_axis_label=y_axis_label,
        x_axis_label=x_axis_label)


