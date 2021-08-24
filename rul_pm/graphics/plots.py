import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rul_pm.dataset.lives_dataset import AbstractLivesDataset
from rul_pm.graphics.utils.curly_brace import curlyBrace
from rul_pm.iterators.iterators import LifeDatasetIterator
from rul_pm.results.results import (
    FittedLife,
    compute_sample_weight,
    models_cv_results,
    split_lives,
    unexpected_breaks,
    unexploited_lifetime,
)
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse


def plot_lives(ds: AbstractLivesDataset):
    """
    Plot each life
    """
    fig, ax = plt.subplots()
    it = LifeDatasetIterator(ds)
    for _, y in it:
        ax.plot(y)
    return fig, ax


def cv_plot_errors_wrt_RUL(bin_edges, error_histogram, **kwargs):
    """"""
    fig, ax = plt.subplots(**kwargs)
    labels = []
    heights = []
    xs = []
    yerr = []

    for i in range(len(error_histogram)):
        xs.append(i)
        heights.append(np.mean(error_histogram[i]))
        yerr.append(np.std(error_histogram[i]))
        labels.append(f"[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f})")

    ax.bar(height=heights, x=xs, yerr=yerr, tick_label=labels)
    ax.set_xlabel("RUL")
    ax.set_ylabel("RMSE")

    return fig, ax


def _boxplot_errors_wrt_RUL_multiple_models(
    bin_edge: np.array,
    model_results: dict,
    ax=None,
    y_axis_label: Optional[str] = None,
    x_axis_label: Optional[str] = None,
    hold_out=False,
    **kwargs,
):
    def set_box_color(bp, color):
        plt.setp(bp["boxes"], color=color)
        plt.setp(bp["whiskers"], color=color)
        plt.setp(bp["caps"], color=color)
        plt.setp(bp["medians"], color=color)

    if ax is None:
        _, ax = plt.subplots(**kwargs)
    labels = []
    n_models = len(model_results)
    nbins = len(bin_edge) - 1

    for i in range(nbins):
        labels.append(f"[{bin_edge[i]:.1f}, {bin_edge[i+1]:.1f})")

    max_value = -np.inf
    min_value = np.inf
    colors = sns.color_palette("hls", n_models)
    for model_number, model_name in enumerate(model_results.keys()):
        model_data = model_results[model_name]
        if hold_out:
            for errors in model_data.errors:
                min_value = min(min_value, np.min(errors))
                max_value = max(max_value, np.max(errors))
        else:
            min_value = min(min_value, np.min(model_data.mean_error))
            max_value = max(max_value, np.max(model_data.mean_error))
        positions = []
        for i in range(nbins):
            positions.append((model_number * 0.5) + (i * n_models))
        if hold_out:
            box = ax.boxplot(
                np.array(model_data.errors, dtype=object),
                positions=positions,
                widths=0.2,
            )
        else:
            box = ax.boxplot(model_data.mean_error, positions=positions, widths=0.2)
        set_box_color(box, colors[model_number])
        ax.plot([], c=colors[model_number], label=model_name)

    ticks = []
    for i in range(nbins):
        x = np.mean(
            [(model_number * 0.5) + (i * n_models) for model_number in range(n_models)]
        )
        ticks.append(x)

    max_x = np.max(ticks) + 2
    ax.set_xlabel("RUL" + ("" if x_axis_label is None else x_axis_label))
    ax.set_ylabel("$y - \hat{y}$" + ("" if y_axis_label is None else y_axis_label))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.legend()
    ax2 = ax.twinx()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_ylim(ax.get_ylim())
    curlyBrace(
        ax.figure, ax2, (max_x, 0), (max_x, min_value), str_text="Over estim.", c="#000"
    )
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    curlyBrace(
        ax.figure,
        ax2,
        (max_x, max_value),
        (max_x, 0),
        str_text="Under estim.",
        c="#000",
    )

    return ax.figure, ax


def cv_boxplot_errors_wrt_RUL_multiple_models(
    results_dict: dict,
    nbins: int,
    y_axis_label: Optional[str] = None,
    x_axis_label: Optional[str] = None,
    fig=None,
    ax=None,
    **kwargs,
):
    """Boxplots of difference between true and predicted RUL over Cross-validated results


    Parameters
    ----------
    results_dict: dict
                  Dictionary with the results of the fitted models
    nbins: int
           Number of bins to divide the
    y_axis_label: Optional[str]. Default None,
                  Optional string to be added to the y axis
    x_axis_label: Optional[str]=None
                  Optional string to be added to the x axis
    fig:
       Optional figure in which the plot will be
    ax: Optional. Default None
        Optional axis in which the plot will be drawed.
        If an axis is not provided, it will create one.

    Keyword arguments
    -----------------
    **kwargs

    Return
    -------
    fig, ax:
    """
    if fig is None:
        fig, ax = plt.subplots(**kwargs)

    bin_edges, model_results = models_cv_results(results_dict, nbins)
    return _boxplot_errors_wrt_RUL_multiple_models(
        bin_edges,
        model_results,
        fig=fig,
        ax=ax,
        y_axis_label=y_axis_label,
        x_axis_label=x_axis_label,
    )


def hold_out_boxplot_errors_wrt_RUL_multiple_models(
    results_dict: dict,
    nbins: int,
    y_axis_label: Optional[str] = None,
    x_axis_label: Optional[str] = None,
    fig=None,
    ax=None,
    **kwargs,
):
    """Boxplots of difference between true and predicted RUL on a hold-out set


    Parameters
    ----------
    results_dict: dict
                  Dictionary with the results of the fitted models
    nbins: int
           Number of bins to divide the
    y_axis_label: Optional[str]. Default None,
                  Optional string to be added to the y axis
    x_axis_label: Optional[str]=None
                  Optional string to be added to the x axis
    fig:
       Optional figure in which the plot will be
    ax: Optional. Default None
        Optional axis in which the plot will be drawed.
        If an axis is not provided, it will create one.

    Keyword arguments
    -----------------
    **kwargs

    Return
    -------
    fig, ax:
    """
    if fig is None:
        fig, ax = plt.subplots(**kwargs)

    bin_edges, model_results = models_cv_results(results_dict, nbins)
    return _boxplot_errors_wrt_RUL_multiple_models(
        bin_edges,
        model_results,
        fig=fig,
        ax=ax,
        y_axis_label=y_axis_label,
        x_axis_label=x_axis_label,
        hold_out=True,
    )


def _cv_barplot_errors_wrt_RUL_multiple_models(
    bin_edges,
    model_results,
    fig=None,
    ax=None,
    y_axis_label=None,
    x_axis_label=None,
    **kwargs,
):
    """[summary]

    Args:
        bin_edges ([type]): [description]
        model_results ([type]): [description]
        fig ([type], optional): [description]. Defaults to None.
        ax ([type], optional): [description]. Defaults to None.
        y_axis_label ([type], optional): [description]. Defaults to None.
        x_axis_label ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if fig is None:
        fig, ax = plt.subplots(**kwargs)
    labels = []
    n_models = len(model_results)
    nbins = len(bin_edges) - 1

    width = 1.0 / 1.5

    for i in range(nbins):
        labels.append(f"[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f})")

    colors = sns.color_palette("hls", n_models)
    for model_number, model_name in enumerate(model_results.keys()):
        model_data = model_results[model_name]

        positions = []
        for i in range(nbins):
            positions.append((model_number * width) + (i * n_models))
        rect = ax.bar(
            positions,
            np.mean(model_data.mae, axis=0),
            yerr=np.std(model_data.mae, axis=0),
            label=model_name,
            width=width,
            color=colors[model_number],
        )

    ticks = []
    for i in range(nbins):
        x = np.mean(
            [(model_number * 0.5) + (i * n_models) for model_number in range(n_models)]
        )
        ticks.append(x)

    ax.set_xlabel("RUL" + ("" if x_axis_label is None else x_axis_label))
    ax.set_ylabel("$y - \hat{y}$" + ("" if y_axis_label is None else y_axis_label))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.legend()

    return fig, ax


def cv_barplot_errors_wrt_RUL_multiple_models(
    results_dict: dict,
    nbins: int,
    y_axis_label=None,
    x_axis_label=None,
    fig=None,
    ax=None,
    **kwargs,
):
    """Boxplots of difference between true and predicted RUL

    Parameters
    ----------
    nbins: int
           Number of boxplots
    """
    if fig is None:
        fig, ax = plt.subplots(**kwargs)

    bin_edges, model_results = models_cv_results(results_dict, nbins)
    return _cv_barplot_errors_wrt_RUL_multiple_models(
        bin_edges,
        model_results,
        fig=fig,
        ax=ax,
        y_axis_label=y_axis_label,
        x_axis_label=x_axis_label,
    )


def hold_out_barplot_errors_wrt_RUL_multiple_models(
    results_dict: dict,
    nbins: int,
    y_axis_label=None,
    x_axis_label=None,
    fig=None,
    ax=None,
    **kwargs,
):
    return cv_barplot_errors_wrt_RUL_multiple_models(
        results_dict,
        nbins,
        y_axis_label=y_axis_label,
        x_axis_label=x_axis_label,
        fig=fig,
        ax=ax,
        **kwargs,
    )


def _cv_shadedline_plot_errors_wrt_RUL_multiple_models(
    bin_edges,
    model_results,
    fig=None,
    ax=None,
    y_axis_label=None,
    x_axis_label=None,
    **kwargs,
):
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
    if fig is None:
        fig, ax = plt.subplots(**kwargs)
    labels = []
    n_models = len(model_results)
    nbins = len(bin_edges) - 1

    width = 1.0 / n_models

    for i in range(nbins):
        labels.append(f"[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f})")
    max_value = -np.inf
    min_value = np.inf
    colors = sns.color_palette("hls", n_models)
    for model_number, model_name in enumerate(model_results.keys()):
        model_data = model_results[model_name]
        min_value = min(min_value, np.min(model_data.mean_error))
        max_value = max(max_value, np.max(model_data.mean_error))
        positions = []
        for i in range(nbins):
            positions.append((model_number * width) + (i * n_models))

        mean_error = np.mean(model_data.mean_error, axis=0)
        std_error = np.std(model_data.mean_error, axis=0)
        rect = ax.plot(
            positions, mean_error, label=model_name, color=colors[model_number]
        )
        ax.fill_between(
            positions,
            mean_error - std_error,
            mean_error + std_error,
            alpha=0.3,
            color=colors[model_number],
        )

    ticks = []
    for i in range(nbins):
        x = np.mean(
            [(model_number * 0.5) + (i * n_models) for model_number in range(n_models)]
        )
        ticks.append(x)

    ax.set_xlabel("RUL" + ("" if x_axis_label is None else x_axis_label))
    ax.set_ylabel("$y - \hat{y}$" + ("" if y_axis_label is None else y_axis_label))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.legend()
    max_x = np.max(ticks) + 1
    ax2 = ax.twinx()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_ylim(ax.get_ylim())
    curlyBrace(
        fig, ax2, (max_x, 0), (max_x, min_value), str_text="Over estim.", c="#000"
    )
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    curlyBrace(
        fig, ax2, (max_x, max_value), (max_x, 0), str_text="Under estim.", c="#000"
    )

    return fig, ax


def cv_shadedline_plot_errors_wrt_RUL_multiple_models(
    results_dict: dict,
    nbins: int,
    y_axis_label=None,
    x_axis_label=None,
    fig=None,
    ax=None,
    **kwargs,
):
    """Boxplots of difference between true and predicted RUL
    The format of the input should be:

    .. highlight:: python
    .. code-block:: python

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

    bin_edges, model_results = models_cv_results(results_dict, nbins)
    return _cv_shadedline_plot_errors_wrt_RUL_multiple_models(
        bin_edges,
        model_results,
        fig=fig,
        ax=ax,
        bins=nbins,
        y_axis_label=y_axis_label,
        x_axis_label=x_axis_label,
    )


def plot_unexploited_lifetime(
    results_dict: dict,
    max_window: int,
    n: int,
    ax=None,
    units: Optional[str] = "",
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    n_models = len(results_dict)
    colors = sns.color_palette("hls", n_models)
    for i, model_name in enumerate(results_dict.keys()):
        m, ulft = unexploited_lifetime(results_dict[model_name], max_window, n)
        ax.plot(m, ulft, label=model_name, color=colors[i])
    ax.legend()
    ax.set_title("Unexploited lifetime")
    ax.set_xlabel("Fault window size" + units)
    ax.set_ylabel(units)
    return ax


def plot_unexpected_breaks(
    results_dict: dict,
    max_window: int,
    n: int,
    ax=None,
    units: Optional[str] = "",
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    n_models = len(results_dict)
    colors = sns.color_palette("hls", n_models)
    for i, model_name in enumerate(results_dict.keys()):
        m, ub = unexpected_breaks(results_dict[model_name], max_window, n)
        ax.plot(m, ub, label=model_name, color=colors[i])
    ax.set_title("Unexpected breaks")
    ax.set_xlabel("Fault window size" + units)
    ax.set_ylabel("Risk of breakage")
    ax.legend()
    return ax


def plot_true_and_predicted(
    results_dict: dict,
    ax=None,
    units: str = "Hours [h]",
    cv: int = 0,
    markersize: float = 0.7,
    **kwargs,
):
    """Plots the predicted and the true remaining useful lifes

    Parameters
    ----------
    results_dict : dict
        Dictionary with an interface conforming the requirements of the module
    ax : optional
        Axis to plot. If it is missing a new figure will be created, by default None
    units : str, optional
       Units of time to be used in the axis labels, by default 'Hours [h]'
    cv : int, optional
        Number of the CV results, by default 0

    Returns
    -------
    ax
        The axis on which the plot has been made
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, **kwargs)

    for model_name in results_dict.keys():
        r = results_dict[model_name]
        y_predicted = r[cv]["predicted"]
        y_true = r[cv]["true"]
        ax.plot(y_predicted, "o", label="Predicted", markersize=markersize)
        ax.plot(y_true, label="True")
        ax.set_ylabel(units)
        ax.set_xlabel(units)
        ax.legend()

    return ax


def plot_life(
    life: FittedLife,
    ax=None,
    units: Optional[str] = "",
    markersize: float = 0.7,
    add_fitted: bool = False,
    **kwargs,
):
    if ax is None:
        _, ax = plt.subplots(1, 1, **kwargs)

    time = life.time
    ax.plot(life.time, life.y_pred, "o", label="Predicted", markersize=markersize)
    ax.plot(life.time, life.y_true, label="True")
    if life.y_true[0] > 0:
        time1 = np.hstack((time[-1], time[-1] + life.y_true[-1]))
        ax.plot(time1, [life.y_true[-1], 0], label="Regressed true")
    if add_fitted:
        time1 = np.hstack((time[-1], time[-1] + life.y_pred[-1]))
        ax.plot(time1, [life.y_pred[-1], 0], label="Fitted")
        ax.plot(
            life.time,
            life.y_pred_fitted.predict_line(life.time),
            label="Picewise fitted",
        )

    ax.set_ylabel(units)
    ax.set_xlabel(units)
    _, max = ax.get_ylim()
    ax.set_ylim(0 - max / 10, max)
    ax.legend()

    return ax





def plot_test_set_predictions(results: dict, ncols: int = 3, CV: int = 0, alpha=1.0):

    init = False
    for model_name in results.keys():

        lives_model = split_lives(
            results[model_name][CV]["true"], results[model_name][CV]["predicted"]
        )
        NROW = math.ceil(len(lives_model) / ncols)
        if not init:
            fig, ax = plt.subplots(NROW, ncols, figsize=(25, 25))

        for i, life in enumerate(lives_model):
            row = int(i / ncols)
            col = i % ncols
            if not init:
                ax[row, col].plot(life.y_true, label="True")
            ax[row, col].plot(life.y_pred, label=model_name, alpha=alpha)
        init = True
    for a in ax.flatten():
        a.legend()
    return fig, ax 
