import math
from typing import Dict, Iterable, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from temporis.dataset.transformed import TransformedDataset
from rul_pm.graphics.utils.curly_brace import curlyBrace
from rul_pm.results.results import (
    PredictionResult,
    FittedLife,
    models_cv_results,
    split_lives,
    unexpected_breaks,
    unexploited_lifetime,
)


def plot_lives(ds: TransformedDataset):
    """
    Plot each life
    """
    fig, ax = plt.subplots()
    it = ds
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
        hold_out = model_data.n_folds == 1
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

    max_x = np.max(ticks) + 1
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


def boxplot_errors_wrt_RUL(
    results_dict: Dict[str, List[PredictionResult]],
    nbins: int,
    y_axis_label: Optional[str] = None,
    x_axis_label: Optional[str] = None,
    ax=None,
    **kwargs,
):
    """Boxplots of difference between true and predicted RUL over Cross-validated results


    Parameters
    ----------
    results_dict: Dict[str, List[PredictionResult]]
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
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    else:
        fig = ax.figure

    bin_edges, model_results = models_cv_results(results_dict, nbins)
    return _boxplot_errors_wrt_RUL_multiple_models(
        bin_edges,
        model_results,
        fig=fig,
        ax=ax,
        y_axis_label=y_axis_label,
        x_axis_label=x_axis_label,
    )


def _cv_barplot_errors_wrt_RUL_multiple_models(
    bin_edges,
    model_results: dict,
    fig=None,
    ax=None,
    y_axis_label: Optional[str] = None,
    x_axis_label: Optional[str] = None,
    color_palette: str = "hls",
    **kwargs,
):
    """Plot the barplots given the errors

    Parameters
    ----------
        bin_edges: np.ndarray:

        model_results: dict
            Dictionary with the results
        fig: Optional[plt.Figure]
            Figure
        ax: Optional[ax.Axis] Defaults to None.
            Axis
        y_axis_label: Optional[str] Defaults to None.
            Y Label
        x_axis_label:Optional[str]
            X Label

    Returns:
        Tuple[fig, axis]
    """
    if fig is None:
        fig, ax = plt.subplots(**kwargs)
    labels = []
    n_models = len(model_results)
    nbins = len(bin_edges) - 1

    width = 1.0 / 1.5

    for i in range(nbins):
        labels.append(f"[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f})")

    colors = sns.color_palette(color_palette, n_models)
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


def barplot_errors_wrt_RUL(
    results_dict: Dict[str, List[PredictionResult]],
    nbins: int,
    y_axis_label=None,
    x_axis_label=None,
    fig=None,
    ax=None,
    color_palette: str = "hls",
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
        color_palette=color_palette,
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
        hold_out = model_data.n_folds == 1
        if hold_out:
            for errors in model_data.errors:
                min_value = min(min_value, np.min(errors))
                max_value = max(max_value, np.max(errors))
        else:
            min_value = min(min_value, np.min(model_data.mean_error))
            max_value = max(max_value, np.max(model_data.mean_error))

        positions = []
        for i in range(nbins):
            positions.append((model_number * width) + (i * n_models))

        if not hold_out:
            mean_error = np.mean(model_data.mean_error, axis=0)
            std_error = np.std(model_data.mean_error, axis=0)
        else:
            mean_error = np.array([np.mean(e, axis=0) for e in model_data.errors])
            std_error = np.array([np.std(e, axis=0) for e in model_data.errors])

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


def shadedline_plot_errors_wrt_RUL(
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
    ax: Optional[matplotlib.axes.Axes] = None,
    units: Optional[str] = "",
    **kwargs,
) -> matplotlib.axes.Axes:
    """Plot the risk of unexpected breaks with respect to the maintenance window

    Parameters
    ----------
    results_dict : dict
        Dictionary with the results
    max_window : int
        Maximum size of the maintenance windows
    n : int
        Number of points used to evaluate the window size
    ax : Optional[matplotlib.axes.Axes], optional
        axis on which to draw, by default None
    units : Optional[str], optional
        Units to use in the xlabel, by default ""

    Returns
    -------
    matplotlib.axes.Axes
        The axis in which the plot was made
    """
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


def plot_J_Cost(
    results: dict,
    window: int,
    step: int,
    ax=None,
    ratio_min: float = 1 / 120,
    ration_max: float = 1 / 5,
    ratio_n_points: int = 50,
    label: str = "",
):
    a, b = unexpected_breaks(results, window_size=window, step=step)
    c, d = unexploited_lifetime(results, window_size=window, step=step)

    ratio = np.linspace(ratio_min, ration_max, ratio_n_points)
    if ax is None:
        fig, ax = plt.subplots(figsize=(17, 5))
    v = []
    q = []
    labels = []
    for r in ratio:
        UB_c = 1.0
        UL_c = UB_c * r
        v.append(np.min(np.array(b) * UB_c + np.array(d) * UL_c))
        labels.append(f"{int(UB_c)}:{UL_c}")

    def f(x):
        UB_c = 1 / x
        UL_c = 1
        if x == 0:
            return ""
        return f"{int(UB_c)}:{int(UL_c)}"

    ax.plot(ratio, v, label=label)
    ax.set_xticks([1 / 120, 1 / 30, 1 / 20, 1 / 15, 1 / 10, 1 / 7, 1 / 5])
    ax.set_xticklabels([f(x) for x in ax.get_xticks()])
    ax.set_xlabel(
        "Ratio between UL and UB. How many minutes of UL are equal to 1 breakage"
    )
    ax.set_ylabel("J")
    return ax


def plot_true_and_predicted(
    results_dict: dict,
    ax=None,
    units: str = "Hours [h]",
    cv: int = 0,
    markersize: float = 0.7,
    **kwargs,
):
    """Plots the predicted and the true remaining useful lives

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


def plot_J_Cost(
    results: dict,
    window: int,
    step: int,
    ax=None,
    ratio_min: float = 1 / 120,
    ratio_max: float = 1 / 5,
    ratio_n_points: int = 50,
):
    def label_formatter(x):
        UB_c = 1 / x
        UL_c = 1
        if x == 0:
            return ""
        return f"{int(UB_c)}:{int(UL_c)}"

    if ax is None:
        fig, ax = plt.subplots(figsize=(17, 5))

    ratio = np.linspace(ratio_min, ratio_max, ratio_n_points)
    n_models = len(results)
    colors = sns.color_palette("hls", n_models)
    for i, model_name in enumerate(results.keys()):
        a, b = unexpected_breaks(results[model_name], window_size=window, step=step)
        c, d = unexploited_lifetime(results[model_name], window_size=window, step=step)

        v = []
        labels = []
        for r in ratio:
            UB_c = 1.0
            UL_c = UB_c * r
            v.append(np.min(np.array(b) * UB_c + np.array(d) * UL_c))
            labels.append(f"{int(UB_c)}:{UL_c}")

        ax.plot(ratio, v, "-o", label=model_name, color=colors[i])

    ticks = ax.get_xticks().tolist()
    ticks.append(ratio[0])
    ax.set_xticks(ticks)
    ax.set_xticklabels([label_formatter(x) for x in ax.get_xticks()])
    ax.set_xlabel(
        "Ratio between UL and UB. How many minutes of UL are equal to 1 breakage"
    )
    ax.set_ylabel("J")
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

    ax.plot(
        life.time[: len(life.y_pred)],
        life.y_pred,
        "o",
        label="Predicted",
        markersize=markersize,
    )
    ax.plot(life.time, life.y_true, label="True")
    if life.y_true[-1] > 0:
        time1 = np.hstack((time[-1], time[-1] + life.y_true[-1]))
        ax.plot(time1, [life.y_true[-1], 0], label="Regressed true")
    if add_fitted:
        time1 = np.hstack(
            (time[len(life.y_pred) - 1], time[len(life.y_pred) - 1] + life.y_pred[-1])
        )
        ax.plot(time1, [life.y_pred[-1], 0], label="Projected end")
        # ax.plot(
        #    life.time,
        #    life.y_pred_fitted.predict_line(life.time),
        #    label="Picewise fitted",
        # )

    ax.set_ylabel(units)
    ax.set_xlabel(units)
    _, max = ax.get_ylim()
    ax.set_ylim(0 - max / 10, max)
    ax.legend()

    return ax


def plot_predictions(
    results: Union[PredictionResult, List[PredictionResult]],
    ncols: int = 3,
    alpha=1.0,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    **kwargs,
):
    """Plot a matrix of predictions

    Parameters
    ----------
    results : dict
        Dictionary with the results
    ncols : int, optional
        Number of colmns in the plot, by default 3
    alpha : float, optional
        Opacity of the predicted curves, by default 1.0
    xlabel : Optional[str], optional
        Xlabel, by default None
    ylabel : Optional[str], optional
        YLabel, by default None

    Return
    ------
    fig, ax:
        Figure and axis
    """

    def linear_to_subindices(i, ncols):
        row = int(i / ncols)
        col = i % ncols
        return row, col

    if isinstance(results, PredictionResult):
        results = [results]

    init = False

    for model_results in results:
        lives_model = split_lives(model_results.true_RUL, model_results.predicted_RUL)
        NROW = math.ceil(len(lives_model) / ncols)
        if not init:
            fig, ax = plt.subplots(NROW, ncols, squeeze=False, **kwargs)

        for i, life in enumerate(lives_model):
            row, col = linear_to_subindices(i, ncols)

            if not init:
                ax[row, col].plot(life.time, life.y_true, label="True")

            ax[row, col].plot(
                life.time, life.y_pred, label=model_results.name, alpha=alpha
            )
            if xlabel is not None:
                ax[row, col].set_xlabel(xlabel)
            if ylabel is not None:
                ax[row, col].set_ylabel(ylabel)
        init = True
    for j in range(len(lives_model), NROW * ncols):
        row, col = linear_to_subindices(j, ncols)
        fig.delaxes(ax[row, col])

    for a in ax.flatten():
        a.legend()
    return fig, ax
