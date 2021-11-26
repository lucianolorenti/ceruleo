"""Visualization utilities for RUL estimation models


It is possible to visualize how it grows the unexploited lifetime grows as the conservative window size grows and how the unexpected breaks decrease as the conservative window size grows.


"""
import math
from typing import Dict, Iterable, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
from ceruleo.dataset.transformed import TransformedDataset
from ceruleo.graphics.utils.curly_brace import curlyBrace
from ceruleo.results.results import (FittedLife, PredictionResult,
                                     models_cv_results, split_lives,
                                     unexpected_breaks, unexploited_lifetime)


def plot_lives(ds: TransformedDataset):
    """
    Plot each life

    Parameters:

        ds: A transformed dataset
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


    Parameters:
    
        results_dict: Dictionary with the results of the fitted models
        nbins: Number of bins to divide the
        y_axis_label: Optional string to be added to the y axis
        x_axis_label: Optional string to be added to the x axis
        ax: Optional axis in which the plot will be drawed.
            If an axis is not provided, it will create one.

    Keyword arguments:

        **kwargs

    Return:
        ax
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
    ax=None,
    y_axis_label: Optional[str] = None,
    x_axis_label: Optional[str] = None,
    color_palette: str = "hls",
    bar_width: float=1/1.5,
    **kwargs,
):
    """Plot the barplots given the errors

    Parameters:

        bin_edges: np.ndarray:

        model_results: Dictionary with the results
        ax: Axis
        y_axis_label: Y Label
        x_axis_label: X Label

    Returns:

        Tuple[fig, axis]
    """
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    fig  = ax.figure
    labels = []
    n_models = len(model_results)
    nbins = len(bin_edges) - 1

   

    for i in range(nbins):
        labels.append(f"[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f})")

    colors = sns.color_palette(color_palette, n_models)
    
    mean_data = []
    std_data = []
    model_names = []
    for model_number, model_name in enumerate(model_results.keys()):
        model_data = model_results[model_name]
        mean_data.append(np.mean(model_data.mae, axis=0))
        std_data.append(np.std(model_data.mae, axis=0))
        model_names.append(model_name)

    model_names = np.array(model_names)

    bar_group_width = n_models*(bar_width+1)
    group_separation = int(bar_group_width/2)
    mean_data = np.vstack(mean_data)
    std_data = np.vstack(std_data)
    n_models, n_bins = mean_data.shape
    indices = np.argsort(mean_data[:,  0])
    for i in range(n_bins):
        mean_data[:, i] = mean_data[indices, i]
        std_data[:, i] = std_data[indices, i]
    
    for model_name, model_index in zip(model_names[indices], range(n_models)):    
        
        positions =  model_index+np.array(range(n_bins))  * (bar_group_width + group_separation)

        rect = ax.bar(
            positions,
            mean_data[model_index, :],
            yerr=std_data[model_index, :],
            label=model_name,
            width=bar_width,
            color=colors[model_index],
        )

    ticks = []
    dx = 0
    for i in range(nbins):
        ticks.append( dx + bar_group_width/2)
        dx += bar_group_width + group_separation

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
    ax=None,
    color_palette: str = "hls",
    **kwargs,
):
    """Barlots of difference between true and predicted RUL

    Parameters:

        results_dict: Dictionary with the results for each model
        nbins: Number of bins in wich divide the RUL target
        y_axis_label: Y label
        x_axis_label: X label
        ax: Axis
        color_palette: 

    """
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    bin_edges, model_results = models_cv_results(results_dict, nbins)
    return _cv_barplot_errors_wrt_RUL_multiple_models(
        bin_edges,
        model_results,
        ax=ax,
        y_axis_label=y_axis_label,
        x_axis_label=x_axis_label,
        color_palette=color_palette,
    )


def _cv_shadedline_plot_errors_wrt_RUL_multiple_models(
    bin_edges,
    model_results,
    ax=None,
    y_axis_label=None,
    x_axis_label=None,
    **kwargs,
):
    """Plot a shaded regions for each model

    """
    if ax is None:
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
        ax2.figure, ax2, (max_x, 0), (max_x, min_value), str_text="Over estim.", c="#000"
    )
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    curlyBrace(
        ax2.figure, ax2, (max_x, max_value), (max_x, 0), str_text="Under estim.", c="#000"
    )

    return ax


def shadedline_plot_errors_wrt_RUL(
    results_dict: dict,
    nbins: int,
    y_axis_label=None,
    x_axis_label=None,
    ax=None,
    **kwargs,
):
    """Shaded line

    Parameters:
        results_dict: _description_
        nbins:_description_
        y_axis_label: _description_, by default None
        x_axis_label:_description_, by default None
        ax: _description_, by default None

    Returns:
        ax: The axis
    """

    if ax is None:
        fig, ax = plt.subplots(**kwargs)

    bin_edges, model_results = models_cv_results(results_dict, nbins)
    return _cv_shadedline_plot_errors_wrt_RUL_multiple_models(
        bin_edges,
        model_results,
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
    add_shade: bool = True,
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    n_models = len(results_dict)
    colors = sns.color_palette("hls", n_models)
    for i, model_name in enumerate(results_dict.keys()):
        m, ulft, std_ul = unexploited_lifetime(results_dict[model_name], max_window, n)
        ax.plot(m, ulft, label=model_name, color=colors[i])
        if add_shade:
            ax.fill_between(m, ulft+std_ul, ulft-std_ul, alpha=0.1, color=colors[i])
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
    add_shade: bool = True,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Plot the risk of unexpected breaks with respect to the maintenance window

    Parameters:
        results_dict: Dictionary with the results
        max_window: Maximum size of the maintenance windows
        n: Number of points used to evaluate the window size
        ax: axis on which to draw, by default None
        units: Units to use in the xlabel, by default ""

    Returns:

        ax: The axis in which the plot was made
    """
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    n_models = len(results_dict)
    colors = sns.color_palette("hls", n_models)
    for i, model_name in enumerate(results_dict.keys()):
        m, mean_ub, std_ub = unexpected_breaks(results_dict[model_name], max_window, n)
        ax.plot(m, mean_ub, label=model_name, color=colors[i])
        if add_shade:
            ax.fill_between(m, mean_ub+std_ub, mean_ub-std_ub, alpha=0.1, color=colors[i])

    ax.set_title("Unexpected breaks")
    ax.set_xlabel("Fault window size" + units)
    ax.set_ylabel("Risk of breakage")
    ax.legend()
    return ax




def plot_J_Cost(
    results: Dict[str, List[PredictionResult]],
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
        window_ub, mean_ub, std_ub = unexpected_breaks(results[model_name], window_size=window, step=step)
        window_ul, mean_ul, std_ul = unexploited_lifetime(results[model_name], window_size=window, step=step)

        v = []

        labels = []
        from uncertainties import unumpy
   
        for r in ratio:
            UB_c = 1.0
            UL_c = UB_c * r
            
            v.append(np.mean(unumpy.uarray(mean_ub, std_ub) * UB_c + unumpy.uarray(mean_ul, std_ul) * UL_c))

            labels.append(f"{int(UB_c)}:{UL_c}")

      
  
        mean = unumpy.nominal_values(v)
        std = unumpy.std_devs(v)
        ax.plot(ratio, mean, "-o", label=model_name, color=colors[i])
        ax.fill_between(ratio, mean+std, mean-std, color=colors[i], alpha=0.2)

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
    plot_target:bool = True,
    add_regressed:bool = True,
    start_x:int= 0,
    label:str = '',
    **kwargs,
):
    """Plot a single life

    Parameters:
        life: A fitted life
        ax: The axis where to plot
        units : Optional[str], optional
            _description_, by default ""
        markersize: Size of the marker
        add_fitted: Wether to add the LS fitted line to the points
        plot_target: Wether to plot the true RUL values
        add_regressed:
        start_x: Initial point of the time-indepedent feature to plot
        label: 

    Returns:
    
        ax: Axis
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, **kwargs)

    time = life.time

    ax.plot(
        life.time[start_x: len(life.y_pred)],
        life.y_pred[start_x:],
        "o",
        label=f"Predicted {label}",
        markersize=markersize,
    )
    if plot_target:
        ax.plot(life.time, life.y_true, label="True", linewidth=3)
    if add_regressed and life.y_true[-1] > 0:
            time1 = np.hstack((time[-1], time[-1] + life.y_true[-1]))
            ax.plot(time1, [life.y_true[-1], 0], label="Regressed true")
    if add_fitted:
        #time1 = np.hstack(
        #    (time[len(life.y_pred) - 1], time[len(life.y_pred) - 1] + life.y_pred[-1])
        #)
        #ax.plot(time1, [life.y_pred[-1], 0], label="Projected end")
        ax.plot(
           life.time,
           life.y_pred_fitted,
           label="Picewise fitted",
        )
        ax.plot(
           life.time,
           life.y_true_fitted,
           label="Picewise fitted",
        )

    ax.set_ylabel(units)
    ax.set_xlabel(units)
    _, max = ax.get_ylim()
    ax.set_ylim(0 - max / 10, max)
    legend = ax.legend(markerscale=15,)



    return ax


def plot_predictions_grid(
    results: Union[PredictionResult, List[PredictionResult]],
    ncols: int = 3,
    alpha=1.0,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    **kwargs,
):
    """Plot a matrix of predictions

    Parameters:
    
        results: Dictionary with the results
        ncols: Number of colmns in the plot, by default 3
        alpha: Opacity of the predicted curves, by default 1.0
        xlabel: Xlabel, by default None
        ylabel: YLabel, by default None

    Return:

        ax: The axis on which the plot has been made
    """

    def linear_to_subindices(i, ncols):
        row = int(i / ncols)
        col = i % ncols
        return row, col

    if isinstance(results, PredictionResult):
        results = [results]

    init = False

    for model_results in results:
        lives_model = split_lives(model_results)
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
    return ax


def plot_predictions(
    result: PredictionResult,
    ax:Optional[matplotlib.axes.Axes]=None,
    units: str = "Hours [h]",
    markersize: float = 0.7,
    plot_fitted: bool  = True,
    model_name:str = '',
    **kwargs,
) -> matplotlib.axes.Axes:
    """Plots the predicted and the true remaining useful lives

    Parameters:
    
        result: A PredictionResult object
        ax:  Axis to plot. If it is missing a new figure will be created
        units: Units of time to be used in the axis labels
        cv: Number of the CV results

    Returns:

        ax: The axis on which the plot has been made
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, **kwargs)



    y_predicted = result.predicted_RUL
    y_true = result.true_RUL
    ax.plot(y_predicted, "o", label=f"Predicted {model_name}", markersize=markersize)
    ax.plot(y_true, label="True")
    x = 0


    if plot_fitted:
        try:
            fitted = np.hstack([life.y_pred_fitted for life in split_lives(result)])
            ax.plot(fitted, label='Fitted')
            
        except:
            
            pass
    ax.set_ylabel(units)
    ax.set_xlabel(units)
    legend = ax.legend()
    for l in legend.legendHandles:
        l.set_markersize(6)


    return ax
