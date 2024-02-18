from copy import copy
from typing import Callable, List, Optional, Tuple, Union

import matplotlib
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ceruleo.dataset.ts_dataset import AbstractPDMDataset


def add_vertical_line(ax, v_x, label, color, line, n_lines):

    miny, maxy = ax.get_ylim()
    ax.axvline(v_x, label=label, color=color)
    txt = ax.text(
        v_x,
        miny + (maxy - miny) * (0.5 + 0.5 * (line / n_lines)),
        label,
        color=color,
        fontweight="semibold",
    )
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="w")])


def durations_histogram(
    datasets: Union[AbstractPDMDataset, List[AbstractPDMDataset]],
    xlabel: str = 'Cycle Duration',
    label: Union[str, List[str]] = "1",
    bins: int = 15,
    units: str = "m",
    vlines: Tuple[float, str] = [],
    ax:matplotlib.axes.Axes=None,
    add_mean: bool = True,
    add_median: bool = True,
    transform: Callable[[float], float] = lambda x: x,
    threshold: float = np.inf,
    color=None,
    **kwargs,
) ->  matplotlib.axes.Axes:
    """Generate an histogram from the lives durations of the dataset

    Example:
    '''
        durations_histogram(
            [train_dataset,validation_dataset],
            label=['Train','Validation'],
            xlabel='Unit Cycles',
            units='cycles',
            figsize=(17, 5));
    '''

    Parameters:
        datasets: Dataset from which take the lives durations
        xlabel: Label of the x axis, by default Cycle Duration
        label: Label of each dataset to use as label in the boxplot, by default 1
        bins:  Number of bins to compute in the histogram, by default 15
        units: Units of time of the lives. Useful to generate labels, by default m
        vlines: Vertical lines to add to the figure in the form [(x_coordinate, label)]
        ax: Axis where to draw the plot. If missing a new figure will be created
        add_mean: Whether to add a vertical line with the mean value, by default True
        add_median: whether to add a vertical line with the median value, by default True
        transform: A function to transform each duration, by default identity transform
        threshold: Includes duration less than the threshold, by default np.inf

    Returns:
        The axis in which the histogram was created

    """
    if isinstance(datasets, list):
        assert len(datasets) == len(label)
    else:
        datasets = [datasets]
        label = [label]

    durations = []
    for ds in datasets:
        durations.append([transform(duration) for duration in ds.durations()])

    return histogram_from_durations(
        durations,
        xlabel=xlabel,
        label=label,
        bins=bins,
        units=units,
        vlines=vlines,
        ax=ax,
        add_mean=add_mean,
        add_median=add_median,
        threshold=threshold,
        color=color,
        **kwargs,
    )


def histogram_from_durations(
    durations: Union[List[float], List[List[float]]],
    xlabel: str,
    label: Union[str, List[str]] = "",
    bins: int = 15,
    units: str = "m",
    vlines: List[Tuple[float, str]] = [],
    ax=None,
    add_mean: bool = True,
    add_median: bool = True,
    threshold: float = np.inf,
    color=None,
    alpha=1.0,
    **kwargs,
) ->  matplotlib.axes.Axes:
    
    if ax is None:
        _, ax = plt.subplots(1, 1, **kwargs)

    if isinstance(durations[0], list):
        assert isinstance(label, list)
        assert len(durations) == len(label)
    else:
        durations = [durations]
        label = [label]

    for l, dur in zip(label, durations):
        if len(l) > 0:
            l += " "
        vlines = copy(vlines)
        if add_mean:
            vlines.append((np.mean(dur), l + "Mean"))
        if add_median:
            vlines.append((np.median(dur), l + "Median"))
        dur = [d for d in dur if d < threshold]
        ax.hist(dur, bins, color=color, alpha=alpha, label=l)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of run-to-failure cycles")

    colors = sns.color_palette("hls", len(vlines))
    for i, (v_x, l) in enumerate(vlines):
        label = f"{l}: {v_x:.2f} {units}"
        add_vertical_line(ax, v_x, label, colors[i], i, len(vlines))
    ax.legend()

    return ax


def durations_boxplot(
    datasets: Union[AbstractPDMDataset, List[AbstractPDMDataset]],
    xlabel: Union[str, List[str]],
    ylabel: str = 'Cycle Duration',
    ax:Optional[matplotlib.axes.Axes]=None,
    hlines: List[Tuple[float, str]] = [],
    units: str = "m",
    transform: Callable[[float], float] = lambda x: x,
    maxy: Optional[float] = None,
    **kwargs,
) ->  matplotlib.axes.Axes:
    """Generate boxplots of the lives duration

    Example:

        ax = durations_boxplot(
            [train_dataset, validation_dataset],
            xlabel=['Train', 'Validation'],
            ylabel='Unit Cycles',
            figsize=(17, 5))

    Parameters:
        datasets: Dataset from which take the lives durations
        xlabel:  Label of each dataset to use as label in the boxplot
        ylabel: Label of the y axis
        ax: Axis where to draw the plot.If missing a new figure will be created
        hlines: Horizontal lines to add to the figure in the form [(y_coordinate, label)]
        units: Units of time of the lives. Useful to generate labels
        transform: A function to transform each duration
        maxy: Maximum y value of the plot

    Returns:
        Axis where plot has been drawn
    """
    if isinstance(datasets, list):
        assert isinstance(xlabel, list)
        assert len(datasets) == len(xlabel)
    else:
        datasets = [datasets]
        xlabel = [xlabel]

    durations = []
    for ds in datasets:
        durations.append([transform(duration) for duration in ds.durations()])

    return boxplot_from_durations(
        durations,
        xlabel=xlabel,
        ylabel=ylabel,
        ax=ax,
        hlines=hlines,
        units=units,
        maxy=maxy,
        **kwargs,
    )


def boxplot_from_durations(
    durations: Union[List[float], List[List[float]]],
    xlabel: Union[str, List[str]],
    ylabel: str,
    ax=None,
    hlines: List[Tuple[float, str]] = [],
    units: str = "m",
    maxy: Optional[float] = None,
    **kwargs,
)->  matplotlib.axes.Axes:
    
    if isinstance(durations[0], list):
        assert isinstance(xlabel, list)
        assert len(durations) == len(xlabel)
    else:
        durations = [durations]
        xlabel = [xlabel]

    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    ax.boxplot(durations, labels=xlabel)
    ax.set_ylabel(ylabel)
    if maxy is not None:
        miny, _ = ax.get_ylim()
        ax.set_ylim(miny, maxy)
    colors = sns.color_palette("hls", len(hlines))
    for i, (pos, label) in enumerate(hlines):
        ax.axhline(pos, label=f"{label}: {pos:.2f} {units}", color=colors[i])
    _, labels = ax.get_legend_handles_labels()
    if len(labels) > 0:
        ax.legend()

    return ax
