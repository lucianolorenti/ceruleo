from copy import copy
from typing import Callable, List, Optional, Tuple, Union

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rul_pm.dataset.lives_dataset import AbstractLivesDataset


def add_vertical_line(ax, v_x, label, color, line, n_lines):
    miny, maxy = ax.get_ylim()
    ax.axvline(v_x, label=label, color=color)
    txt = ax.text(v_x,
                  miny + (maxy - miny) * (0.5 + 0.5 * (line / n_lines)),
                  label,
                  color=color,
                  fontweight='semibold')
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])


def lives_duration_histogram(datasets: Union[AbstractLivesDataset,
                                             List[AbstractLivesDataset]],
                             xlabel: str,
                             label: Union[str, List[str]] = '',
                             bins: int = 15,
                             units: str = 'm',
                             vlines: Tuple[float, str] = [],
                             ax=None,
                             add_mean: bool = True,
                             add_median: bool = True,
                             transform: Callable[[float], float] = lambda x: x,
                             threshold: float = np.inf,
                             color=None,
                             **kwargs):
    """Generate an histogram from the lives durations of the dataset

    Parameters
    ----------
    dataset : Union[AbstractLivesDataset, List[AbstractLivesDataset]]
        Dataset from which take the lives durations
    xlabel: str
        Label of the x axis
    label: Union[str, List[str]] = '',
        Label of each dataset to use as label in the boxplot
    bins : int, optional
        Number of bins to compute in the histogram, by default 15
    units : str, optional
        Units of time of the lives. Useful to generate labels, by default 'm'
    vlines : List[Tuple[float, str]], optional
        Vertical lines to be added to the plot
        Each element of the list should be the x position in the first element of the tuple,
        and the second elmenet of the tuple should be the label of the line
        By default []
    ax :  optional
        Axis where to draw the plot. 
        If missing a new figure will be created, by default None
    add_mean : bool, optional
        whether to add a vertical line with the mean value, by default True
    add_median : bool, optional
        whether to add a vertical line with the median value, by default True
    transform : Callable[[float], float], optional
        A function to transform each duration, by default lambdax:x
    threshold : float, optional
        Includes duration less than the threshold, by default np.inf

    Returns
    -------
    fig, ax 
        
    """
    if isinstance(datasets, list):
        assert len(datasets) == len(label)
    else:
        datasets = [datasets]
        label = [label]

    durations = []
    for ds in datasets:
        durations.append([transform(duration) for duration in ds.durations()])

    return lives_duration_histogram_from_durations(durations,
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
                                                   **kwargs)


def lives_duration_histogram_from_durations(
        durations: Union[List[float], List[List[float]]],
        xlabel: str,
        label: Union[str, List[str]] = '',
        bins: int = 15,
        units: str = 'm',
        vlines: List[Tuple[float, str]] = [],
        ax=None,
        add_mean: bool = True,
        add_median: bool = True,
        threshold: float = np.inf,
        color=None,
        alpha=1.0,
        **kwargs):
    """Generate an histogram from the lives durations

    Parameters
    ----------
    durations : Union[List[float], List[List[float]]]
        Duration of each live
    xlabel: str
        Label of the x axis
    label: Union[str, List[str]] = ''
        Label of each boxplot specified in durations
    bins : int, optional
        Number of bins to compute in the histogram, by default 15
    units : str, optional
        Units of time of the lives. Useful to generate labels, by default 'm'
    vlines : List[Tuple[float, str]], optional
        Vertical lines to be added to the plot
        Each element of the list should be the x position in the first element of the tuple,
        and the second elmenet of the tuple should be the label of the line
        By default []
    ax :  optional
        Axis where to draw the plot. 
        If missing a new figure will be created, by default None
    add_mean : bool, optional
        whether to add a vertical line with the mean value, by default True
    add_median : bool, optional
        whether to add a vertical line with the median value, by default True
    transform : Callable[[float], float], optional

    Returns
    -------
    [type]
        [description]
    """
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
            l += ' '
        vlines = copy(vlines)
        if add_mean:
            vlines.append((np.mean(dur), l + 'Mean'))
        if add_median:
            vlines.append((np.median(dur), l + 'Median'))
        dur = [d for d in dur if d < threshold]
        ax.hist(dur, bins, color=color, alpha=alpha, label=l)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Number of lives')

    colors = sns.color_palette("hls", len(vlines))
    for i, (v_x, l) in enumerate(vlines):
        label = f'{l}: {v_x:.2f} {units}'
        add_vertical_line(ax, v_x, label, colors[i], i, len(vlines))
    ax.legend()

    return ax.figure, ax


def durations_boxplot(datasets: Union[AbstractLivesDataset,
                                      List[AbstractLivesDataset]],
                      xlabel: Union[str, List[str]],
                      ylabel: str,
                      ax=None,
                      hlines: List[Tuple[float, str]] = [],
                      units: str = 'm',
                      transform: Callable[[float], float] = lambda x: x,
                      maxy: Optional[float] = None,
                      **kwargs):
    """Generate boxplots of the lives duration

    Parameters
    ----------
    datasets : Union[AbstractLivesDataset, List[AbstractLivesDataset]]
        [description]
    xlabel : Union[str, List[str]]
        [description]
    ylabel : str
        [description]
    ax : [type], optional
        [description], by default None
    hlines : List[Tuple[float, str]], optional
        [description], by default []
    units : str, optional
        [description], by default 'm'
    transform : Callable[[float], float], optional
        [description], by default lambdax:x
    maxy : Optional[float], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
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

    return durations_boxplot_from_durations(durations,
                                            xlabel=xlabel,
                                            ylabel=ylabel,
                                            ax=ax,
                                            hlines=hlines,
                                            units=units,
                                            maxy=maxy,
                                            **kwargs)


def durations_boxplot_from_durations(durations: Union[List[float],
                                                      List[List[float]]],
                                     xlabel: Union[str, List[str]],
                                     ylabel: str,
                                     ax=None,
                                     hlines: List[Tuple[float, str]] = [],
                                     units: str = 'm',
                                     maxy: Optional[float] = None,
                                     **kwargs):
    """Generate an histogram from a list of durations

    Parameters
    ----------
    durations : Union[List[float], List[List[float]]]
        [description]
    xlabel : Union[str, List[str]]
        [description]
    ylabel : str
        [description]
    ax : [type], optional
        [description], by default None
    hlines : List[Tuple[float, str]], optional
        [description], by default []
    units : str, optional
        [description], by default 'm'
    maxy : Optional[float], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """
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
        ax.axhline(pos, label=f'{label}: {pos:.2f} {units}', color=colors[i])
    _, labels = ax.get_legend_handles_labels()
    if len(labels) > 0:
        ax.legend()

    return ax.figure, ax
