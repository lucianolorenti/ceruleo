from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
from ceruleo.dataset.ts_dataset import AbstractTimeSeriesDataset


def correlation_analysis(
    dataset: AbstractTimeSeriesDataset,
    corr_threshold: float = 0,
    features: Optional[List[str]] = None,
    ax: matplotlib.axes.Axes = Optional[None],
    **kwargs,
):
    """Plot the correlated features in a dataset

    Parameters:
    
        dataset: The dataset
        corr_threshold: Minimum threshold to consider that the correlation is high
        features: List of features
        ax: The axis where to draw

    Returns:
        ax: the axis
    """

    df = correlation_analysis(
        dataset, corr_threshold, features=list(set(features) - set(["relative_time"]))
    )
    df1 = df[(df["Abs mean correlation"] > corr_threshold)]

    df1.reset_index(inplace=True)
    df1.sort_values(by="Mean Correlation", ascending=True, inplace=True)
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    labels = []
    for i, (_, r) in enumerate(df1.iterrows()):
        f1 = r["Feature 1"]
        f2 = r["Feature 2"]
        label = f"{f1}\n{f2}"
        ax.barh(
            y=i,
            width=r["Mean Correlation"],
            label=label,
            xerr=r["Std Correlation"],
            color="#7878FF",
        )
        labels.append(label)

    ax.axvline(x=0.90, linestyle="--")
    ax.axvline(x=-0.90, linestyle="--")

    ax.set_yticks(list(range(len(labels))))
    ax.set_yticklabels(labels)
    xticks = ax.get_xticks()

    ax.set_xticks([-1, -0.90, -0.5, 0, 0.5, 0.90, 1])
    ax.set_xlabel("Correlation")
    return ax
