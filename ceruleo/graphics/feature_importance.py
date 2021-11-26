from typing import List, Optional, Type

import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize


def time_series_importance(
    n_features:int,
    window_size:int,
    coefficients:np.ndarray,
    column_names: List[str],
    t: Optional[float] = None,
    ax=None,
    colormap: str = "Greens",
    features_to_plot: Optional[int] = None,
    base_alpha:float =0.2,
    normalizer_cls: Type[Normalize] = Normalize
):  
    """Plot the feature importance by time-stamp and feature

    Parameters
    ----------
    n_features : int
        Total number of features used in the model
    window_size : int
        Window size used in the model
    coefficients : np.ndarray
        Coefficient array with shape (1 x n_features*window_size)
    column_names : List[str]
        Name of the columns
    t : Optional[float], optional
        [description], by default None
    ax : [type], optional
        Axis where to put the graphic, by default None
    colormap : str, optional
        Color map to use, by default "Greens"
    features_to_plot : Optional[int], optional
        Maximum number of features to plot, by default None
        If it is omitted all the features will be used.
        The features_to_plot most important features are going to be plotted.
        The importance will be computed as the sum of the timestamp  importance
        per feature
    normalizer_cls : Type[Normalize], by  default Normalize
        Color mapper class
    """
    def color_threshold(importance, t):
        if importance > t:
            color = "green"
            alpha = 1.0
        else:
            color = "black"
            alpha = 0.2
        return color, alpha

    if ax is None:
        fig, ax = plt.subplots(figsize=(17, 5))
    else:
        fig = ax.get_figure()
    im = coefficients.reshape(window_size, n_features)
    cmap = plt.get_cmap(colormap)
    
    norm = normalizer_cls(np.min(im), np.max(im), clip=True)

    importance = np.sum(im, axis=0)

    features_order = np.argsort(importance)[::-1]
    if features_to_plot is None:
        features_to_plot = len(features_order)
    features_order = features_order[:features_to_plot][::-1]
    n_selected_features = len(features_order)

    for w in range(window_size):
        
        for y, f in enumerate(features_order):

            importance = im[w, f]
            if t is not None:
                color, alpha = color_threshold(importance, 0)
            else:
                color, alpha = cmap(norm(importance)), np.clip(
                    norm(importance) + base_alpha, 0, 1
                )

            ax.scatter(w, y, color=color, alpha=alpha, marker="s", s=75)
    ax.set_yticks(list(range(n_selected_features)))
    ax.set_yticklabels(column_names[features_order])
    ax.set_xlim(-1, window_size + 0.5)
    ax.set_ylim(-1, n_selected_features + 0.5)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))
    disp = (np.max(im) - np.min(im)) * 0.05
    cbar.set_ticks([np.min(im), np.max(im) - disp])
    cbar.ax.set_yticklabels(["Less important", "More Important"])
    ax.set_xlabel("Time window")
    return ax
