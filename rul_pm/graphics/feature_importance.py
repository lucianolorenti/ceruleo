import matplotlib.patches as patches
import matplotlib.pyplot as plt 
from typing import List, Optional
import matplotlib.cm as cm
import numpy as np 


def time_series_importance(
    n_features, window_size, coefficients, column_names:List[str], t: Optional[float] = None, ax=None,
    colormap:str = 'Greens'
):
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

    norm = plt.Normalize(np.min(im), np.max(im))

    for w in range(window_size):
        for f in range(n_features):
            importance = im[w, f]
            if t is not None:
                color, alpha = color_threshold(importance, 0)
            else:
                color, alpha = cmap(norm(importance)), np.clip(
                    norm(importance) + 0.2, 0, 1
                )

            ax.scatter(w, f, color=color, alpha=alpha, marker="s", s=75)
    ax.set_yticks(list(range(n_features)))
    ax.set_yticklabels(column_names)
    ax.set_xlim(-1, window_size + 0.5)
    ax.set_ylim(-1, n_features + 0.5)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))
    disp = (np.max(im) - np.min(im)) * 0.05
    cbar.set_ticks([np.min(im) + disp, np.max(im) - disp])
    cbar.ax.set_yticklabels(["Less important", "More Important"])
    ax.set_xlabel('Time window')
    return ax
