import matplotlib.pyplot as plt
import numpy as np


def plot_kurtogram(Kwav, freq, Level_w, ax=None):
    """
    Plots the kurtogram.

    Parameters
    -----------
    Kwav: np.array
        kurtogram
    freq_w: np.array
        frequency vector
    Level_w: np.array
        vector of levels
    ax: Optional[Axis]

    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    aspect = (1/(Kwav.shape[0] / Kwav.shape[1]))

    im = ax.imshow(np.clip(Kwav, 0, np.inf), aspect=aspect, cmap="viridis")
    x_ticks = [x for x in ax.get_xticks() if x >= 0 and x < len(freq)]
    ax.set_xticks(x_ticks)

    ax.set_xticklabels([np.round(freq[int(i)] / 1000, 2) for i in x_ticks])
    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel("Level")
    ax.set_yticks(list(range(len(Level_w))))
    ax.set_yticklabels([np.round(j, 2) for j in Level_w])
    ax.grid(False)
    ax.figure.colorbar(im)
    return ax
