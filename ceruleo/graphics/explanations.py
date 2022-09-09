from typing import List, Optional, Tuple, Type


import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
from matplotlib.colors import LogNorm, Normalize
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


def XCM_explanation(
    features_explanations: np.ndarray, times_explanations: np.ndarray, *, cmap="bwr"
) -> matplotlib.figure.Figure:
    """Plot the explanations of the XCM model

    Parameters

        features_explanations: Features explanations provided by from ceruleo.models.keras.catalog.XCM.explain
        times_explaantion: Times explanations provided by from ceruleo.models.keras.catalog.XCM.explain
        cmap: Colormap

    Return

        fig: matplotlib Figure
    """

    fig, ax = plt.subplots(1, 2, figsize=(17, 5))
    ax[0].set_title("Feature importance")
    im1 = ax[0].imshow(features_explanations.T, cmap=cmap)
    ax[0].set_ylabel("Features")
    ax[0].set_xlabel("Time")
    ax[0].grid(None)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax, orientation="vertical")

    ax[1].set_title("Temporal importance")
    im2 = ax[1].imshow(
        np.repeat(np.expand_dims(times_explanations, 1), 3, axis=1).T, cmap=cmap
    )
    ax[1].set_yticks([])
    ax[1].grid(None)
    ax[1].set_xlabel("Time")
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, cax=cax, orientation="vertical")
    return fig


def show_LASSOLayer(
    model, iterator_shape: Tuple[int, int], *, scale: bool = True, **fig_kwargs
):
    """Shows the LASSO layer

    Parameters

        model: tf.keras.Model with a LassoLayer at the beggining
        iterator_shape: Input shape of the model

    Returns

        fig: matplotlib.figure.Figure
    """
    im = np.abs(model.layers[1].w.numpy().reshape(iterator_shape).T)
    if scale:
        im = (im - im.min()) / (im.max() - im.min())

    fig, ax = plt.subplots(**fig_kwargs)
    c = ax.imshow(im, interpolation="nearest", aspect="auto")
    fig.colorbar(c)
    return fig
