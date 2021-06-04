import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rul_pm.graphics.plots import plot_true_and_predicted
from rul_pm.models.keras.keras import KerasTrainableModel
from tensorflow.keras.callbacks import Callback
from rul_pm.iterators.utils import true_values


logger = logging.getLogger(__name__)


class PredictionCallback(Callback):
    """Generate a plot after each epoch with the predictions

    Parameters
    ----------
    model : KerasTrainableModel
        The model used predict
    output_path : Path
        Path of the output image
    dataset : [type]
        The dataset that want to be plotted
    """

    def __init__(self, model: KerasTrainableModel, output_path: Path, batcher, units:str):

        super().__init__()
        self.output_path = output_path
        self.batcher = batcher
        self.pm_model = model
        self.units = units

    def on_epoch_end(self, epoch, logs={}):

        y_pred = self.pm_model.predict(self.batcher)
        y_true = true_values(self.batcher)
        ax = plot_true_and_predicted(
            {"Model": [{"true": y_true, "predicted": y_pred}]},
            figsize=(17, 5), 
            units=self.units
        )
        ax.legend()
        ax.figure.savefig(self.output_path, dpi=ax.figure.dpi)

        plt.close(ax.figure)


class TerminateOnNaN(Callback):
    """Callback that terminates training when a NaN loss is encountered."""

    def __init__(self, batcher):
        super().__init__()
        self.batcher = batcher

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                logger.info("Batch %d: Invalid loss, terminating training" % (batch))
                self.model.stop_training = True
                self.batcher.stop = True
