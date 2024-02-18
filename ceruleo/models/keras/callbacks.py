import logging
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from ceruleo.graphics.results import plot_predictions
from ceruleo.iterators.utils import true_values
from ceruleo.results.results import PredictionResult
from tensorflow.keras.callbacks import Callback

logger = logging.getLogger(__name__)


class PredictionCallback(Callback):
    """
    Generate a plot after each epoch with the predictions

    Parameters:
        model: The model used predict
        output_path: Path of the output image
        dataset: The dataset that want to be plotted
    """

    def __init__(
        self,
        output_path: Path,
        dataset: tf.data.Dataset,
        units: str = "",
        filename_suffix: str = "",
    ):
        super().__init__()
        self.output_path = output_path
        self.dataset = dataset
        self.units = units
        self.suffix = filename_suffix
        if len(filename_suffix) > 0:
            self.output_path = self.output_path.with_stem(
                filename_suffix + "_" + self.output_path.stem
            )

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.dataset)
        y_true = true_values(self.dataset)
        ax = plot_predictions(
            PredictionResult("Model", y_true, y_pred),
            figsize=(17, 5),
            units=self.units,
        )
        ax.legend()

        ax.figure.savefig(self.output_path, dpi=ax.figure.dpi)

        plt.close(ax.figure)
