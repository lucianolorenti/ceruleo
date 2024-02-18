import functools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ceruleo.graphics.duration import durations_boxplot, durations_histogram
import pytest
from ceruleo.dataset.ts_dataset import AbstractPDMDataset
from ceruleo.graphics.analysis import plot_correlation_analysis
from ceruleo.graphics.explanations import XCM_explanation
from ceruleo.graphics.results import (
    barplot_errors_wrt_RUL,
    boxplot_errors_wrt_RUL,
    plot_predictions,
    plot_predictions_grid,
    shadedline_plot_errors_wrt_RUL,
)
from ceruleo.results.results import PredictionResult

np.random.seed(19680801)

PATH = Path(__file__).resolve().parent
TEST_IMAGES_PATH = PATH / "test_images"


def helper_test_plot(output_filename: str):
    def inner_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(args, kwargs)
            (TEST_IMAGES_PATH).mkdir(parents=True, exist_ok=True)
            figure = func(*args, **kwargs)
            EXPECTED_IMAGE_PATH = TEST_IMAGES_PATH / (
                output_filename + "-expected" + ".png"
            )
            if not (EXPECTED_IMAGE_PATH).is_file():
                figure.savefig(EXPECTED_IMAGE_PATH)
                assert False
            CURRENT_IMAGE_PATH = TEST_IMAGES_PATH / (output_filename + ".png")
            figure.savefig(CURRENT_IMAGE_PATH)
            assert (
                np.mean(
                    np.abs(
                        plt.imread(EXPECTED_IMAGE_PATH) - plt.imread(CURRENT_IMAGE_PATH)
                    )
                )
                < 0.1
            )

        return wrapper

    return inner_decorator


def create_predictions(name: str, number_of_lives: int) -> PredictionResult:
    y_trues = []
    y_preds = []
    for i in range(number_of_lives):
        N = np.random.randint(1500) + 200
        y_true = np.linspace(500, 0, N)
        s = np.random.rand() * 500
        y_pred = y_true + np.random.randn(N) * s
        y_trues.append(y_true)
        y_preds.append(y_pred)

    y_true = np.hstack((*y_trues,))
    y_pred = np.hstack((*y_preds,))
    return PredictionResult(name, y_true, y_pred)


class MockDataset(AbstractPDMDataset):
    def __init__(self, nlives: int):
        super().__init__()
        self.lives = [
            pd.DataFrame(
                {
                    "feature1": np.linspace(0, 100, 50),
                    "feature2": np.random.randint(2, size=(50,)),
                    "RUL": np.linspace(100, 0, 50),
                }
            )
            for i in range(nlives)
        ]

    def get_time_series(self, i: int):
        return self.lives[i]

    @property
    def rul_column(self):
        return "RUL"

    @property
    def n_time_series(self):
        return len(self.lives)


@pytest.fixture(scope="class")
def predictions():
    return {
        "Model A": [create_predictions("Model A", 5), create_predictions("Model A", 5)],
        "Model B": [create_predictions("Model B", 5), create_predictions("Model B", 8)],
    }


class TestGraphics:
    @helper_test_plot(output_filename="test_plot_predictions_grid_1")
    def test_plot_predictions_grid_1(self):
        y_true = np.linspace(500, 0, num=500)
        y_pred = y_true + np.random.rand(500) * 15
        r = PredictionResult("Example", y_true, y_pred)
        ax = plot_predictions_grid(r, ncols=1)
        return ax[0][0].figure

    @helper_test_plot(output_filename="test_plot_predictions_grid_2")
    def test_plot_predictions_grid_2(self):
        y_true = np.linspace(500, 0, num=500)
        y_pred = y_true + np.random.rand(500) * 15
        y_true = np.hstack((y_true, y_true))
        y_pred = np.hstack((y_pred, y_pred))
        r = PredictionResult("Example", y_true, y_pred)
        ax = plot_predictions_grid(r, ncols=2)
        return ax[0][0].figure

    @helper_test_plot(output_filename="test_plot_predictions_grid_2")
    def test_plot_predictions_grid_3(self):
        y_true = np.linspace(500, 0, num=500)
        y_pred = y_true + np.random.rand(500) * 15
        y_true = np.hstack((y_true, y_true))
        y_pred = np.hstack((y_pred, y_pred))
        r = PredictionResult("Example", y_true, y_pred)
        ax = plot_predictions_grid(r, ncols=1)
        return ax[0][0].figure

    @helper_test_plot(output_filename="test_boxplot_errors_wrt_RUL")
    def test_boxplot_errors_wrt_RUL(self, predictions):
        ax = boxplot_errors_wrt_RUL(predictions, nbins=5)
        return ax.figure

    @helper_test_plot(
        output_filename="test_barplot_errors_wrt_RUL",
    )
    def test_barplot_errors_wrt_RUL(self, predictions):
        ax = barplot_errors_wrt_RUL(predictions, nbins=5)
        return ax.figure

    @helper_test_plot(
        output_filename="test_shadedline_plot_errors_wrt_RUL",
    )
    def test_shadedline_plot_errors_wrt_RUL(self, predictions):
        ax = shadedline_plot_errors_wrt_RUL(predictions, nbins=5)
        return ax.figure

    @helper_test_plot(
        output_filename="test_XCM_explanation",
    )
    def test_XCM_explanation(self):
        mmap = np.random.rand(40, 40)
        mtime = np.random.rand(40)
        fig = XCM_explanation(mmap, mtime)
        return fig

    @helper_test_plot(
        output_filename="test_correlation_plot",
    )
    def test_correlation_plot(self):
        dataset = MockDataset(7)
        ax = plot_correlation_analysis(dataset)
        return ax.figure

    @helper_test_plot(
        output_filename="test_durations_boxplot",
    )
    def test_durations_boxplot(self):
        dataset = MockDataset(7)
        ax = durations_boxplot(dataset, xlabel="Test Dataset")
        return ax.figure

    @helper_test_plot(
        output_filename="test_durations_histogram",
    )
    def test_durations_histogram(self):
        dataset = MockDataset(7)
        ax = durations_histogram(dataset)
        return ax.figure
