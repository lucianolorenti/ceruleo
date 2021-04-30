from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from rul_pm.dataset.lives_dataset import AbstractLivesDataset
from rul_pm.models.model import TrainableModel
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold
from tqdm.auto import tqdm


def compute_rul_line(rul: float, n: int, tt: Optional[np.array] = None):
    if tt is None:
        tt = -np.ones(n)
    z = np.zeros(n)
    z[0] = rul
    for i in range(len(tt) - 1):
        z[i + 1] = max(z[i] + tt[i], 0)
        if z[i + 1] - 0 < 0.0000000000001:
            break
    return z


def preventive_ruls(train_dataset, test_dataset, window, transform=lambda x: x):
    ruls = [transform(life[test_dataset.rul_column].iloc[0]) for life in train_dataset]
    mean_rul = np.mean(ruls)
    median_rul = np.mean(ruls)
    y_mean_rul = []
    y_median_rul_z = []

    for life in test_dataset:
        tt = transform(np.diff(life[test_dataset.rul_column][window:]))
        y_mean_rul.extend(compute_rul_line(mean_rul, life.shape[0] - window + 1, tt))
        y_median_rul_z.extend(
            compute_rul_line(median_rul, life.shape[0] - window + 1, tt)
        )
    return y_mean_rul, y_median_rul_z


def summary(y_true, y_pred, train_dataset, test_dataset, window, transform=lambda x: x):
    y_mean_rul, y_median_rul_z = preventive_ruls(
        train_dataset, test_dataset, window, transform
    )

    mean_mse = mse(y_true, y_mean_rul)
    median_mse = mse(y_true, y_median_rul_z)
    model_mse = mse(y_true, y_pred)

    return pd.DataFrame(
        {"Model": [model_mse], "Mean RUL": [mean_mse], "Median RUL": [median_mse]}
    )


def cv_predictions(
    model: TrainableModel,
    dataset: AbstractLivesDataset,
    cv=KFold(n_splits=5),
    fit_params={},
    progress_bar=False,
) -> Tuple[List[List], List[List]]:
    """
    Train a model using cross validation and return the predictions of each fold

    Parameters
    ----------
    model: TrainableModel
           The model to train

    dataset: AbstractLivesDataset
             The dataset from which obtain the folds

    cv:  default  sklearn.model_selection.KFold(n_splits=5)
        The dataset splitter

    Return
    ------
    Tuple[List, List]:

    Then length of the lists is equal to the nomber of folds. The first element
    of the tuple contains the true values of the hold-out set and the second
    element contains the predictions values of the hold-out set
    """
    progress_bar_fun = iter
    if progress_bar:
        progress_bar_fun = tqdm

    predictions = []
    true_values = []
    for train_index, test_index in progress_bar_fun(cv.split(dataset)):
        model.fit(dataset[train_index], **fit_params)
        y_pred = model.predict(dataset[test_index])
        y_true = model.true_values(dataset[test_index])
        predictions.append(y_pred)
        true_values.append(y_true)

    return (true_values, predictions)


def cv_error_histogram(
    y_true: List[List], y_pred: List[List], nbins: int = 5,
    bin_edges: Optional[np.array] = None
) -> Tuple[np.array, np.array]:
    """
    Compute the error histogram

    Compute the error with respect to the RUL considering the results of different
    folds

    Parameters
    ----------
    y_true: List[List]
            List with the true values of each hold-out set of a cross validation
    y_pred: List[List]
            List with the predictions of each hold-out set of a cross validation
    nbins: int
           Number of bins to compute the histogram

    Return
    ------
    Tuple[np.array, np.array]
    * bin_edges: Array of len(y_true)+1 containing the limits of the histogram
    * error_histogram: An array of len(y_true) containing the mse of the points
                       corresponding to the (bin_edges[i], bin_endges[i+1]) for each
                       fold
    """
    if bin_edges is None:
        max_value = np.max([np.max(y) for y in y_true])
        bin_edges = np.linspace(0, max_value, nbins + 1)
    error_histogram = [[] for _ in range(len(bin_edges) - 1)]
    for y_pred, y_true in zip(y_pred, y_true):
        for j in range(len(bin_edges) - 1):
            indices = np.where((y_true >= bin_edges[j]) & (y_true <= bin_edges[j + 1]))[
                0
            ]
            if len(indices) > 0:
                error_histogram[j].append(
                    y_true[indices] - y_pred[indices]
                )
    return bin_edges, error_histogram


def regression_metrics(y_true: List[List], y_pred: List[List]):
    mses = []
    maes = []
    for y_true_elem, y_pred_elem in zip(y_true, y_pred):
        mses.append(mse(y_true_elem, y_pred_elem))
        maes.append(mae(y_true_elem, y_pred_elem))

    data = np.round([np.mean(mses), np.std(mses), np.mean(maes), np.std(maes)], 2)
    return pd.Series(
        [f"{data[0]} \pm {data[1]}", f"{data[2]} \pm {data[3]}"]
    )
