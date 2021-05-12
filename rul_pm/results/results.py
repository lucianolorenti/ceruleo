"""Compute evaluating results of fitted models

The main structure used on this functions is a dictionary in which
each the keys are the model name, and the elements are list of dictionaries.
Each of the dictionaries contain two keys: true, predicted. 
Those elements are list of the predictions

Example:
```
    {
        'Model Name': [
            {
                'true': [true_0, true_1,...., true_n],
                'predicted': [pred_0, pred_1,...., pred_n]
            },
            {
                'true': [true_0, true_1,...., true_m],
                'predicted': [pred_0, pred_1,...., pred_m]
            },
            ...
        'Model Name 2': [
             {
                'true': [true_0, true_1,...., true_n],
                'predicted': [pred_0, pred_1,...., pred_n]
            },
            {
                'true': [true_0, true_1,...., true_m],
                'predicted': [pred_0, pred_1,...., pred_m]
            },
            ...
        ]
    }
```
"""
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from rul_pm.dataset.lives_dataset import AbstractLivesDataset
from rul_pm.models.model import TrainableModel
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


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


def preventive_ruls(train_dataset,
                    test_dataset,
                    window,
                    transform=lambda x: x):
    ruls = [
        transform(life[test_dataset.rul_column].iloc[0])
        for life in train_dataset
    ]
    mean_rul = np.mean(ruls)
    median_rul = np.mean(ruls)
    y_mean_rul = []
    y_median_rul_z = []

    for life in test_dataset:
        tt = transform(np.diff(life[test_dataset.rul_column][window:]))
        y_mean_rul.extend(
            compute_rul_line(mean_rul, life.shape[0] - window + 1, tt))
        y_median_rul_z.extend(
            compute_rul_line(median_rul, life.shape[0] - window + 1, tt))
    return y_mean_rul, y_median_rul_z


def summary(y_true,
            y_pred,
            train_dataset,
            test_dataset,
            window,
            transform=lambda x: x):
    y_mean_rul, y_median_rul_z = preventive_ruls(train_dataset, test_dataset,
                                                 window, transform)

    mean_mse = mse(y_true, y_mean_rul)
    median_mse = mse(y_true, y_median_rul_z)
    model_mse = mse(y_true, y_pred)

    return pd.DataFrame({
        "Model": [model_mse],
        "Mean RUL": [mean_mse],
        "Median RUL": [median_mse]
    })


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


class CVResults:
    def __init__(self,
                 y_true: List[List],
                 y_pred: List[List],
                 nbins: int = 5,
                 bin_edges: Optional[np.array] = None):
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

        """
        if bin_edges is None:
            max_value = np.max([np.max(y) for y in y_true])
            bin_edges = np.linspace(0, max_value, nbins + 1)
        self.n_folds = len(y_true)
        self.n_bins = len(bin_edges) - 1
        self.bin_edges = bin_edges
        self.mean_error = np.zeros((self.n_folds, self.n_bins))
        self.mae = np.zeros((self.n_folds, self.n_bins))
        self.mse = np.zeros((self.n_folds, self.n_bins))
        for i, (y_pred, y_true) in enumerate(zip(y_pred, y_true)):
            self._add_fold_result(i, y_pred, y_true)

    def _add_fold_result(self, fold: int, y_pred: np.array, y_true: np.array):
        for j in range(len(self.bin_edges) - 1):
            mask = ((y_true >= self.bin_edges[j]) &
                    (y_true <= self.bin_edges[j + 1]))
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue
            errors = y_true[indices] - y_pred[indices]
            self.mean_error[fold, j] = np.mean(errors)
            self.mae[fold, j] = np.mean(np.abs(errors))
            self.mse[fold, j] = np.mean((errors)**2)


def models_cv_results(results_dict: dict, nbins: int):
    """Create a dictionary with the result of each cross validation of the model
    The format of the input should be:
    {
        'Model Name': [
            {
                'true': np.array,
                'predicted': np.array
            },
            {
                'true': np.array,
                'predicted': np.array
            },
            ...
        'Model Name 2': [
             {
                'true': np.array,
                'predicted': np.array
            },
            {
                'true': np.array,
                'predicted': np.array
            },
            ...
        ]
    }

    Parameters
    ----------
    dict: string-> CVResults
           Number of boxplots
    """

    max_y_value = np.max([
        r['true'].max() for model_name in results_dict.keys()
        for r in results_dict[model_name]
    ])
    bin_edges = np.linspace(0, max_y_value, nbins + 1)

    model_results = {}

    for model_name in results_dict.keys():
        trues = []
        predicted = []
        for results in results_dict[model_name]:
            trues.append(results['true'])
            predicted.append(results['predicted'])
        model_results[model_name] = CVResults(trues,
                                              predicted,
                                              nbins=nbins,
                                              bin_edges=bin_edges)

    return bin_edges, model_results


class FittedLife:
    """[summary]

    Parameters
    ----------

    y_true (np.array): [description]
    y_pred (np.array): [description]
    """
    def __init__(self, y_true: np.array, y_pred: np.array):
        self.y_true = np.squeeze(y_true)
        self.y_pred = np.squeeze(y_pred)
        self.time = np.hstack(([0], np.cumsum(np.diff(self.y_true[::-1]))))
        self.fitted = self._fitrls()

    def _fitrls(self):
        def pred(x, p):
            return p[0] + p[1] * x

        N = len(self.y_pred)
        D = np.zeros((2, N))
        D[1, :] = self.time
        D[0, :] = 1
        mod = sm.RecursiveLS(self.y_pred, D.T)
        res = mod.fit()
        self.params = res.params
        return np.array([pred(x, self.params) for x in self.time])

    def predicted_end_of_life(self):
        return -self.params[0] / self.params[1]

    def end_of_life(self):
        return self.y_true[0]

    def maintenance_point(self, m: float = 0):
        """[summary]

        Args:
            m (float, optional): [description]. Defaults to 0.

        Returns:
            [type]: [description]
        """
        return self.predicted_end_of_life() - m

    def unexploited_lifetime(self, m: float = 0):
        """[summary]

        Machine Learning for Predictive Maintenance: A Multiple Classifiers Approach
        Susto, G. A., Schirru, A., Pampuri, S., McLoone, S., & Beghi, A. (2015). 

        Parameters
        ----------
            m (float, optional): [description]. Defaults to 0.

        Returns:
            float: unexploited lifetime
        """
        if self.maintenance_point(m) < self.end_of_life():
            return self.end_of_life() - self.maintenance_point(m)
        else:
            return 0

    def unexpected_break(self, m: float = 0):
        """[summary]

        Machine Learning for Predictive Maintenance: A Multiple Classifiers Approach
        Susto, G. A., Schirru, A., Pampuri, S., McLoone, S., & Beghi, A. (2015). 

        Parameters
        ----------
            m (float, optional): [description]. Defaults to 0.

        Returns:
            float: unexpected breaks
        """
        if self.maintenance_point(m) < self.end_of_life():
            return 0
        else:
            return 1


def split_lives(y_true: np.array, y_pred: np.array) -> List[FittedLife]:
    lives_indices = ([0] +
                     (np.where(np.diff(np.squeeze(y_true)) > 0)[0]).tolist() +
                     [len(y_true)])
    lives = []
    for i in range(len(lives_indices) - 1):
        r = range(lives_indices[i] + 1, lives_indices[i + 1])
        try:
            lives.append(FittedLife(y_true[r], y_pred[r]))
        except Exception as e:
            logger.error(e)
    return lives


def split_lives_from_results(d: dict) -> List[FittedLife]:
    y_true = d['true']
    y_pred = d['predicted']
    return split_lives(y_true, y_pred)


def unexploited_lifetime(d: dict, window_size: int, step: int):
    bb = [split_lives_from_results(cv) for cv in d]
    qq = []
    windows = np.array(range(0, window_size, step))
    for m in windows:
        jj = []
        for r in bb:
            ul_cv_list = [life.unexploited_lifetime(m) for life in r]
            mean_ul_cv = np.mean(ul_cv_list)
            std_ul_cv = np.std(ul_cv_list)
            jj.append(mean_ul_cv)
        qq.append(np.mean(jj))
    return windows, qq


def unexpected_breaks(d, window_size: int, step: int):
    bb = [split_lives_from_results(cv) for cv in d]
    qq = []
    windows = np.array(range(0, window_size, step))
    for m in windows:
        jj = []
        for r in bb:
            ul_cv_list = [life.unexpected_break(m) for life in r]
            mean_ul_cv = np.mean(ul_cv_list)
            std_ul_cv = np.std(ul_cv_list)
            jj.append(mean_ul_cv)
        qq.append(np.mean(jj))
    return windows, qq


def regression_metrics(data: dict, threshold: float = np.inf) -> dict:
    """Compute regression metrics for each model

    Parameters
    ----------
    data : dict
        Dictionary with the model predictions.
        The dictionary must conform the results specification of this module
    threshold : float, optional
        Compute metrics errors only in RUL values less than the threshold, by default np.inf

    Returns
    -------
    dict
        A dictionary with the following format
        ```
        {
            'Model 1': {
                'mean':
                'std':
            },
            ....
            'Model nane n': {
                'mean': ,
                'std': ,
            }
        }
        ```
    """
    metrics_dict = {}
    for m in data.keys():
        errors = []
        for r in data[m]:
            y = np.where(r['true'] <= threshold)[0]
            y_true = r['true'][y]
            y_pred = r['predicted'][y]
            errors.append(mae(y_true, y_pred))
        metrics_dict[m] = {'mean': np.mean(errors), 'std': np.std(errors)}
    return metrics_dict
