"""Compute evaluating results of fitted models

The main structure used on this functions is a dictionary in which
each the keys are the model name, and the elements are list of dictionaries.
Each of the dictionaries contain two keys: true, predicted. 
Those elements are list of the predictions

Example:

.. highlight:: python
.. code-block:: python

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



"""
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rul_pm.results.picewise_regression import (
    PiecewesieLinearFunction,
    PiecewiseLinearRegression,
)
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse

logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    mae: float
    mse: float
    fitting_time: float = 0
    prediction_time: float = 0


@dataclass
class PredictionResult:
    name: str
    true_RUL: np.ndarray
    predicted_RUL: np.ndarray
    metrics: MetricsResult = MetricsResult(0, 0)

    def compute_metrics(self):
        self.metrics.mae = mae(self.true_RUL, self.predicted_RUL)
        self.metrics.mse = mse(self.true_RUL, self.predicted_RUL)


def compute_sample_weight(sample_weight, y_true, y_pred, c: float = 0.9):
    if sample_weight == "relative":
        sample_weight = np.abs(y_true - y_pred) / (np.clip(y_true, c, np.inf))
    else:
        sample_weight = 1
    return sample_weight


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


class CVResults:
    def __init__(
        self,
        y_true: List[List],
        y_pred: List[List],
        nbins: int = 5,
        bin_edges: Optional[np.array] = None,
    ):
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
        self.errors = []
        for i, (y_pred, y_true) in enumerate(zip(y_pred, y_true)):
            self._add_fold_result(i, y_pred, y_true)

    def _add_fold_result(self, fold: int, y_pred: np.array, y_true: np.array):
        y_pred = np.squeeze(y_pred)
        y_true = np.squeeze(y_true)

        for j in range(len(self.bin_edges) - 1):

            mask = (y_true >= self.bin_edges[j]) & (y_true <= self.bin_edges[j + 1])
            indices = np.where(mask)[0]

            if len(indices) == 0:
                continue

            errors = y_true[indices] - y_pred[indices]

            self.mean_error[fold, j] = np.mean(errors)

            self.mae[fold, j] = np.mean(np.abs(errors))
            self.mse[fold, j] = np.mean((errors) ** 2)
            self.errors.append(errors)


def model_cv_results(
    results: List[PredictionResult],
    nbins: Optional[int] = None,
    bin_edges: Optional[np.ndarray] = None,
) -> CVResults:
    if nbins is None and bin_edges is None:
        raise ValueError("nbins and bin_edges cannot be both None")
    if nbins is None:
        nbins = len(bin_edges) - 1
    if bin_edges is None:
        max_y_value = np.max([r.true_RUL.max() for r in results])
        bin_edges = np.linspace(0, max_y_value, nbins + 1)

    trues = []
    predicted = []
    for results in results:
        trues.append(results.true_RUL)
        predicted.append(results.predicted_RUL)
    return CVResults(trues, predicted, nbins=nbins, bin_edges=bin_edges)


def models_cv_results(
    results_dict: Dict[str, List[PredictionResult]], nbins: int
) -> Tuple[np.ndarray, Dict[str, CVResults]]:
    """Create a dictionary with the result of each cross validation of the model"""

    max_y_value = np.max(
        [
            r.true_RUL.max()
            for model_name in results_dict.keys()
            for r in results_dict[model_name]
        ]
    )
    bin_edges = np.linspace(0, max_y_value, nbins + 1)
    model_results = {}
    for model_name in results_dict.keys():

        model_results[model_name] = model_cv_results(
            results_dict[model_name], bin_edges=bin_edges
        )

    return bin_edges, model_results


class FittedLife:
    """Represent a Fitted Life

    Parameters
    ----------

    y_true: np.array
        The true RUL target
    y_pred: np.array
        The predicted target
    time: Optional[Union[np.array, int]]
        Time feature
    fit_line_not_increasing: Optional[bool] = False,
        Wether the fitted line can increase or not.
    RUL_threshold: Optional[float]
        Indicates the thresholding value used during  de fit, By default None

    """

    def __init__(
        self,
        y_true: np.array,
        y_pred: np.array,
        time: Optional[Union[np.array, int]] = None,
        fit_line_not_increasing: Optional[bool] = False,
        RUL_threshold: Optional[float] = None,
    ):
        self.fit_line_not_increasing = fit_line_not_increasing
        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)

        if time is not None:
            self.degrading_start = FittedLife._degrading_start(y_true, RUL_threshold)
            if isinstance(time, np.ndarray):
                self.time = time
            else:
                self.time = np.array(np.linspace(0, y_true[0], n=len(y_true)))
        else:
            self.degrading_start, self.time = FittedLife.compute_time_feature(
                y_true, RUL_threshold
            )

        self.y_pred_fitted = self._fit_picewise_linear_regression(y_pred)
        self.y_true_fitted = self._fit_picewise_linear_regression(y_true)
        self.RUL_threshold = RUL_threshold
        self.y_pred = y_pred
        self.y_true = y_true

    @staticmethod
    def compute_time_feature(y_true: np.array, RUL_threshold: Optional[float] = None):
        degrading_start = FittedLife._degrading_start(y_true, RUL_threshold)
        time = FittedLife._compute_time(y_true, degrading_start)
        return degrading_start, time

    @staticmethod
    def _degrading_start(
        y_true: np.array, RUL_threshold: Optional[float] = None
    ) -> float:
        """Obtain the index when the life value is lower than the RUL_threshold

        Parameters
        ----------
        y_true : np.array
            Array of true values of the RUL of the life
        RUL_threshold : float


        Returns
        -------
        float
            if RUL_threshold is None, the degradint start if the first index.
            Otherwise it is the first index in which y_true < RUL_threshold
        """
        degrading_start = 0
        if RUL_threshold is not None:
            degrading_start_i = np.where(y_true < RUL_threshold)
            if len(degrading_start_i[0]) > 0:
                degrading_start = degrading_start_i[0][0]
        return degrading_start

    @staticmethod
    def _compute_time(y_true: np.array, degrading_start: int) -> np.array:
        """Compute the passage of time from the true RUL

        The passage of time is computed as the cumulative sum of the first
        difference of the true labels. In case there are tresholded values,
        the time steps of the thresholded zone is assumed to be as the median values
        of the time steps computed of the zones of the life in which we have information.

        Parameters
        ----------
        y_true : np.array
            The true RUL labels
        degrading_start : int
            The index in which the true RUL values starts to be lower than the treshold

        Returns
        -------
        np.array
            [description]
        """

        time_diff = np.diff(np.squeeze(y_true)[degrading_start:][::-1])
        time = np.zeros(len(y_true))
        if degrading_start > 0:
            if len(time_diff) > 0:
                time[0 : degrading_start + 1] = np.median(time_diff)
            else:
                time[0 : degrading_start + 1] = 1
        time[degrading_start + 1 :] = time_diff
        return np.cumsum(time)

    def _fit_picewise_linear_regression(self, y: np.array) -> PiecewesieLinearFunction:
        """Fit the array trough a picewise linear regression

        Parameters
        ----------
        y : np.array
            Points to be fitted

        Returns
        -------
        PiecewesieLinearFunction
            The Picewise linear function fitted
        """
        pwlr = PiecewiseLinearRegression(not_increasing=self.fit_line_not_increasing)
        for j in range(len(y)):
            pwlr.add_point(self.time[j], y[j])
        line = pwlr.finish()

        return line

    def rmse(self, sample_weight=None) -> float:
        N = len(self.y_pred)
        sw = compute_sample_weight(sample_weight, self.y_true[:N], self.y_pred)
        return np.sqrt(np.mean(sw * (self.y_true[:N] - self.y_pred) ** 2))

    def mae(self, sample_weight=None) -> float:
        N = len(self.y_pred)
        sw = compute_sample_weight(sample_weight, self.y_true[:N], self.y_pred)
        return np.mean(sw * np.abs(self.y_true[:N] - self.y_pred))

    def predicted_end_of_life(self):
        z = np.where(self.y_pred == 0)[0]
        if len(z) == 0:
            return self.time[len(self.y_pred) - 1] + self.y_pred[-1]
        else:
            return self.time[z[0]]

    def end_of_life(self):
        z = np.where(self.y_true == 0)[0]
        if len(z) == 0:
            return self.time[len(self.y_pred) - 1] + self.y_true[-1]
        else:
            return self.time[z[0]]

    def maintenance_point(self, m: float = 0):
        """Compute the maintenance point

        The maintenance point is computed as the predicted end of life - m

        Parameters
        -----------
            m: float, optional
                Fault horizon  Defaults to 0.

        Returns
        --------
            float
                Time of maintenance
        """
        return self.predicted_end_of_life() - m

    def unexploited_lifetime(self, m: float = 0):
        """Compute the unexploited lifetime given a fault horizon window

        Machine Learning for Predictive Maintenance: A Multiple Classifiers Approach
        Susto, G. A., Schirru, A., Pampuri, S., McLoone, S., & Beghi, A. (2015).

        Parameters
        ----------
            m: float, optional
                Fault horizon windpw. Defaults to 0.

        Returns:
            float: unexploited lifetime
        """

        if self.maintenance_point(m) < self.end_of_life():
            return self.end_of_life() - self.maintenance_point(m)
        else:
            return 0

    def unexpected_break(self, m: float = 0, tolerance: float = 0):
        """Compute wether an unexpected break will produce using a fault horizon window of size m

        Machine Learning for Predictive Maintenance: A Multiple Classifiers Approach
        Susto, G. A., Schirru, A., Pampuri, S., McLoone, S., & Beghi, A. (2015).

        Parameters
        ----------
            m: float, optional
                Fault horizon windpw. Defaults to 0.

        Returns
        -------
            bool
                Unexploited lifetime
        """
        if self.maintenance_point(m) - tolerance < self.end_of_life():
            return False
        else:
            return True


def split_lives_indices(y_true: np.array):
    """Obtain a list of indices for each life

    Parameters
    ----------
    y_true : np.array
        True vector with the RUL

    Returns
    -------
    List[List[int]]
        A list with the indices belonging to each life
    """
    assert len(y_true) >= 2
    lives_indices = (
        [0] + (np.where(np.diff(np.squeeze(y_true)) > 0)[0]+1).tolist() + [len(y_true)]
    )
    indices = []
    for i in range(len(lives_indices) - 1):
        r = range(lives_indices[i], lives_indices[i + 1])
        if len(r) <= 1:
            continue
        indices.append(r)
    return indices


def split_lives(
    results: PredictionResult,
    RUL_threshold: Optional[float] = None,
    fit_line_not_increasing: Optional[bool] = False,
    time: Optional[int] = None,
) -> List[FittedLife]:
    """Divide an array of predictions into a list of FittedLife Object

    Parameters
    ----------
    y_true : np.array
        The true RUL target
    y_pred : np.array
        The predicted RUL
    fit_line_not_increasing : Optional[bool], optional
        Wether the fit line can increase, by default False
    time : Optional[int], optional
        A vector with timestamps. If omitted wil be computed from y_true, by default None

    Returns
    -------
    List[FittedLife]
        FittedLife list
    """
    lives = []
    for r in split_lives_indices(results.true_RUL):
        if np.any(np.isnan(results.predicted_RUL[r])):
            continue
        lives.append(
            FittedLife(
                results.true_RUL[r],
                results.predicted_RUL[r],
                RUL_threshold=RUL_threshold,
                fit_line_not_increasing=fit_line_not_increasing,
                time=time,
            )
        )
    return lives


def split_lives_from_results(d: dict) -> List[FittedLife]:
    y_true = d["true"]
    y_pred = d["predicted"]
    return split_lives(y_true, y_pred)


def unexploited_lifetime(d: dict, window_size: int, step: int):
    bb = [split_lives_from_results(cv) for cv in d]
    return unexploited_lifetime_from_cv(bb, window_size, step)


def unexploited_lifetime_from_cv(
    lives: List[List[FittedLife]], window_size: int, n: int
):
    qq = []
    windows = np.linspace(0, window_size, n)
    for m in windows:
        jj = []
        for r in lives:

            ul_cv_list = [life.unexploited_lifetime(m) for life in r]
            mean_ul_cv = np.mean(ul_cv_list)
            std_ul_cv = np.std(ul_cv_list)
            jj.append(mean_ul_cv)
        qq.append(np.mean(jj))
    return windows, qq


def unexpected_breaks(
    d: dict, window_size: int, step: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the risk of unexpected breaks with respect to the maintenance window size

    Parameters
    ----------
    d : dict
        Dictionary with the results
    window_size : int
        Maximum size of the maintenance windows
    step : int
        Number of points in which compute the risks.
        step different maintenance windows will be used.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        * Maintenance window size evaluated
        * Risk computed for every window size used
    """

    bb = [split_lives_from_results(cv) for cv in d]
    return unexpected_breaks_from_cv(bb, window_size, step)


def unexpected_breaks_from_cv(
    lives: List[List[FittedLife]], window_size: int, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the risk of unexpected breaks given a Cross-Validation results

    Parameters
    ----------
    lives : List[List[FittedLife]]
        Cross validation results.
    window_size : int
        Maximum size of the maintenance window
    n : int
        Number of points to evaluate the risk of unexpected breaks

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        * Maintenance window size evaluated
        * Risk computed for every window size used
    """

    qq = []
    windows = np.linspace(0, window_size, n)
    for m in windows:
        jj = []
        for r in lives:
            ul_cv_list = [life.unexpected_break(m) for life in r]
            mean_ul_cv = np.mean(ul_cv_list)
            std_ul_cv = np.std(ul_cv_list)
            jj.append(mean_ul_cv)
        qq.append(np.mean(jj))
    return windows, np.array(qq)


def metric_J_from_cv(lives: List[List[FittedLife]], window_size: int, n: int, q1, q2):
    J = []
    windows = np.linspace(0, window_size, n)
    for m in windows:
        J_of_m = []
        for r in lives:
            ub_cv_list = np.array([life.unexpected_break(m) for life in r])
            ub_cv_list = (ub_cv_list / (np.max(ub_cv_list) + 0.0000000001)) * q1
            ul_cv_list = np.array([life.unexploited_lifetime(m) for life in r])
            ul_cv_list = (ul_cv_list / (np.max(ul_cv_list) + 0.0000000001)) * q2
            values = ub_cv_list + ul_cv_list
            mean_J = np.mean(values)
            std_ul_cv = np.std(values)
            J_of_m.append(mean_J)
        J.append(np.mean(J_of_m))
    return windows, J


def metric_J(d, window_size: int, step: int):
    lives_cv = [split_lives_from_results(cv) for cv in d]
    return metric_J_from_cv(lives_cv, window_size, step)


def cv_regression_metrics(
    data: List[PredictionResult], threshold: float = np.inf
) -> dict:
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
        A dictionary with the following format:
        {
            'MAE': {
                'mean': 
                'std':
            },
            'MAE SW': {
                'mean': 
                'std':
            },
            'MSE': {
                'mean': 
                'std':
            },
            

        }


    """
    errors = {
        'MAE': [],
        'MAE SW': [],
        'MSE': [],
        'MSE SW': []
    }
    for result in data:
        y_mask = np.where(result.true_RUL <= threshold)[0]
        y_true = np.squeeze(result.true_RUL[y_mask])
        y_pred = np.squeeze(result.predicted_RUL[y_mask])
        sw = compute_sample_weight(
            "relative",
            y_true,
            y_pred,
        )
        try:
            MAE_SW = mae(
                y_true,
                y_pred,
                sample_weight=sw,
            )
        except:
            MAE_SW = np.nan
        MAE = mae(y_true, y_pred)

        try:
            MSE_SW = mse(
                y_true,
                y_pred,
                sample_weight=sw,
            )
        except:
            MSE_SW = np.nan
        MSE = mse(y_true, y_pred)

        errors['MAE'].append(MAE)
        errors['MAE SW'].append(MAE_SW)
        errors['MSE'].append(MSE)
        errors['MSE SW'].append(MSE_SW)
    for k in errors.keys():
        errors[k] = {
            'mean': np.mean(errors[k]),
            'std': np.std(errors[k]),
        }
    return errors


def hold_out_regression_metrics(results: dict, CV: int = 0):
    metrics = []
    for model_name in results.keys():
        sw = compute_sample_weight(
            "relative",
            results[model_name][CV]["true"],
            results[model_name][CV]["predicted"],
        )
        MAE_SW = mae(
            results[model_name][CV]["true"],
            results[model_name][CV]["predicted"],
            sample_weight=sw,
        )
        MAE = mae(results[model_name][CV]["true"], results[model_name][CV]["predicted"])

        MSE_SW = mse(
            results[model_name][CV]["true"],
            results[model_name][CV]["predicted"],
            sample_weight=sw,
        )
        MSE = mse(results[model_name][CV]["true"], results[model_name][CV]["predicted"])
        metrics.append((model_name, MAE_SW, MAE, MSE_SW, MSE))
    return pd.DataFrame(
        metrics,
        columns=["Model", "MAE sample weight", "MAE", "MSE sample weight", "MSE"],
    )
