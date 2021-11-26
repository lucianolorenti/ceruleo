"""Compute evaluating results of fitted models

One of the most important issues regarding PM is the ability to compare and evaluate different methods.

The main data structure used in the results module is a dictionary in which each of the keys is the model name, 
and the elements are a list of PredictionResult. 
Each element of the model array is interpreted as a Fold in CV settings. 

Additionally to the regression error, it is possible to compute some metrics more easily interpretable. In this context, two metrics were defined in [1], namely:

- Frequency of Unexpected Breaks (ρUB) - the percentage of failures not prevented;
- Amount of Unexploited Lifetime (ρUL) - the average number of time that could have been run before failure if the preventative maintenance suggested by the maintenance management mod-ule had not been performed.

When compaing between the predicted end of life with respect to the true end of that particular life three scenarios can happen:

- The predicted end of life occurs before the true one. In that case, the predictions were pessimistic and the tool could have been used more time.
- The remaining useful life arrives at zero after the true remaining useful life. In that case, we incur the risk of the tool breaking.  
- The predicted line coincides with the true line. In that case, we don’t have unexploited time, and the risk of breakage can be considered 0.

Since usually the breakages are considered more harmful, a possible approach to preventing unexpected failures is to consider a more conservative maintenance approach, providing maintenance tasks recommendations some time before the end of life predicted. In that way, a conservative window can be defined in which the recommendation of making maintenance task should be performed at time T-predicted - conservative window size.

[1] Machine Learning for Predictive Maintenance: A Multiple Classifiers Approach
    Susto, G. A., Schirru, A., Pampuri, S., McLoone, S., & Beghi, A. (2015). 


"""
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from ceruleo.results.picewise_regression import (PiecewesieLinearFunction,
                                                PiecewiseLinearRegression)
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from uncertainties import ufloat

logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    """An object that store regression metrics and times
    """
    mae: float
    mse: float
    fitting_time: float = 0
    prediction_time: float = 0


@dataclass
class PredictionResult:
    """A prediction result is composed by a name
    """
    name: str
    true_RUL: np.ndarray
    predicted_RUL: np.ndarray
    metrics: MetricsResult = MetricsResult(0, 0)

    def compute_metrics(self):
        self.metrics.mae = mae(self.true_RUL, self.predicted_RUL)
        self.metrics.mse = mse(self.true_RUL, self.predicted_RUL)


    def __init__(self, name:str,     true_RUL: np.ndarray, predicted_RUL: np.ndarray):
        self.name = name
        self.true_RUL = np.squeeze(true_RUL)
        self.predicted_RUL = np.squeeze(predicted_RUL)
        self.compute_metrics()


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
    """
        Compute the error histogram

        Compute the error with respect to the RUL considering the results of different
        folds

        Parameters:
            y_true: List with the true values of each hold-out set of a cross validation
            y_pred: List with the predictions of each hold-out set of a cross validation
            nbins: Number of bins to compute the histogram

    """
    def __init__(
        self,
        y_true: List[List],
        y_pred: List[List],
        nbins: int = 5,
        bin_edges: Optional[np.array] = None,
    ):

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
    """Represent a Fitted run-to-cycle failure

    Parameters:

        y_true: The true RUL target
        y_pred: The predicted target
        time: Time feature
        fit_line_not_increasing: Wether the fitted line can increase or not.
        RUL_threshold: Indicates the thresholding value used during  de fit

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

        # self.y_pred_fitted_picewise = self._fit_picewise_linear_regression(y_pred)
        # self.y_true_fitted_picewise = self._fit_picewise_linear_regression(y_true)

        self.RUL_threshold = RUL_threshold
        self.y_pred = y_pred
        self.y_true = y_true

        self.y_pred_fitted_coefficients = np.polyfit(self.time, self.y_pred, 1)
        p = np.poly1d(self.y_pred_fitted_coefficients)
        self.y_pred_fitted = p(self.time)

        self.y_true_fitted_coefficients = np.polyfit(self.time, self.y_true, 1)
        p = np.poly1d(self.y_true_fitted_coefficients)
        self.y_true_fitted = p(self.time)

    @staticmethod
    def compute_time_feature(y_true: np.array, RUL_threshold: Optional[float] = None) -> Tuple[float, np.ndarray]:
        """Compute the time feature based on the target

        Parameters:

            y_true: RUL target
            RUL_threshold:

        Returns
        -------

            Degradind start time and time
        """
        degrading_start = FittedLife._degrading_start(y_true, RUL_threshold)
        time = FittedLife._compute_time(y_true, degrading_start)
        return degrading_start, time

    @staticmethod
    def _degrading_start(
        y_true: np.array, RUL_threshold: Optional[float] = None
    ) -> float:
        """Obtain the index when the life value is lower than the RUL_threshold

        Parameters:

            y_true: Array of true values of the RUL of the life
            RUL_threshold: float


        Return:

            degrading_start: if RUL_threshold is None, the degradint start if the first index.
            Otherwise it is the first index in which y_true < RUL_threshold
        """
        degrading_start = 0
        if RUL_threshold is not None:
            degrading_start_i = np.where(y_true < RUL_threshold)
            if len(degrading_start_i[0]) > 0:
                degrading_start = degrading_start_i[0][0]
        else:
            d = np.diff(y_true) == 0
            while (degrading_start< len(d)) and (d[degrading_start]):
                degrading_start += 1
        return degrading_start

    @staticmethod
    def _compute_time(y_true: np.array, degrading_start: int) -> np.array:
        """Compute the passage of time from the true RUL

        The passage of time is computed as the cumulative sum of the first
        difference of the true labels. In case there are tresholded values,
        the time steps of the thresholded zone is assumed to be as the median values
        of the time steps computed of the zones of the life in which we have information.

        Parameters:

            y_true: The true RUL labels
            degrading_start : The index in which the true RUL values starts to be lower than the treshold

        Returns:

            t: Time component
        """
        if len(y_true) == 1:
            return np.array([0])
        
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

    def noisiness(self) -> float:
        """How much the predictions resemble a line

        This metric is computed as the mse of the fitted values
        with respect to the least squares fitted line of this
        values
        """
        return mae(self.y_pred_fitted, self.y_pred)

    def slope_resemblance(self):
        m1 = self.y_true_fitted_coefficients[0]
        m2 = self.y_pred_fitted_coefficients[0]
        d = np.arctan((m1 - m2) / (1 + m1 * m2))
        d = d / (np.pi / 2)
        return 1 - np.abs((d / (np.pi / 2)))

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

        Parameters:

            m: Fault horizon windpw.

        Returns:

            Unexploited lifetime
        """
        if self.maintenance_point(m) - tolerance < self.end_of_life():
            return False
        else:
            return True


def split_lives_indices(y_true: np.array) -> List[List[int]]:
    """Obtain a list of indices for each life

    Parameters:
        y_true: True vector with the RUL

    Returns:
         l: A list with the indices belonging to each life
    """
    assert len(y_true) >= 2
    lives_indices = (
        [0]
        + (np.where(np.diff(np.squeeze(y_true)) > 0)[0] + 1).tolist()
        + [len(y_true)]
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

    Parameters:
        y_true: The true RUL target
        y_pred: The predicted RUL
        fit_line_not_increasing : Optional[bool], optional
            Wether the fit line can increase, by default False
        time:  A vector with timestamps. If omitted wil be computed from y_true

    Returns:
        lives: FittedLife list
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





def unexploited_lifetime(d: PredictionResult, window_size: int, step: int):
    bb = [split_lives(cv) for cv in d]
    return unexploited_lifetime_from_cv(bb, window_size, step)


def unexploited_lifetime_from_cv(
    lives: List[List[FittedLife]], window_size: int, n: int
):
    std_per_window = []
    mean_per_window = []
    windows = np.linspace(0, window_size, n)
    for m in windows:
        jj = []
        for r in lives:

            ul_cv_list = [life.unexploited_lifetime(m) for life in r]

            jj.extend(ul_cv_list)
        mean_per_window.append(np.mean(jj))
        std_per_window.append(np.std(jj))

    return windows, np.array(mean_per_window), np.array(std_per_window)


def unexpected_breaks(
    d: List[PredictionResult], window_size: int, step: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the risk of unexpected breaks with respect to the maintenance window size

    Parameters:
    ----------
    d: Dictionary with the results
    window_size: Maximum size of the maintenance windows
    step: Number of points in which compute the risks.
        step different maintenance windows will be used.

    Returns:
    Tuple[np.ndarray, np.ndarray]
        * Maintenance window size evaluated
        * Risk computed for every window size used
    """

    bb = [split_lives(fold) for fold in d]
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
    std_per_window = []
    mean_per_window = []
    windows = np.linspace(0, window_size, n)
    for m in windows:
        jj = []
        for r in lives:
            ul_cv_list = [life.unexpected_break(m) for life in r]
            jj.extend(ul_cv_list)
        mean_per_window.append(np.mean(jj))
        std_per_window.append(np.std(jj))
    return windows, np.array(mean_per_window), np.array(std_per_window)


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
    lives_cv = [split_lives(cv) for cv in d]
    return metric_J_from_cv(lives_cv, window_size, step)


def cv_regression_metrics_single_model(
    results: List[PredictionResult], threshold: float = np.inf
):
    errors = {
        "MAE": [],
        "MAE SW": [],
        "MSE": [],
        "MSE SW": [],
        "MAPE": []
    }
    for result in results:
        y_mask = np.where(result.true_RUL <= threshold)[0]
        y_true = np.squeeze(result.true_RUL[y_mask])
        y_pred = np.squeeze(result.predicted_RUL[y_mask])
        mask = np.isfinite(y_pred)
        y_pred = y_pred[mask]
        y_true = y_true[mask]

        if len(np.unique(y_pred)) == 1:
            continue


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
        try:
            MAE = mae(y_true, y_pred)
        except:
            MAE = np.nan

        try:
            MSE_SW = mse(
                y_true,
                y_pred,
                sample_weight=sw,
            )
        except:
            MSE_SW = np.nan

        try:
            MSE = mse(y_true, y_pred)
        except:
            MSE = np.nan

        try:
            MAPE = mape(y_true, y_pred)
        except:
            MAPE = np.nan

        lives = split_lives(result)
        errors["MAE"].append(MAE)
        errors["MAE SW"].append(MAE_SW)
        errors["MSE"].append(MSE)
        errors["MSE SW"].append(MSE_SW)
        errors["MAPE"].append(MAPE)

    errors1 = {}
    for k in errors.keys():
        errors1[k] = ufloat(np.round(np.mean(errors[k]),2), np.round(np.std(errors[k]), 2))
    return errors1


def cv_regression_metrics(
    results_dict: Dict[str, List[PredictionResult]], threshold: float = np.inf
) -> dict:
    """Compute regression metrics for each model

    Parameters:
    
        data: Dictionary with the model predictions.
            
        threshold: Compute metrics errors only in RUL values less than the threshold

    Returns:


        d: { ['Model]: 
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
            ]
            
    """
    out = {}
    for model_name in results_dict.keys():
        out[model_name] = cv_regression_metrics_single_model(
            results_dict[model_name], threshold
        )
    return out
