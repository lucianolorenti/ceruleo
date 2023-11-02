import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


from tensorflow.python.keras.losses import LossFunctionWrapper


def root_mean_squared_error(y_true: np.array, y_pred: np.array) -> float:
    """Root mean squared error

    Parameters:
            y_true: True RUL values
            y_pred: Predicted RUL values
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))


def asymmetric_loss_pm(
    y_true,
    y_pred,
    *,
    theta_l,
    alpha_l,
    gamma_l,
    theta_r,
    alpha_r,
    gamma_r,
    relative_weight: bool = True,
) -> float:
    """
    Customizable Asymmetric Loss Functions for Machine Learning-based Predictive Maintenance


    Ehrig, L., Atzberger, D., Hagedorn, B., Klimke, J., & Döllner, J. (2020, October).
    Customizable Asymmetric Loss Functions for Machine Learning-based Predictive Maintenance.
    In 2020 8th International Conference on Condition Monitoring and Diagnosis
    (CMD) (pp. 250-253). IEEE.

    [Reference](https://ieeexplore.ieee.org/document/9287246)

    Parameters:
        y_true: True RUL values
        y_pred: Predicted RUL values
        theta_l: Linear to exponential change point for overpredictions (Positive)
        alpha_l: Quadratic term parameters for overpredictions
        gamma_l: Exponential term parameters for overpredictions
        theta_r: Linear to exponential change point for underpredictions
        alpha_r: Quadratic term parameters for underpredictions
        gamma_r: Exponential term parameters for underpredictions
        relative_weight: Wether to use weigthing relative to the RUL

    Returns:
        l: the loss computed
    """

    errors = y_true - y_pred
    weight = tf.abs(errors) / (
        tf.clip_by_value(y_true, clip_value_min=0.9, clip_value_max=np.inf)
    )

    ll_exp = tf.cast(K.less(errors, -theta_l), errors.dtype)
    ll_quad = (1 - ll_exp) * tf.cast(K.less(errors, 0), errors.dtype)

    lr_exp = tf.cast(K.greater(errors, theta_r), errors.dtype)
    lr_quad = (1 - lr_exp) * tf.cast(K.greater(errors, 0), errors.dtype)
    ll_exp = ll_exp * (
        alpha_l
        * theta_l
        * (theta_l + 2 * gamma_l * (K.exp((K.abs(errors) - theta_l) / (gamma_l)) - 1))
    )
    ll_quad = ll_quad * alpha_l * K.pow(errors, 2)

    lr_exp = lr_exp * (
        alpha_r
        * theta_r
        * (theta_r + 2 * gamma_r * (K.exp((errors - theta_r) / (gamma_r)) - 1))
    )
    lr_quad = lr_quad * alpha_r * K.pow(errors, 2)

    if relative_weight:
        a = tf.reduce_mean(weight * (ll_exp + ll_quad + lr_exp + lr_quad))
    else:
        a = tf.reduce_mean(ll_exp + ll_quad + lr_exp + lr_quad)

    return a


class AsymmetricLossPM(LossFunctionWrapper):
    """
    Customizable Asymmetric Loss Functions for Machine Learning-based Predictive Maintenance


    Ehrig, L., Atzberger, D., Hagedorn, B., Klimke, J., & Döllner, J. (2020, October).
    Customizable Asymmetric Loss Functions for Machine Learning-based Predictive Maintenance.
    In 2020 8th International Conference on Condition Monitoring and Diagnosis
    (CMD) (pp. 250-253). IEEE.

    [Reference](https://ieeexplore.ieee.org/document/9287246)

    Parameters:
        theta_l: Linear to exponential change point for overpredictions
        alpha_l: Quadratic term parameters for overpredictions
        gamma_l: Exponential term parameters for overpredictions
        theta_r: Linear to exponential change point for underpredictions
        alpha_r: Quadratic term parameters for underpredictions
        gamma_r: Exponential term parameters for underpredictions
        relative_weight: Wether to use weigthing relative to the RUL
    """

    def __init__(
        self,
        *,
        theta_l: float,
        alpha_l: float,
        gamma_l: float,
        theta_r: float,
        alpha_r: float,
        gamma_r: float,
        relative_weight: bool = True,
        name="asymmetric_loss_pm",
    ):
        super().__init__(
            asymmetric_loss_pm,
            theta_l=theta_l,
            alpha_l=alpha_l,
            gamma_l=gamma_l,
            theta_r=theta_r,
            alpha_r=alpha_r,
            gamma_r=gamma_r,
            relative_weight=relative_weight,
            name=name,
        )


def relative_mae(C: float = 0.9):
    """
    MAE weighted by the relative error

    Parameters:
        C: Minimal value for the RUL

    Returns:
        The loss function
    """
    mae = tf.keras.losses.MeanAbsoluteError()

    def concrete_relative_mae(y_true, y_pred):
        errors = y_true - y_pred
        sw = tf.abs(errors) / (
            tf.clip_by_value(y_true, clip_value_min=C, clip_value_max=np.inf)
        )
        return mae(y_true, y_pred, sample_weight=sw)

    return concrete_relative_mae


def relative_mse(C: float = 0.9):
    """
    MSE weighted by the relative error

    Parameters:
        C: Minimal value for the RUL

    Returns:
        The loss function
    """
    mse = tf.keras.losses.MeanSquaredError()

    def concrete_relative_mse(y_true, y_pred):
        errors = y_true - y_pred
        sw = tf.abs(errors) / (
            tf.clip_by_value(y_true, clip_value_min=C, clip_value_max=np.inf)
        )
        return mse(y_true, y_pred, sample_weight=sw)

    return concrete_relative_mse
