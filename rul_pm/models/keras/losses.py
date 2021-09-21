import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import mse
import numpy as np
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.util.tf_export import keras_export
from rul_pm.models.keras.weibull import NotCensoredWeibull


def class_to_reg(y_true, y_pred, bins):

    aa = tf.convert_to_tensor(np.array([bins]).T, dtype=tf.float32)
    y_pred = tf.matmul(y_pred, aa)

    y_true = tf.gather(aa, tf.cast(tf.squeeze(y_true), tf.int64))
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))


def weighted_binary_cross_entropy(weights: dict, from_logits: bool = False):
    """
    Return a function for calculating weighted binary cross entropy
    It should be used for multi-hot encoded labels

    # Example
    y_true = tf.convert_to_tensor([1, 0, 0, 0, 0, 0], dtype=tf.int64)
    y_pred = tf.convert_to_tensor(
        [0.6, 0.1, 0.1, 0.9, 0.1, 0.], dtype=tf.float32)
    weights = {
        0: 1.,
        1: 2.
    }
    # with weights
    loss_fn = get_loss_for_multilabels(weights=weights, from_logits=False)
    loss = loss_fn(y_true, y_pred)
    # tf.Tensor(0.6067193, shape=(), dtype=float32)

    # without weights
    loss_fn = get_loss_for_multilabels()
    loss = loss_fn(y_true, y_pred)
    # tf.Tensor(0.52158177, shape=(), dtype=float32)

    # Another example
    y_true = tf.convert_to_tensor([[0., 1.], [0., 0.]], dtype=tf.float32)
    y_pred = tf.convert_to_tensor([[0.6, 0.4], [0.4, 0.6]], dtype=tf.float32)
    weights = {
        0: 1.,
        1: 2.
    }
    # with weights
    loss_fn = get_loss_for_multilabels(weights=weights, from_logits=False)
    loss = loss_fn(y_true, y_pred)
    # tf.Tensor(1.0439969, shape=(), dtype=float32)

    # without weights
    loss_fn = get_loss_for_multilabels()
    loss = loss_fn(y_true, y_pred)
    # tf.Tensor(0.81492424, shape=(), dtype=float32)

    @param weights A dict setting weights for 0 and 1 label. e.g.
        {
            0: 1.
            1: 8.
        }
        For this case, we want to emphasise those true (1) label,
        because we have many false (0) label. e.g.
            [
                [0 1 0 0 0 0 0 0 0 1]
                [0 0 0 0 1 0 0 0 0 0]
                [0 0 0 0 1 0 0 0 0 0]
            ]



    @param from_logits If False, we apply sigmoid to each logit
    @return A function to calcualte (weighted) binary cross entropy
    """
    assert 0 in weights
    assert 1 in weights

    def weighted_cross_entropy_fn(y_true, y_pred):
        tf_y_true = tf.cast(y_true, dtype=y_pred.dtype)
        tf_y_pred = tf.cast(y_pred, dtype=y_pred.dtype)

        weights_v = tf.where(tf.equal(tf_y_true, 1), weights[1], weights[0])
        ce = K.binary_crossentropy(tf_y_true, tf_y_pred, from_logits=from_logits)
        loss = K.mean(tf.multiply(ce, weights_v))
        return loss

    return weighted_cross_entropy_fn


def time_to_failure_rul(weights: dict):

    binary_loss_fun = weighted_binary_cross_entropy(weights)

    def time_to_failure_rul_fun(y_true, y_pred):
        """ Final loss calculation function to be passed to optimizer"""
        mse_loss = mse(y_true[:, 0], y_pred[:, 0])
        binary_loss = binary_loss_fun(y_true[:, 1], y_pred[:, 1])
        return binary_loss + mse_loss

    return time_to_failure_rul_fun


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        weights = np.array([0.5,2,10])
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss


def root_mean_squared_error(y_true, y_pred):
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
):

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
    """Customizable Asymmetric Loss Functions for Machine Learning-based Predictive Maintenance

    Parameters
    ----------
    theta_l : float
        [description]
    alpha_l : float
        [description]
    gamma_l : float
        [description]
    theta_r : float
        [description]
    alpha_r : float
        [description]
    gamma_r : float
        [description]
    relative_weight : bool, optional
        [description], by default Truename='asymmetric_loss_pm'
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
    mae = tf.keras.losses.MeanAbsoluteError()

    def concrete_relative_mae(y_true, y_pred):
        errors = y_true - y_pred
        sw = tf.abs(errors) / (
            tf.clip_by_value(y_true, clip_value_min=C, clip_value_max=np.inf)
        )
        return mae(y_true, y_pred, sample_weight=sw)

    return concrete_relative_mae


def relative_mse(C: float = 0.9):
    mse = tf.keras.losses.MeanSquaredError()

    def concrete_relative_mse(y_true, y_pred):
        errors = y_true - y_pred
        sw = tf.abs(errors) / (
            tf.clip_by_value(y_true, clip_value_min=C, clip_value_max=np.inf)
        )
        return mse(y_true, y_pred, sample_weight=sw)

    return concrete_relative_mse


def get(loss_info: dict):
    if "class_name" in loss_info:
        return tf.keras.losses.get(**loss_info)
    if loss_info["type"] == "weibull_not_censored":
        return NotCensoredWeibull(**loss_info.get("config", {}))
    elif loss_info["type"] == "asymmetric":
        return AsymmetricLossPM(**loss_info.get("config", {}))
    else:
        return tf.keras.losses.get(loss_info["type"])
