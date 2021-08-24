import numpy as np
import tensorflow as tf
from scipy.special import loggamma
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Concatenate, Dense, Lambda, Multiply


class TFWeibullDistribution:
    @staticmethod
    def log_likelihood(x: tf.Tensor, alpha: tf.Tensor, beta: tf.Tensor):
        ya = (x + 0.00000000001) / alpha
        return tf.reduce_mean(K.log(beta) + (beta * K.log(ya)) - K.pow(ya, beta))

    @staticmethod
    def kl_divergence(l1: tf.Tensor, k1: tf.Tensor, l2: tf.Tensor, k2: tf.Tensor):
        term_1 = K.log(k1 / K.pow(l1, k1))
        term_2 = K.log(k2 / K.pow(l2, k2))
        term_3 = (k1 - k2) * (K.log(l1) - 0.5772 / k1)
        term_4 = K.pow((l1 / l2), k2) * K.exp(tf.math.lgamma((k2 / k1) + 1))
        tf.print(term_1, term_2, term_3, term_4)
        return K.mean(term_1 - term_2 + term_3 + term_4 - 1)


class WeibullDistribution:
    @staticmethod
    def mean(alpha, beta):
        return alpha * np.exp(loggamma(1 + (1 / beta)))

    @staticmethod
    def mode(alpha, beta):
        vmode = alpha * np.power((beta - 1) / beta, 1 / beta)
        vmode[beta <= 1] = 0
        return vmode

    @staticmethod
    def median(alpha, beta):
        return alpha * (np.power(np.log(2.0), (1 / beta)))

    @staticmethod
    def variance(alpha, beta):
        return alpha ** 2 * (
            np.exp(loggamma(1 + (2 / beta))) - (np.exp(loggamma(1 + (1 / beta)))) ** 2
        )

    @staticmethod
    def quantile(q, alpha, beta):
        return alpha * np.power(-np.log(1 - q), 1 / beta)


class NotCensoredWeibull(tf.keras.losses.Loss):
    def __init__(self, regression_weight: float = 5, likelihood_weight:float = 1):
        super().__init__()
        self.regression_weight = regression_weight
        self.likelihood_weight = likelihood_weight

    def call(self, y_true, y_pred):
        pRUL = y_pred[:, 0]
        alpha = y_pred[:, 1]
        beta = y_pred[:, 2]
        y_true = tf.squeeze(y_true)

        reg_loss = tf.keras.losses.MeanAbsoluteError()(pRUL, y_true)

        log_liks = TFWeibullDistribution.log_likelihood(y_true, alpha, beta)

        # log_liks = K.clip(log_liks, K.log(0.0000000001), K.log(1 - 0.0000000001))
        # + kl_weibull(alpha, beta, alpha, 2.0 )
        loss = -self.likelihood_weight*log_liks + self.regression_weight * reg_loss
        # + K.pow(ya,beta)
        return loss


class WeibullLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        return_params=True,
        regression="mode",
        name="WeibullParams",
        *args,
        **kwargs
    ):
        super().__init__(name=name, *args, **kwargs)
        self.return_params = return_params
        if self.return_params:
            self.params = Concatenate(name="Weibullparams")
        if regression == "mode":
            self.fun = self.mode
        elif regression == "mean":
            self.fun = self.mean
        elif regression == "median":
            self.fun = self.median

    def mean(self, lambda_pipe, k_pipe):
        inner_gamma = Lambda(lambda x: tf.math.exp(tf.math.lgamma(1 + (1 / x))))(k_pipe)
        return Multiply(name="RUL")([lambda_pipe, inner_gamma])

    def median(self, lambda_pipe, k_pipe):
        return lambda_pipe * (tf.math.pow(tf.math.log(2.0), tf.math.reciprocal(k_pipe)))

    def mode(self, alpha, beta):
        mask = K.cast(K.greater(beta, 1), tf.float32)
        beta = tf.clip_by_value(beta, 1 + 0.00000000001, np.inf)
        return mask * alpha * tf.math.pow((beta - 1) / beta, (1 / beta))

    def _result(self, alpha, beta):
        RUL = self.fun(alpha, beta)
        if self.return_params:
            return self.params([alpha, beta])
        else:
            return RUL


class WeibullParameters(WeibullLayer):
    def __init__(self, hidden, regression="mode", return_params=True, *args, **kwargs):
        super(WeibullParameters, self).__init__(
            return_params=True, regression=regression, name="", *args, **kwargs
        )

        self.W = Dense(hidden, activation="relu")

        self.xalpha1 = Dense(hidden, activation="relu")
        self.xalpha2 = Dense(1, name="w_alpha", activation='softplus')

        self.xbeta1 = Dense(hidden, activation="relu")
        self.xbeta2 = Dense(1, name="w_beta", activation='softplus')

    def call(self, input_tensor, training=False):

        x = self.W(input_tensor)

        alpha = self.xalpha1(x)
        alpha = self.xalpha2(alpha)

        beta = self.xbeta1(x)
        beta = self.xbeta2(beta)

        RUL = self.mode(alpha, beta)

        x = Concatenate(axis=1)([RUL, alpha, beta])

        return x
