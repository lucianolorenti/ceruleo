import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from ceruleo.models.keras.dataset import KerasTrainableModel
from ceruleo.models.keras.losses import weighted_categorical_crossentropy
from ceruleo.models.keras.weibull import WeibullLayer
from sklearn.base import BaseEstimator, TransformerMixin
from tcn import TCN
from tensorflow.keras import Model
from tensorflow.keras.layers import (Attention, BatchNormalization, Dense,
                                     Dropout, Input)

tfd = tfp.distributions


class VariationalWeibull(WeibullLayer):
    def __init__(self, hidden):
        super(VariationalWeibull, self).__init__(
            return_params=True, name='')
        self.x1 = Dense(hidden,
                        activation='relu')

        self.scale_uniform = Dense(1, activation='relu')

        self.x2 = Dense(hidden,
                        activation='relu')

        self.k1 = Dense(1, activation='relu')
        self.k2 = Dense(1, activation='relu')

        self.uniform_sampler = tfd.Sample(
            tfd.Uniform(),
            sample_shape=(1))

    def call(self, input):

        lambda_ = self.x1(input)
        lambda_ = tf.math.exp(self.scale_uniform(lambda_))
        uniform = self.uniform_sampler.sample(tf.shape(input)[0])
        lambda_ = uniform*lambda_

        k = self.x2(input)
        k1 = self.k1(k)
        k2 = self.k2(k)
        uniform1 = self.uniform_sampler.sample(tf.shape(input)[0])

        k = tf.nn.softplus(
            k1*tf.math.pow((-1 * tf.math.log(1-uniform1)), k2) + 1)

        return self._result(lambda_, k)


class RawAndBinClasses(BaseEstimator, TransformerMixin):
    """
        A target transformer that outputs
        the RUL + nbins binary vectors
    """

    def __init__(self, nbins):
        self.nbins = nbins

    def fit(self, X, y=None):
        self.max_RUL = int(X.max())
        self.value_ranges = np.linspace(0, self.max_RUL, num=self.nbins+1)
        return self

    def transform(self, X):
        v = X
        classes = []
        for j in range(len(self.value_ranges)-1):
            lower = self.value_ranges[j]
            upper = self.value_ranges[j+1]
            classes.append(((v >= lower) & (v < upper)))
        v = np.vstack((v, *classes)).T
        return v


class SoftmaxRegression(KerasTrainableModel):
    """
    The network contains stacked layers of 1-dimensional convolutional layers
    followed by max poolings

    Parameters
    ----------
    self: list of tuples (filters: int, kernel_size: int)
          Each element of the list is a layer of the network. The first element of the tuple contaings
          the number of filters, the second one, the kernel size.
    """

    def __init__(self, raw_and_bins, alpha, window, batch_size, step, transformer, shuffle, models_path,
                 patience=4, cache_size=30, output_size=3, padding='same'):
        super(SoftmaxRegression, self).__init__(window,
                                                batch_size,
                                                step,
                                                transformer,
                                                shuffle,
                                                models_path,
                                                patience=patience,
                                                output_size=output_size,
                                                cache_size=30,
                                                callbacks=[])
        if raw_and_bins is not None:
            self.raw_and_bins = raw_and_bins
            weights = [1 for _ in range(self.raw_and_bins.nbins)]
            self.wcc = weighted_categorical_crossentropy(weights)
            self.output_size = self.raw_and_bins.nbins
        else:
            self.output_size = 1
        self.alpha = alpha

    def _generate_batcher(self, train_batcher, val_batcher):
        n_features = self.transformer.n_features

        def gen_train():
            for X, y in train_batcher:
                yield X, y

        def gen_val():
            for X, y in val_batcher:
                yield X, y

        a = tf.data.Dataset.from_generator(
            gen_train, (tf.float32, tf.float32),
            (tf.TensorShape([None, self.window, n_features]),
             tf.TensorShape([None, self.output_size])))
        b = tf.data.Dataset.from_generator(
            gen_val, (tf.float32, tf.float32),
            (tf.TensorShape([None, self.window, n_features]),
             tf.TensorShape([None, self.output_size])))
        return a, b

    def _loss(self, y_true, y_pred):
        # cross entropy loss
        bin_true = y_true[:, 1:]
        cont_true = y_true[:, 0]

        # y_pred_rul = y_pred[:, 0]
        # y_pred_bins = y_pred[:, 1:]
        y_pred_bins = y_pred
        cls_loss = self.wcc(bin_true, y_pred_bins)
        # MSE loss
        idx_tensor = self.raw_and_bins.value_ranges[:-1]
        pred_cont = tf.reduce_sum(y_pred_bins * idx_tensor, 1)
        # pred_cont = tf.keras.backend.argmax(y_pred, axis=1)
        rmse_loss_softmax = tf.losses.mean_squared_error(cont_true, pred_cont)

        # mse_loss = tf.losses.mean_squared_error(cont_true, y_pred_rul)
        # Total loss
        total_loss = (cls_loss +
                      self.alpha * rmse_loss_softmax
                      )
        return total_loss

    def mse_softmax(self, y_true, y_pred):
        # cross entropy loss
        # bin_true = y_true[:, 1:]
        cont_true = y_true[:, 0]

        # y_pred_rul = y_pred[:, 0]
        # y_pred_bins = y_pred[:, 1:]
        y_pred_bins = y_pred
        idx_tensor = self.raw_and_bins.value_ranges[:-1]
        pred_cont = tf.reduce_sum(y_pred_bins * idx_tensor, 1)
        # pred_cont = tf.keras.backend.argmax(y_pred, axis=1)
        return tf.sqrt(tf.losses.mean_squared_error(cont_true, pred_cont))

    def mse_rul(self,  y_true, y_pred):
        # cross entropy loss
        cont_true = y_true[:, 0]
        y_pred_rul = y_pred[:, 0]
        return tf.losses.mean_squared_error(cont_true, y_pred_rul)

    def compile(self):
        self.compiled = True
        self.model.compile(loss='mse',

                           optimizer=tf.keras.optimizers.Adam(lr=0.0001))

    def build_model(self):

        # function to split the input in multiple outputs
        def splitter(x):
            return [x[:, :, i:i+1] for i in range(n_features)]

        n_features = self.transformer.n_features

        i = Input(shape=(self.window, n_features))

        m = TCN(use_skip_connections=True,
                use_batch_norm=True,
                return_sequences=True,
                dropout_rate=0.1)(i)
        m = Attention(64, self.window-1)(m)
        m = Dense(100, activation='relu')(m)
        m = Dropout(0.5)(m)
        m = BatchNormalization()(m)
        proba = Dense(150, activation='relu')(m)
        proba = BatchNormalization()(proba)
        proba = Dropout(0.1)(proba)
        proba = Dense(1, activation='linear')(proba)

        return Model(inputs=i, outputs=proba)

    @property
    def name(self):
        return 'ConvolutionalSimple'

    def get_params(self, deep=False):
        params = super().get_params()
        params.update({
        })
        return params
