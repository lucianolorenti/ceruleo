
from typing import List

import tensorflow as tf
from ceruleo.models.keras.dataset import KerasTrainableModel
from ceruleo.models.keras.losses import time_to_failure_rul
from tensorflow.keras import Input, Model, optimizers
from tensorflow.keras.layers import LSTM, Dense


class MultiTaskRUL(KerasTrainableModel):
    """
        A Multi task network that learns to regress the RUL and the Time to failure

        Two Birds with One Network: Unifying Failure Event Prediction and Time-to-failure Modeling
        Karan Aggarwal, Onur Atan, Ahmed K. Farahat, Chi Zhang, Kosta Ristovski, Chetan Gupta

        The target

        Parameters
        -----------
        layers_lstm : List[int]
                      Number of LSTM layers
        layers_dense : List[int]
                       Number of dense layers
        window: int

        batch_size: int
        step: int
        transformer
        shuffle
        models_path
        patience: int = 4
        cache_size: int = 30
    """

    def __init__(self,
                 layers_lstm: List[int],
                 layers_dense: List[int],
                 window: int,
                 batch_size: int,
                 step: int, transformer, shuffle, models_path,
                 patience: int = 4, cache_size: int = 30):
        super().__init__(window,
                         batch_size,
                         step,
                         transformer,
                         shuffle,
                         models_path,
                         patience=4,
                         cache_size=30)
        self.layers_dense = layers_dense
        self.layers_lstm = layers_lstm

    def compile(self):
        super().compile()
        self.model.compile(
            optimizer=optimizers.Adam(lr=0.001),
            loss=time_to_failure_rul(weights={
                0: 1.,
                1: 2.
            }),
            # {
            #    'rul': MeanSquaredError(),
            #    'ttf': BinaryCrossentropy(from_logits=True),
            # },
            loss_weights=[1.0, 1.0],
        )

    @property
    def name(self):
        return "MultiTaskRULTTF"

    def build_model(self):
        n_features = self.transformer.n_features

        input = Input(shape=(self.window, n_features))
        x = input
        n_filters = 15
        if len(self.layers_lstm) > 1:
            for n_filters in self.layers_lstm:
                x = LSTM(n_filters, recurrent_dropout=0.2,
                         return_sequences=True)(x)

        x = LSTM(n_filters, recurrent_dropout=0.2, return_sequences=False)(x)

        for n_filters in self.layers_dense:
            x = Dense(n_filters, activation='elu')(x)

        RUL_output = Dense(1, activation='elu', name='rul')(x)

        FP_output = Dense(1, activation='sigmoid', name='ttf')(x)

        output = tf.keras.layers.Concatenate(axis=1)([RUL_output, FP_output])
        model = Model(
            inputs=[input],
            outputs=[output],
        )
        return model

    def _generate_batcher(self, train_batcher, val_batcher):
        n_features = self.transformer.n_features

        def gen_train():
            for X, y in train_batcher:
                yield X, y

        def gen_val():
            for X, y in val_batcher:
                yield X, y

        a = tf.data.Dataset.from_generator(
            gen_train, (tf.float32, tf.float32), (tf.TensorShape(
                [None, self.window, n_features]), tf.TensorShape([None, 2])))
        b = tf.data.Dataset.from_generator(
            gen_val, (tf.float32, tf.float32), (tf.TensorShape(
                [None, self.window, n_features]), tf.TensorShape([None, 2])))
        return a, b
