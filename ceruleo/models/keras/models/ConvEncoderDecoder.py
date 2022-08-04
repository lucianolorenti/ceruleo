import tensorflow as tf
from ceruleo.models.keras.dataset import KerasTrainableModel
from tcn import TCN
from tensorflow.keras import Input, Model, optimizers
from tensorflow.keras.layers import (AveragePooling1D, Concatenate, Conv1D,
                                     Dense, Dropout, Flatten, Lambda,
                                     UpSampling1D)


class EncoderDecoder(KerasTrainableModel):

    """

        Parameters
        -----------
        n_filters : int

        filter_size : int

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
                 hidden_size: int,
                 dropout: float,
                 window: int,
                 batch_size: int,
                 step: int, transformer,
                 shuffle, models_path,
                 patience: int = 4,
                 cache_size: int = 30,
                 **kwargs):
        super().__init__(window,
                         batch_size,
                         step,
                         transformer,
                         shuffle,
                         models_path,
                         patience=patience,
                         cache_size=cache_size,
                         **kwargs)
        self.hidden_size = hidden_size

        self.dropout = dropout

    def compile(self):
        self.compiled = True
        self.model.compile(
            loss=self.loss,
            optimizer=optimizers.Adam(lr=self.learning_rate),
            metrics=self.metrics,
            loss_weights={'rul': 1, 'signal': 1})

    def _generate_keras_batcher(self, train_batcher, val_batcher):
        n_features = self.transformer.n_features

        def gen_train():
            for X, y in train_batcher:
                yield X, {'signal': X, 'rul': y}

        def gen_val():
            for X, y in val_batcher:
                yield X, {'signal': X, 'rul': y}

        a = tf.data.Dataset.from_generator(
            gen_train,
            (tf.float32, {'signal': tf.float32, 'rul': tf.float32}),
            (
                tf.TensorShape([None, self.window, n_features]),
                {
                    'signal': tf.TensorShape([None, self.window, n_features]),
                    'rul': tf.TensorShape([None, 1])
                }
            )
        )
        b = tf.data.Dataset.from_generator(
            gen_val,
            (tf.float32, {'signal': tf.float32, 'rul': tf.float32}),
            (
                tf.TensorShape([None, self.window, n_features]),
                {
                    'signal': tf.TensorShape([None, self.window, n_features]),
                    'rul': tf.TensorShape([None, 1])
                }
            )
        )
        return a, b

    def build_model(self):
        n_features = self.transformer.n_features
        input = Input(shape=(self.window, n_features))
        x = input

        encoder = TCN(self.hidden_size, kernel_size=5, use_batch_norm=True, use_skip_connections=True,
                      dropout_rate=self.dropout, return_sequences=True, dilations=(1, 2, 4))(x)
        encoder = AveragePooling1D(2)(encoder)
        decoder = UpSampling1D(2)(encoder)
        decoder = Conv1D(n_features, kernel_size=5, padding='same',
                         activation='relu', name='signal')(decoder)

        encoder_last_state = Flatten()(encoder)
        decoder_last_state = Lambda(lambda X: X[:, -1, :])(decoder)

        output = Concatenate()([encoder_last_state, decoder_last_state])
        output = Dense(100, activation='relu')(output)
        output = Dropout(self.dropout)(output)
        output = Dense(50, activation='relu')(output)
        output = Dropout(self.dropout)(output)
        output = Dense(1, name='rul')(output)

        model = Model(
            inputs=[input],
            outputs={'signal': decoder, 'rul': output},
        )

        return model

    @property
    def name(self):
        return "ConvolutionalEncoderDecoder"
