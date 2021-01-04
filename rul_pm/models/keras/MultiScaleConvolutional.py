
import tensorflow as tf
from rul_pm.models.keras.keras import KerasTrainableModel
from rul_pm.models.keras.layers import ExpandDimension
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Activation, Add, BatchNormalization,
                                     Conv2D, Dense, Dropout, Flatten)


class MultiScaleConvolutionalModel(KerasTrainableModel):
    """
    Remaining useful life prediction using multi-scale deep convolutional neural network


    Author links open overlay panel
    Han Li
    Wei Zhao
    Yuxi Zhanga
    Enrico Zio
    """

    def __init__(self,  n_msblocks: int, scales: list, n_hidden: int, l2: float = 0.5, dropout: float = 0.5, **kwargs):
        super().__init__(**kwargs)

        self.scales = scales
        self.n_msblocks = n_msblocks
        self.n_hidden = n_hidden
        self.l2 = l2
        self.dropout = dropout

    def _create_scale_conv(self, hidden, scale):
        return Conv2D(hidden,
                      (scale, 1),
                      padding='same',
                      activation='tanh')

    def build_model(self):

        n_features = self.transformer.n_features
        input = Input((self.window, n_features))
        x = input
        x = ExpandDimension()(x)
        for i in range(self.n_msblocks):
            convs = [self._create_scale_conv(self.n_hidden, scale)(x)
                     for scale in self.scales]
            x = Add()(convs)
        x = Conv2D(10, (10, 1), padding='same', activation='tanh')(x)
        x = Conv2D(1, (3, 1), padding='same', activation='tanh')(x)
        x = Flatten()(x)
        x = Dense(
            64, kernel_regularizer=tf.keras.regularizers.l2(self.l2))(x)
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)
        x = Dropout(self.dropout)(x)
        x = Dense(1)(x)
        return Model(inputs=[input], outputs=[x])

    @property
    def name(self):
        return "MultiScaleConvolutional"
