from typing import Tuple
from rul_pm.models.keras.layers import ExpandDimension
from tensorflow.keras import Input, Model, optimizers
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten


class XiangQiangJianQiaoModel(tf.keras.Model):

    """
        Model presented in Remaining useful life estimation in prognostics using deep convolution neural networks

        Deafult parameters reported in the article
        Number of filters:	10
        Window size:	30/20/30/15
        Filter length: 10

        Neurons in fully-connected layer	100
        Dropout rate	0.5
        batch_size = 512


        Parameters
        -----------
        n_filters : int

        filter_size : int


    """

    def __init__(self,
                 n_filters: int,
                 filter_size: int,
                 dropout: float,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.dropout = dropout

        

    def call(self, x):
        pass

    def build_model(self, input_shape):
        input = Input(shape=input_shape)
        x = input

        x = ExpandDimension()(x)
        x = Conv2D(self.n_filters, (self.filter_size, 1),
                   padding='same', activation='tanh',
                   )(x)

        x = Conv2D(self.n_filters, (self.filter_size, 1),
                   padding='same', activation='tanh',
                   )(x)

        x = Conv2D(self.n_filters, (self.filter_size, 1),
                   padding='same', activation='tanh',
                   )(x)

        x = Conv2D(self.n_filters, (self.filter_size, 1),
                   padding='same', activation='tanh')(x)

        x = Conv2D(1, (3, 1), padding='same', activation='tanh')(x)

        x = Flatten()(x)
        x = Dropout(self.dropout)(x)
        x = Dense(100,
                  activation='tanh')(x)
        output = Dense(
            1,
            activation='linear')(x)
        model = Model(
            inputs=[input],
            outputs=[output],
        )
        return model

    def model(self, input_shape: Tuple[int]):
        x = Input(shape=input_shape)
        return Model(inputs=[x], outputs=self.call(x))

