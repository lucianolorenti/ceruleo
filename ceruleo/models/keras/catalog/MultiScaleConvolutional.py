from typing import Tuple
import tensorflow as tf

from ceruleo.models.keras.layers import ExpandDimension
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
)


def MultiScaleConvolutionalModel(
    input_shape: Tuple[int, int],
    *,
    n_msblocks: int,
    scales: list,
    n_hidden: int,
    l2: float = 0.5,
    dropout: float = 0.5,
    activation: str = 'relu'
):
    """Remaining useful life prediction using multi-scale deep convolutional neural network


    Li, H., Zhao, W., Zhang, Y., & Zio, E. (2020). 
    Remaining useful life prediction using multi-scale deep convolutional neural network. 
    Applied Soft Computing, 89, 106113.
    
    (Reference)[https://www.sciencedirect.com/science/article/pii/S1568494620300533?casa_token=wp27UPxwVTIAAAAA:e_fMuKsvfQf8VZ7DgsXKCi6mpnrcx2hI0tbfe5xrLRhxmc2vaR-uW_Qq23v5yBqziBoSM5gu]
    
    Parameters:

        input_shape: Input shape of the iterator
        n_msblocks: Number of scale blocks
        scales: List of scales
        n_hidden: Hidden convolutional dimensions
        l2: regularization
        dropout: Dropout factor
        activation
    """

    def _create_scale_conv(hidden, scale):
        return Conv2D(hidden, (scale, 1), padding="same", activation=activation)

    input = Input(input_shape)
    x = input
    x = ExpandDimension()(x)
    for i in range(n_msblocks):
        convs = [
            _create_scale_conv(n_hidden, scale)(x) for scale in scales
        ]
        x = Add()(convs)
    x = Conv2D(10, (10, 1), padding="same", activation=activation)(x)
    x = Conv2D(1, (3, 1), padding="same", activation=activation)(x)
    x = Flatten()(x)
    x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = BatchNormalization()(x)
    x = Activation("tanh")(x)
    x = Dropout(dropout)(x)
    x = Dense(1, activation='relu')(x)
    return Model(inputs=[input], outputs=[x])
