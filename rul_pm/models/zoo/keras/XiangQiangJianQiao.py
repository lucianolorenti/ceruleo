from typing import List, Optional, Tuple
from rul_pm.models.keras.layers import ExpandDimension
from tensorflow.keras import Input, Model, optimizers
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten
import tensorflow as tf


def XiangQiangJianQiaoModel(
    input_shape: Tuple[int, int],
    convolutional_layers: List[Tuple[int, int]] = [(10, 10), (10, 10), (10, 10)],
    dropout: float = 0.5,
    dense_dimension: int = 100,
    convolutional_activation: str = "tanh",
) -> tf.keras.Model:
    """Model presented in "Remaining useful life estimation in prognostics using deep convolution neural networks"


    Xiang Li, Qian Ding, Jian-Qiao Sun


    Deafult parameters reported in the article
    ------------------------------------------

    Number of filters:	10
    Window size:	30/20/30/15
    Filter length: 10

    Neurons in fully-connected layer:	100
    Dropout rate:	0.5
    batch_size: 512

    Parameters
    ----------
    input_shape : Tuple[int, int]
        [description]
    convolutional_layers : List[Tuple[int, int]], optional
        List of tuples with the convolutional dimensions, by default [(10, 10), (10, 10), (10, 10)]
        Each element of the list is a tuple that contains
        (number of filter, kernel size)

    dropout : float, optional
        Dropout rate, by default 0.5
    dense_dimension : int, optional
        Dimension of the fully connected layer, by default 100
    convolutional_activation : str, optional
        Activation of the convolutional layers, by default "tanh"

    Returns
    -------
    tf.keras.Model
        
    """

    input = Input(shape=input_shape)
    x = input

    x = ExpandDimension()(x)
    for n_filters, filter_size in convolutional_layers:
        x = Conv2D(
            n_filters,
            (filter_size, 1),
            padding="same",
            activation=convolutional_activation,
        )(x)

    x = Conv2D(1, (3, 1), padding="same", activation=convolutional_activation)(x)

    x = Flatten()(x)
    x = Dropout(dropout)(x)
    x = Dense(dense_dimension, activation=convolutional_activation)(x)
    output = Dense(1, activation="relu")(x)
    model = Model(
        inputs=[input],
        outputs=[output],
    )
    return model
