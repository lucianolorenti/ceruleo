from typing import Tuple

from tensorflow.keras.layers import (
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Permute,
    Reshape,
)


def MVCNN(
    input_shape: Tuple[int, int],
    *,
    window = 64,
    dropout: float=0.1,
    
):
    """
    Model presented in Remaining useful life estimation in prognostics using deep convolution neural networks

    Li, X., Ding, Q., & Sun, J. Q. (2018).
    Remaining useful life estimation in prognostics using deep convolution neural networks.
    Reliability Engineering & System Safety, 172, 1-11.

    Deafult parameters reported in the article:

        Number of filters:	10
        Window size:	30/20/30/15
        Filter length: 10

        Neurons in fully-connected layer	100
        Dropout rate	0.5
        batch_size = 512


    Parameters:

        window: Window size of the convolutional windows
        dropout: 
        
    """
    n_features = input_shape[1]
    window = input_shape[0]

    input = Input(shape=input_shape)
    x = input
    x = Permute((2, 1))(x)
    x = Reshape((input_shape[0], input_shape[1], window))(x)

    x = Conv2D(window, (1, 1), activation="relu", padding="same")(x)

    x1 = Conv2D(window, (2, 2), activation="relu", padding="same")(x)
    x1 = Conv2D(window, (2, 2), activation="relu", padding="same")(x1)

    x2 = Conv2D(window, (3, 3), activation="relu", padding="same")(x)
    x2 = Conv2D(window, (3, 3), activation="relu", padding="same")(x2)

    x3 = Conv2D(window, (5, 5), activation="relu", padding="same")(x)
    x3 = Conv2D(window, (5, 5), activation="relu", padding="same")(x3)

    x = Concatenate(axis=1)([x, x1, x2, x3])
    x = Conv2D(window, input_shape)(x)
    x = Flatten()(x)
    x = Dense(100, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(100, activation="relu")(x)
    x = Dropout(dropout)(x)
    output = Dense(1)(x)
    model = Model(
        inputs=[input],
        outputs=[output],
    )
    return model
