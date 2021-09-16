import numpy as np
import tensorflow as tf

from tensorflow.python.keras.layers  import (BatchNormalization, Concatenate, MaxPool1D, Activation)
from rul_pm.models.keras.keras import KerasTrainableModel
from rul_pm.models.keras.layers import ExpandDimension, RemoveDimension
from tensorflow.keras import Input, Model, optimizers
from tensorflow.keras.layers import (
    Layer,
    LayerNormalization,
    MultiHeadAttention,
    Add,
    Conv1D,
    Conv2D,
    Dense,
    Embedding,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
)


class InceptionTime(KerasTrainableModel):
    def __init__(
        self,
        nb_filters=32,
        use_residual=True,
        use_bottleneck=True,
        depth=6,
        kernel_size=41,
        bottleneck_size:int = 32,
        inception_number:int = 3,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.bottleneck_size = bottleneck_size
        self.inception_number = inception_number

    def compile(self):
        self.compiled = True
        self.model.compile(
            loss=self.loss,
            optimizer=optimizers.Adam(
                lr=self.learning_rate,
                beta_1=0.85,
                beta_2=0.9,
                epsilon=0.001,
                amsgrad=True,
            ),
            metrics=self.metrics,
        )

    def _inception_module(self, input_tensor, stride=1, activation="linear"):
        print(self.bottleneck_size)
        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = Conv1D(
                filters=self.bottleneck_size,
                kernel_size=1,
                padding="same",
                activation=activation,
                use_bias=False,
            )(input_tensor)
        else:
            input_inception = input_tensor

  
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(self.inception_number)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(
                Conv1D(
                    filters=self.nb_filters,
                    kernel_size=kernel_size_s[i],
                    strides=stride,
                    padding="same",
                    activation=activation,
                    use_bias=False,
                )(input_inception)
            )

        max_pool_1 = MaxPool1D(pool_size=3, strides=stride, padding="same")(
            input_tensor
        )

        conv_6 = Conv1D(
            filters=self.nb_filters,
            kernel_size=1,
            padding="same",
            activation=activation,
            use_bias=False,
        )(max_pool_1)

        conv_list.append(conv_6)

        x = Concatenate(axis=2)(conv_list)
        x = BatchNormalization()(x)
        x = Activation(activation="relu")(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = Conv1D(
            filters=int(out_tensor.shape[-1]),
            kernel_size=1,
            padding="same",
            use_bias=False,
        )(input_tensor)
        shortcut_y = BatchNormalization()(shortcut_y)

        x = Add()([shortcut_y, out_tensor])
        x = Activation("relu")(x)
        return x

    def build_model(self, input_shape):


        input = Input(shape=input_shape)

        x = input
        input_res = input

        for d in range(self.depth):

            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = GlobalAveragePooling1D()(x)

        output_layer = Dense(1, activation="relu")(gap_layer)

        model = Model(inputs=input, outputs=output_layer)

        return model

    def get_params(self, deep=False):
        d = super().get_params()
        d["nb_filters"] = self.nb_filters
        d["use_residual"] = self.use_residual
        d["use_bottleneck"] = self.use_bottleneck
        d["depth"] = self.depth
        d["kernel_size"] = self.kernel_size
        d["bottleneck_size"] = self.bottleneck_size
        return d

    @property
    def name(self):
        return "InceptionTime"
