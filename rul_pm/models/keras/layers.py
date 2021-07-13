
import typing
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
from tensorflow.python.framework import tensor_shape


def ExpandDimension():
    return Lambda(lambda x: K.expand_dims(x))



def RemoveDimension(axis=0):
    return Lambda(lambda x: K.squeeze(x, axis=axis))


class MultiHeadAttention(tf.keras.layers.Layer):
    r"""MultiHead Attention layer.
    Defines the MultiHead Attention operation as described in
    [Attention Is All You Need](https://arxiv.org/abs/1706.03762) which takes
    in the tensors `query`, `key`, and `value`, and returns the dot-product attention
    between them:
    >>> mha = MultiHeadAttention(head_size=128, num_heads=12)
    >>> query = np.random.rand(3, 5, 4) # (batch_size, query_elements, query_depth)
    >>> key = np.random.rand(3, 6, 5) # (batch_size, key_elements, key_depth)
    >>> value = np.random.rand(3, 6, 6) # (batch_size, key_elements, value_depth)
    >>> attention = mha([query, key, value]) # (batch_size, query_elements, value_depth)
    >>> attention.shape
    TensorShape([3, 5, 6])
    If `value` is not given then internally `value = key` will be used:
    >>> mha = MultiHeadAttention(head_size=128, num_heads=12)
    >>> query = np.random.rand(3, 5, 5) # (batch_size, query_elements, query_depth)
    >>> key = np.random.rand(3, 6, 10) # (batch_size, key_elements, key_depth)
    >>> attention = mha([query, key]) # (batch_size, query_elements, key_depth)
    >>> attention.shape
    TensorShape([3, 5, 10])
    Arguments:
        head_size: int, dimensionality of the `query`, `key` and `value` tensors
            after the linear transformation.
        num_heads: int, number of attention heads.
        output_size: int, dimensionality of the output space, if `None` then the
            input dimension of `value` or `key` will be used,
            default `None`.
        dropout: float, `rate` parameter for the dropout layer that is
            applied to attention after softmax,
        default `0`.
        use_projection_bias: bool, whether to use a bias term after the linear
            output projection.
        return_attn_coef: bool, if `True`, return the attention coefficients as
            an additional output argument.
        kernel_initializer: initializer, initializer for the kernel weights.
        kernel_regularizer: regularizer, regularizer for the kernel weights.
        kernel_constraint: constraint, constraint for the kernel weights.
        bias_initializer: initializer, initializer for the bias weights.
        bias_regularizer: regularizer, regularizer for the bias weights.
        bias_constraint: constraint, constraint for the bias weights.
    Call Arguments:
        inputs:  List of `[query, key, value]` where
            * `query`: Tensor of shape `(..., query_elements, query_depth)`
            * `key`: `Tensor of shape '(..., key_elements, key_depth)`
            * `value`: Tensor of shape `(..., key_elements, value_depth)`, optional, if not given `key` will be used.
        mask: a binary Tensor of shape `[batch_size?, num_heads?, query_elements, key_elements]`
        which specifies which query elements can attendo to which key elements,
        `1` indicates attention and `0` indicates no attention.
    Output shape:
        * `(..., query_elements, output_size)` if `output_size` is given, else
        * `(..., query_elements, value_depth)` if `value` is given, else
        * `(..., query_elements, key_depth)`
    """

    def __init__(
        self,
        head_size: int,
        num_heads: int,
        output_size: int = None,
        dropout: float = 0.0,
        use_projection_bias: bool = True,
        return_attn_coef: bool = False,
        kernel_initializer: typing.Union[str,
                                         typing.Callable] = "glorot_uniform",
        kernel_regularizer: typing.Union[str, typing.Callable] = None,
        kernel_constraint: typing.Union[str, typing.Callable] = None,
        bias_initializer: typing.Union[str, typing.Callable] = "zeros",
        bias_regularizer: typing.Union[str, typing.Callable] = None,
        bias_constraint: typing.Union[str, typing.Callable] = None,
        **kwargs
    ):
        warnings.warn(
            "`MultiHeadAttention` will be deprecated in Addons 0.13. "
            "Please use `tf.keras.layers.MultiHeadAttention` instead.",
            DeprecationWarning,
        )

        super().__init__(**kwargs)

        if output_size is not None and output_size < 1:
            raise ValueError("output_size must be a positive number")

        self.head_size = head_size
        self.num_heads = num_heads
        self.output_size = output_size
        self.use_projection_bias = use_projection_bias
        self.return_attn_coef = return_attn_coef

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.dropout = tf.keras.layers.Dropout(dropout)
        self._droput_rate = dropout

    def build(self, input_shape):

        num_query_features = input_shape[0][-1]
        num_key_features = input_shape[1][-1]
        num_value_features = (
            input_shape[2][-1] if len(input_shape) > 2 else num_key_features
        )
        output_size = (
            self.output_size if self.output_size is not None else num_value_features
        )

        self.query_kernel = self.add_weight(
            name="query_kernel",
            shape=[self.num_heads, num_query_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.key_kernel = self.add_weight(
            name="key_kernel",
            shape=[self.num_heads, num_key_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.value_kernel = self.add_weight(
            name="value_kernel",
            shape=[self.num_heads, num_value_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.projection_kernel = self.add_weight(
            name="projection_kernel",
            shape=[self.num_heads, self.head_size, output_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        if self.use_projection_bias:
            self.projection_bias = self.add_weight(
                name="projection_bias",
                shape=[output_size],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.projection_bias = None

        super().build(input_shape)

    def call(self, inputs, training=None, mask=None):

        # einsum nomenclature
        # ------------------------
        # N = query elements
        # M = key/value elements
        # H = heads
        # I = input features
        # O = output features

        query = inputs[0]
        key = inputs[1]
        value = inputs[2] if len(inputs) > 2 else key

        # verify shapes
        if key.shape[-2] != value.shape[-2]:
            raise ValueError(
                "the number of elements in 'key' must be equal to the same as the number of elements in 'value'"
            )

        if mask is not None:
            if len(mask.shape) < 2:
                raise ValueError("'mask' must have atleast 2 dimensions")
            if query.shape[-2] != mask.shape[-2]:
                raise ValueError(
                    "mask's second to last dimension must be equal to the number of elements in 'query'"
                )
            if key.shape[-2] != mask.shape[-1]:
                raise ValueError(
                    "mask's last dimension must be equal to the number of elements in 'key'"
                )

        # Linear transformations
        query = tf.einsum("...NI , HIO -> ...NHO", query, self.query_kernel)
        key = tf.einsum("...MI , HIO -> ...MHO", key, self.key_kernel)
        value = tf.einsum("...MI , HIO -> ...MHO", value, self.value_kernel)

        # Scale dot-product, doing the division to either query or key
        # instead of their product saves some computation
        depth = tf.constant(self.head_size, dtype=tf.float32)
        query /= tf.sqrt(depth)

        # Calculate dot product attention
        logits = tf.einsum("...NHO,...MHO->...HNM", query, key)

        # apply mask
        if mask is not None:
            mask = tf.cast(mask, tf.float32)

            # possibly expand on the head dimension so broadcasting works
            if len(mask.shape) != len(logits.shape):
                mask = tf.expand_dims(mask, -3)

            logits += -10e9 * (1.0 - mask)

        attn_coef = tf.nn.softmax(logits)

        # attention dropout
        attn_coef_dropout = self.dropout(attn_coef, training=training)

        # attention * value
        multihead_output = tf.einsum(
            "...HNM,...MHI->...NHI", attn_coef_dropout, value)

        # Run the outputs through another linear projection layer. Recombining heads
        # is automatically done.
        output = tf.einsum(
            "...NHI,HIO->...NO", multihead_output, self.projection_kernel
        )

        if self.projection_bias is not None:
            output += self.projection_bias

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output

    def compute_output_shape(self, input_shape):
        num_value_features = (
            input_shape[2][-1] if len(input_shape) > 2 else input_shape[1][-1]
        )
        output_size = (
            self.output_size if self.output_size is not None else num_value_features
        )

        output_shape = input_shape[0][:-1] + (output_size,)

        if self.return_attn_coef:
            num_query_elements = input_shape[0][-2]
            num_key_elements = input_shape[1][-2]
            attn_coef_shape = input_shape[0][:-2] + (
                self.num_heads,
                num_query_elements,
                num_key_elements,
            )

            return output_shape, attn_coef_shape
        else:
            return output_shape

    def get_config(self):
        config = super().get_config()

        config.update(
            head_size=self.head_size,
            num_heads=self.num_heads,
            output_size=self.output_size,
            dropout=self._droput_rate,
            use_projection_bias=self.use_projection_bias,
            return_attn_coef=self.return_attn_coef,
            kernel_initializer=tf.keras.initializers.serialize(
                self.kernel_initializer),
            kernel_regularizer=tf.keras.regularizers.serialize(
                self.kernel_regularizer),
            kernel_constraint=tf.keras.constraints.serialize(
                self.kernel_constraint),
            bias_initializer=tf.keras.initializers.serialize(
                self.bias_initializer),
            bias_regularizer=tf.keras.regularizers.serialize(
                self.bias_regularizer),
            bias_constraint=tf.keras.constraints.serialize(
                self.bias_constraint),
        )

        return config


class ConcreteDropout(tf.keras.layers.Layer):
    """Concrete Dropout layer class from https://arxiv.org/abs/1705.07832.
    Dropout Feature Ranking for Deep Learning Models
    Chun-Hao Chang
    Ladislav Rampasek
    Anna Goldenberg
    Arguments:
        dropout_regularizer:
            Positive float, satisfying $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and
            N the number of instances in the dataset.
            The factor of two should be ignored for cross-entropy loss,
            and used only for the eucledian loss.
        init_min:
            Minimum value for the randomly initialized dropout rate, in [0, 1].
        init_min:
            Maximum value for the randomly initialized dropout rate, in [0, 1],
            with init_min <= init_max.
        name:
            String, name of the layer.

    """

    def __init__(self, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.9, name=None,
                 training=True, **kwargs):

        super(ConcreteDropout, self).__init__(name=name,
                                              **kwargs)
        assert init_min <= init_max, \
            'init_min must be lower or equal to init_max.'

        self.dropout_regularizer = dropout_regularizer

        self.p_logit = None
        self.p = None
        self.init_min = (np.log(init_min) - np.log(1. - init_min))
        self.init_max = (np.log(init_max) - np.log(1. - init_max))
        self.training = training

    def build(self, input_shape):
        self.window = input_shape[-2]
        self.number_of_features = input_shape[-1]
        input_shape = tensor_shape.TensorShape(input_shape)

        self.p_logit = self.add_weight(name='p_logit',
                                       shape=[self.number_of_features],
                                       initializer=tf.random_uniform_initializer(
                                           self.init_min,
                                           self.init_max),
                                       dtype=tf.float32,
                                       trainable=True)

    def concrete_dropout(self, p, x):
        eps = K.cast_to_floatx(K.epsilon())
        temp = 1.0 / 10.0
        unif_noise = K.random_uniform(shape=[self.number_of_features])
        drop_prob = (
            K.log(p + eps)
            - K.log(1. - p + eps)
            + K.log(unif_noise + eps)
            - K.log(1. - unif_noise + eps)
        )
        drop_prob = K.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - p
        x *= random_tensor
        x /= retain_prob
        return x

    def call(self, inputs, training=True):

        p = K.sigmoid(self.p_logit)

        dropout_regularizer = p * K.log(p)
        dropout_regularizer += (1. - p) * K.log(1. - p)
        dropout_regularizer *= self.dropout_regularizer * self.number_of_features
        regularizer = K.sum(dropout_regularizer)
        self.add_loss(regularizer)

        x = self.concrete_dropout(p, inputs)

        return x



class GatedTimeSeries(tf.keras.layers.Layer):
    def __init__(self, regularizer:float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regularizer = regularizer

    def build(self, input_shape):
        self.w = self.add_weight(
            name='gate',
            shape=(1, input_shape[-1]),
            initializer="ones",
            trainable=True,
            regularizer=tf.keras.regularizers.l1(self.regularizer),
        )

    def call(self, inputs):
        activation = 1 - tf.exp(-(self.w * self.w))
        self.add_metric(
            tf.reduce_sum(tf.cast(activation > 0.00001, dtype=tf.float32)),
            name="Number of features",
        )
        return tf.math.multiply(inputs, activation)