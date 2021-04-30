
import numpy as np
import tensorflow as tf
from rul_pm.models.keras.keras import KerasTrainableModel
from rul_pm.models.keras.layers import ExpandDimension, RemoveDimension
from tensorflow.keras import Input, Model, optimizers
from tensorflow.keras.layers import (Layer, LayerNormalization, MultiHeadAttention,
                                     Add, Conv1D, Conv2D, Dense, Embedding,
                                     Dropout, Flatten, GlobalAveragePooling1D)

class Patches(Layer):
    def __init__(self, patch_size, features):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        self.features = features

    def call(self, images):
        batch_size = tf.shape(images)[0]        
   
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.features, 1],
            strides=[1, self.patch_size, self.features, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]      


        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        patch_dims = patches.shape[-1]  

        return patches


class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x


class VisionTransformer(KerasTrainableModel):

    """
        XCM: An Explainable Convolutional Neural Networkfor Multivariate Time Series Classification

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
                 patch_size:int=5,                 
                 projection_dim:int = 64,
                 num_heads:int= 4,
                 transformer_layers:int= 8,
                 mlp_head_units = [2048, 1024],
                 **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.num_patches = (self.window // patch_size) 
        self.projection_dim= projection_dim
        self.num_heads = num_heads
        self.transformer_units = [
               projection_dim * 2,
               projection_dim,
               ]
        self.transformer_layers = transformer_layers
        self.mlp_head_units= mlp_head_units


    def compile(self):
        self.compiled = True
        self.model.compile(
            loss=self.loss,
            optimizer=optimizers.Adam(lr=self.learning_rate,
                                      beta_1=0.85,
                                      beta_2=0.9,
                                      epsilon=0.001,
                                      amsgrad=True),
            metrics=self.metrics)

    def build_model(self):
        n_features = self.transformer.n_features

        input = Input(shape=(self.window, n_features))          
        x = ExpandDimension()(input)
        patches = Patches(self.patch_size, n_features)(x)        
        encoded_patches = PatchEncoder(self.num_patches, self.projection_dim)(patches)

    
        for _ in range(self.transformer_layers):

            x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)

            attention_output = MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1
            )(x1, x1)

            x2 = Add()([attention_output, encoded_patches])

            x3 = LayerNormalization(epsilon=1e-6)(x2)

            x3 = mlp(x3, hidden_units=self.transformer_units, dropout_rate=0.1)

            encoded_patches = Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = Flatten()(representation)
        representation = Dropout(0.5)(representation)

        features = mlp(representation, hidden_units=self.mlp_head_units, dropout_rate=0.5)

        logits = Dense(1, activation='relu')(features)
        # Create the Keras model.
        model = Model(inputs=input, outputs=logits)
        return model

    def get_params(self, deep=False):
        d = super().get_params()
        
        return d

    @property
    def name(self):
        return "XCM"


   