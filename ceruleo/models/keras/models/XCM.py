from typing import Tuple
import numpy as np
import tensorflow as tf
from ceruleo.models.keras.dataset import KerasTrainableModel
from ceruleo.models.keras.layers import ExpandDimension, RemoveDimension
from tensorflow.keras import Input, Model, optimizers
from tensorflow.keras.layers import (Activation, BatchNormalization,
                                     Concatenate, Conv1D, Conv2D, Dense, GlobalAveragePooling1D)


class XCM(KerasTrainableModel):

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
                 n_filters:int=128,
                 filter_window:int=7,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.filter_window = filter_window


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

    def build_model(self, input_shape:Tuple[int, int]):
        input = Input(input_shape)
        x = input
        
        x2d = ExpandDimension()(x)
        x2d = Conv2D(self.n_filters, (input_shape[0], 1),
                   padding='same',
                   name='first_conv2d'
                   )(x2d)
       
        
        
        x2d = BatchNormalization()(x2d)
        
        x2d = Activation('relu')(x2d)
        self.model_fisrt_conv2d = Model(
            inputs=[input],
            outputs=[x2d],
        )
        x2d_input = Input(shape=x2d.shape[1:])


        x2d = Conv2D(1, (1, 1),
                   padding='same',
                   activation='relu',
                   )(x2d_input)
        
        x2d = RemoveDimension(3)(x2d)

        input = Input(shape=(input_shape[0], input_shape[1]))
        x = input
        x1d =  Conv1D(self.n_filters, self.filter_window,
                   padding='same',
                   )(x)
        
        x1d = BatchNormalization()(x1d)
        
        x1d = Activation('relu')(x1d)
        self.model_fisrt_conv1d = Model(
            inputs=[input],
            outputs=[x1d],
        )       
        x1d_input = Input(shape=x1d.shape[1:])

        x1d = Conv1D(1, 1,
                   padding='same', activation='relu',
                   )(x1d_input)

        x = Concatenate()([x2d, x1d])

        x = Conv1D(self.n_filters, self.filter_window, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = GlobalAveragePooling1D()(x)

        x = Dense(self.n_filters,
                  activation='relu')(x)
        output = Dense(
            1,
            activation='relu',
            name='RUL_regressor')(x)

        self.model_regression = Model(
            inputs=[x2d_input, x1d_input],
            outputs=[output]
        )
        model_input = Input(shape=(input_shape[0], input_shape[1]))
        conv2d = self.model_fisrt_conv2d(model_input)
        conv1d = self.model_fisrt_conv1d(model_input)
        output = self.model_regression([conv2d, conv1d])
        
        model = Model(
            inputs=[model_input],
            outputs=[output],
        )
        return model

    def get_params(self, deep=False):
        d = super().get_params()
        d.update( {
              'n_filters': self.n_filters,
                 'filter_window': self.filter_window
        })
        return d

    @property
    def name(self):
        return "XCM"


    def explain(self, input):

        data_input = np.expand_dims(input, axis=0)
        with tf.GradientTape() as tape:
            first_conv1d_layer_output = self.model_fisrt_conv1d(data_input)
            first_conv2d_layer_output = self.model_fisrt_conv2d(data_input)
            tape.watch(first_conv2d_layer_output)
            

            output = self.model_regression([first_conv2d_layer_output,first_conv1d_layer_output])#
            grads = tape.gradient(output, first_conv2d_layer_output)
            filter_weight = np.mean(grads, axis=(0,1,2))


        mmap = np.sum(np.squeeze(first_conv2d_layer_output) * filter_weight, axis=(2))


        with tf.GradientTape() as tape:
            first_conv2d_layer_output = self.model_fisrt_conv2d(data_input)
            first_conv1d_layer_output = self.model_fisrt_conv1d(data_input)
            tape.watch(first_conv1d_layer_output)
            
            output = self.model_regression([first_conv2d_layer_output,first_conv1d_layer_output])


            
            
        grads1d = tape.gradient(output, first_conv1d_layer_output)
        filter1d_weight = np.mean(grads1d, axis=(0,1))
        v = np.sum(np.squeeze(first_conv1d_layer_output) * filter1d_weight, axis=1)
        return (mmap, v)
