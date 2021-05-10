from rul_pm.models.keras.keras import KerasTrainableModel
from rul_pm.models.keras.layers import ExpandDimension
from tensorflow.keras import Input, Model, optimizers
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten


class XiangQiangJianQiaoModel(KerasTrainableModel):

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

    def get_params(self, deep):
        params = super().get_params(deep=deep)
        params.update(
            {
                'n_filters': self.n_filters,
                'filter_size': self.filter_size,
                'dropout': self.dropout
            }
        )
        return params

    @property
    def name(self):
        return "XiangQiangJianQiaoModel"
