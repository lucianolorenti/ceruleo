
class KerasTrainableModel(TrainableModel):
    def __init__(self,
                 window,
                 batch_size,
                 step,
                 transformer,
                 shuffle,
                 models_path,
                 patience=4,
                 cache_size=30):
        super().__init__(window,
                         batch_size,
                         step,
                         transformer,
                         shuffle,
                         models_path,
                         patience=patience,
                         cache_size=cache_size)

    def load_best_model(self):
        self._model.load_weights(self.model_filepath)

    def _checkpoint_callback(self):
        return ModelCheckpoint(filepath=self.model_filepath,
                               save_weights_only=True,
                               monitor='val_loss',
                               mode='min',
                               save_best_only=True)

    def _results(self):
        params = super()._results()
        params.update({
            'best_val_loss': np.min(self.history.history['val_loss']),
            'val_loss': self.history.history['val_loss'],
            'train_loss': self.history.history['loss'],
        })
        return params

    def predict(self, dataset, step=None):
        step = self.step if step is None else step
        n_features = self.transformer.n_features
        batcher = get_batcher(dataset,
                              self.window,
                              512,
                              self.transformer,
                              step,
                              shuffle=False,
                              cache_size=self.cache_size)
        batcher.restart_at_end = False

        def gen_dataset():
            for X, y in batcher:
                yield X

        b = tf.data.Dataset.from_generator(
            gen_dataset, (tf.float32),
            (tf.TensorShape([None, self.window, n_features])))
        return super().predict(b)

    def input_shape(self):
        n_features = self.transformer.n_features
        return (self.window, n_features)

    def fit(self, train_dataset, validation_dataset, verbose=1, epochs=50):
        #if Path(self.results_filename).resolve().is_file():
        #    logger.info(f'Results already present {self.results_filename}')
        #    return None
        if not self.transformer.fitted_:
            self.transformer.fit(train_dataset)
        logger.info('Creating batchers')
        train_batcher = get_batcher(train_dataset,
                                    self.window,
                                    self.batch_size,
                                    self.transformer,
                                    self.step,
                                    shuffle=self.shuffle,
                                    cache_size=self.cache_size)
        val_batcher = get_batcher(validation_dataset,
                                  self.window,
                                  self.batch_size,
                                  self.transformer,
                                  self.step,
                                  shuffle=False,
                                  cache_size=self.cache_size)
        val_batcher.restart_at_end = False

        early_stopping = EarlyStopping(patience=self.patience)
        model_checkpoint_callback = self._checkpoint_callback()

        n_features = self.transformer.n_features

        def gen_train():
            for X, y in train_batcher:
                yield X, y

        def gen_val():
            for X, y in val_batcher:
                yield X, y

        a = tf.data.Dataset.from_generator(
            gen_train, (tf.float32, tf.float32), (tf.TensorShape(
                [None, self.window, n_features]), tf.TensorShape([None])))
        b = tf.data.Dataset.from_generator(
            gen_val, (tf.float32, tf.float32), (tf.TensorShape(
                [None, self.window, n_features]), tf.TensorShape([None])))

        logger.info('Start fitting')
        logger.info(self.model_filepath)
        self.history = self._model.fit(
            a,
            verbose=verbose,
            steps_per_epoch=len(train_batcher),
            epochs=epochs,
            validation_data=b,
            validation_steps=len(val_batcher),
            callbacks=[
                early_stopping,
                # lr_callback,
                TerminateOnNaN(train_batcher),
                model_checkpoint_callback
            ])

        self.save_results()
        return self.load_results()


class FCN(KerasTrainableModel):
    def __init__(self,
                 layers,
                 dropout,
                 l2,
                 window,
                 batch_size,
                 step,
                 transformer,
                 shuffle,
                 models_path,
                 patience=7):
        super(FCN, self).__init__(window,
                                  batch_size,
                                  step,
                                  transformer,
                                  shuffle,
                                  models_path,
                                  patience=patience)
        self.layers_ = []
        self.layers_sizes_ = layers
        self.dropout = dropout
        self.l2 = l2

    def model(self):
        s = Sequential()
        s.add(Flatten())
        for l in self.layers_sizes_:
            s.add(
                Dense(l,
                      activation='relu',
                      kernel_regularizer=regularizers.l2(self.l2)))
            s.add(Dropout(self.dropout))
            s.add(BatchNormalization())
        s.add(Dense(1, activation='relu'))
        return s

    @property
    def name(self):
        return 'FCN'

    def parameters(self):
        params = super().parameters()
        params.update({
            'dropout': self.dropout,
            'l2': self.l2,
            'layers': self.layers_sizes_
        })
        return params


class ConvolutionalSimple(KerasTrainableModel):
    """
    The network contains stacked layers of 1-dimensional convolutional layers
    followed by max poolings

    Parameters
    ----------
    self: list of tuples (filters: int, kernel_size: int)
          Each element of the list is a layer of the network. The first element of the tuple contaings
          the number of filters, the second one, the kernel size.
    """
    def __init__(self,
                 layers,
                 dropout,
                 l2,
                 window,
                 batch_size,
                 step,
                 transformer,
                 shuffle,
                 models_path,
                 patience=7):
        super(ConvolutionalSimple, self).__init__(window,
                                                  batch_size,
                                                  step,
                                                  transformer,
                                                  shuffle,
                                                  models_path,
                                                  patience=patience)
        self.layers_ = []
        self.layers_sizes_ = layers
        self.dropout = dropout
        self.l2 = l2

    def model(self):
        s = Sequential()
        for filters, kernel_size in self.layers_sizes_:
            s.add(
                Conv1D(filters=filters,
                       strides=1,
                       kernel_size=kernel_size,
                       padding='same',
                       activation='relu'))
            s.add(MaxPool1D(pool_size=2, strides=2))
        s.add(Flatten())
        s.add(Dense(50, activation='relu'))
        s.add(Dropout(self.dropout))
        s.add(BatchNormalization())
        s.add(Dense(1, activation='relu'))
        return s

    @property
    def name(self):
        return 'ConvolutionalSimple'

    def parameters(self):
        params = super().parameters()
        params.update({
            'dropout': self.dropout,
            'l2': self.l2,
            'layers': self.layers_sizes_
        })
        return params