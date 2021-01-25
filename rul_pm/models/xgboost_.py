from rul_pm.iterators.iterators import WindowedDatasetIterator
from rul_pm.models.model import TrainableModel
from xgboost import XGBRegressor


class XGBoostModel(TrainableModel):
    def __init__(self,
                 window: int = 15,
                 step: int = 1,
                 transformer=None,
                 cache_size: int = 15,
                 sample_weight: str = 'equal',
                 shuffle: str = 'all',
                 **kwargs):
        super().__init__(window,
                         step,
                         transformer,
                         cache_size=cache_size,
                         sample_weight=sample_weight,
                         shuffle=shuffle)
        self.window = window
        self.transformer = transformer
        self.step = step
        self.xgbr = XGBRegressor(**kwargs)

    def fit(self, train_dataset, validation_dataset=None, refit_transformer: bool = True, **kwargs):
        if refit_transformer:
            self.transformer.fit(train_dataset)
        X_train, y_train, sample_weight_train = self.get_data(
            train_dataset, shuffle=self.shuffle)

        params = {}
        if validation_dataset is not None:
            X_val, y_val, sample_weight_val = self.get_data(
                validation_dataset, shuffle=None)
            params.update({
                'eval_set': [(X_val, y_val)]

            })
        self.xgbr.fit(X_train, y_train,
                      sample_weight=sample_weight_train, **params, **kwargs)
        return self

    def predict(self, dataset):
        X, _, _ = self.get_data(dataset, shuffle=False)
        return self.xgbr.predict(X)

    def feature_importances(self):
        return self.xgbr.feature_importances_.reshape((self.window, self.n_features))
