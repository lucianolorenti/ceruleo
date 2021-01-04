from rul_pm.iterators.iterators import WindowedDatasetIterator
from xgboost import XGBRegressor


class XGBoostModel:
    def __init__(self, window: int, step: int,  transformer, **kwargs):
        self.window = window
        self.transformer = transformer
        self.step = step
        self.xgbr = XGBRegressor(**kwargs)

    def _transform(self, ds, fit=False):
        X, y = WindowedDatasetIterator(
            ds, self.window,
            self.transformer, step=self.step).toArray()
        if fit:
            self.n_features = X.shape[2]
            self.window = X.shape[1]
        X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
        return X, y

    def fit(self, train_dataset):

        X, y = self._transform(train_dataset, fit=True)
        self.xgbr.fit(X, y)
        return self

    def predict(self, dataset):
        return self.xgbr.predict(self._transform(dataset)[0])

    def feature_importances(self):
        return self.xgbr.feature_importances_.reshape((self.window, self.n_features))
