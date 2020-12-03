from xgboost import XGBRegressor
from rul_pm.iterators.iterators import WindowedDatasetIterator


class XGBoostModel:
    def __init__(self, window: int, step: int,  transformer):
        self.window = window
        self.transformer = transformer
        self.step = step
        self.xgbr = XGBRegressor(nthreads=4)

    def _transform(self, ds):
        X, y = WindowedDatasetIterator(
            ds, self.window, 
            self.transformer, step=self.step).toArray()
        X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
        return X, y
        
    def fit(self, train_dataset):
        X, y = self._transform(train_dataset)        
        self.xgbr.fit(X, y)
        return self

    def predict(self, dataset):
        return self.xgbr.predict(self._transform(dataset)[0])

    def feature_importances(self):
        return self.xgbr.feature_importances_
