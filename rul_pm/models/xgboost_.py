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
                 **kwargs):
        super().__init__(window,
                         step,
                         transformer,
                         cache_size=cache_size,
                         sample_weight=sample_weight)
        self.window = window
        self.transformer = transformer
        self.step = step
        self.xgbr = XGBRegressor(**kwargs)

    def fit(self, train_dataset):
        X, y, sample_weight = self.get_data(train_dataset)
        self.xgbr.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, dataset):
        X, _, _ = self.get_data(dataset, shuffle=False)
        return self.xgbr.predict(X)

    def feature_importances(self):
        return self.xgbr.feature_importances_.reshape((self.window, self.n_features))
