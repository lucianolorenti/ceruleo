from typing import Optional

import numpy as np
from rul_pm.iterators.iterators import WindowedDatasetIterator
from rul_pm.models.model import TrainableModel
from xgboost import XGBRegressor, XGBClassifier



class XGBoostModel(TrainableModel):
    """XGBoost model for regression

    Keyword Parameters
    ------------------
    Parameters to construct the xgboost model
    """
    def __init__(self, model_class, **kwargs):
        super().__init__()
        self._model = model_class(**kwargs)

    def fit(self,
            train_windowed_iterator: WindowedDatasetIterator,
            val_windowed_iterator: Optional[WindowedDatasetIterator] = None,
            **kwargs):
        """Fit the model with the given dataset iterator

        Parameters
        ----------
        train_windowed_iterator : WindowedDatasetIterator
            Train dataset iterator
        val_windowed_iterator : Optional[WindowedDatasetIterator], optional
            Validation dataset iterator, by default None

        Returns
        -------
        XGBoostModel
            self
        """
        X_train, y_train, sample_weight_train = train_windowed_iterator.get_data()

        params = {}
        if val_windowed_iterator is not None:
            X_val, y_val, _ = val_windowed_iterator.get_data()
            params.update({'eval_set': [(X_val, y_val)]})
        self._model.fit(X_train,
                      y_train,
                      sample_weight=sample_weight_train,
                      **params,
                      **kwargs)
        return self

    def build_model(self)->XGBRegressor:
        """Return the XGBRegressor model

        Returns
        -------
        XGBRegressor
        """
        return self._model

    def predict(self, dataset_iterator: WindowedDatasetIterator) -> np.array:
        """Predict given the dataset iterator

        Parameters
        ----------
        dataset_iterator : WindowedDatasetIterator
            Dataset iterator from which obtain data to predict

        Returns
        -------
        np.array
            Value predictions
        """
        X, _, _ = dataset_iterator.get_data()
        return self._model.predict(X)

    def feature_importances(self):
        """Obtain the feature importance of the fitted model
        """
        return self._model.feature_importances_.reshape(
            (self.window, self.n_features))

    def get_params(self, deep):
        out = super().get_params(deep=deep)
        out['model'] = self._model
        if deep and hasattr(self._model, 'get_params'):
            for key, value in self._model.get_params(deep=True).items():
                out['model__%s' % key] = value
        return out

    def set_params(self, **params):
        model_params = {}
        for name, value in params.items():
            if '__' in name:
                model_params[name.split('__')[1]] = value
        for name in model_params.keys():
            params.pop(f'model__{name}')

        super().set_params(**params)
        self._model.set_params(**model_params)
        return self


class XGBoostModelRegressor(XGBoostModel):
    def __init__(self, **kwargs):
        super().__init__(XGBRegressor, **kwargs)


class XGBoostModelClassifier(XGBoostModel):
    def __init__(self, **kwargs):
        super().__init__(XGBClassifier, **kwargs)