import copy
import logging

import numpy as np
import pandas as pd
from rul_pm.transformation.utils import PandasToNumpy, TargetIdentity
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


class LifeCumSum(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, life_id_col='life'):
        self.life_id_col = life_id_col
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.columns:
            X.loc[:, f'{c}_cumsum'] = 0
        for life in X[self.life_id_col].unique():
            data = X[X[self.life_id_col] == life]
            data_columns = (data[self.columns]
                            .astype('int')
                            .cumsum())
            for c in self.columns:
                X.loc[X[self.life_id_col] == life,
                      f'{c}_cumsum'] = data_columns[c]
        return X


class Diff(BaseEstimator, TransformerMixin):
    def __init__(self, ignored=[]):        
        self.ignored = ignored

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        X_ignored = None
        columns = X.columns
        if len(self.ignored) > 0:
            X_ignored = X[self.ignored] 
            columns = [f for f in X.columns if f not in self.ignored]
        X_new = X[columns].diff().fillna(0)
        X_new.columns = [f'{f}_diff' for f in columns]
        if X_ignored is not None:
            columns = X_new.columns.values.tolist() + X_ignored.columns.values.tolist()
            X_new =  pd.concat((X_new, X_ignored), axis='columns', ignore_index=True)
            X_new.columns = columns
        return X_new

class LifeExceededCumSum(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None,  life_id_col='life', lambda_=0.5):
        self.lambda_ = lambda_
        self.UCL = None
        self.LCL = None
        self.life_id_col = life_id_col
        self.columns = columns

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = set(X.columns.values) - set(['life'])
        mean = np.nanmean(X[self.columns], axis=0)
        s = np.sqrt(self.lambda_ / (2-self.lambda_)) * \
            np.nanstd(X[self.columns], axis=0)
        self.UCL = mean + 3*s
        self.LCL = mean - 3*s
        return self

    def transform(self, X):
        
        
        cnames = [f'{c}_cumsum' for c in self.columns]
        new_X = pd.DataFrame({}, columns=cnames, index=X.index)            
        for life in X[self.life_id_col].unique():
            data = X[X[self.life_id_col] == life]
            data_columns = data[self.columns]
            mask = (
                (data_columns < (self.LCL)) |
                (data_columns > (self.UCL))
            )
            df_cumsum = mask.astype('int').cumsum()
            for c in self.columns:
                new_X.loc[
                    X[self.life_id_col] == life,
                    f'{c}_cumsum'] = df_cumsum[c]

        return new_X


class OneHotCategoricalPandas(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        self.columns = [c for c in X.columns if c in self.features]
        self.enconder = OneHotEncoder(
            handle_unknown='ignore', sparse=False).fit(X[self.columns])
        logger.info(f'Categorical featuers {self.columns}')
        return self

    def transform(self, X, y=None):
        return self.enconder.transform(X[self.columns])


