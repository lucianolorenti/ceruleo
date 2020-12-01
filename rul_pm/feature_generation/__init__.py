import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


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
        X = X.copy()
        for c in self.columns:
            X.loc[:, f'{c}_cumsum'] = 0
        for life in X[self.life_id_col].unique():
            data = X[X[self.life_id_col] == life]
            data_columns = data[self.columns]
            mask = (
                (data_columns < (self.LCL)) |
                (data_columns > (self.UCL))
            )

            df_cumsum = mask.astype('int').cumsum()
            for c in self.columns:
                X.loc[X[self.life_id_col] == life,
                      f'{c}_cumsum'] = df_cumsum[c]

        return X
