import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ResamplerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, time_feature, time='40s', interpolation_method='linear', enabled=True):
        self.time = time
        self.interpolation_method = interpolation_method
        self.time_feature = time_feature
        self.enabled = enabled

    def fit(self, X, y=None):
        return self

    def transform(self, df):

        if self.enabled:
            X = df.copy()
            X[self.time_feature] = pd.to_timedelta(
                X[self.time_feature], unit='s')
            X = (X
                 .drop_duplicates(subset=self.time_feature)
                 .set_index(self.time_feature)
                 # .resample('5s', origin='start')
                 # .mean()
                 .resample(self.time, origin='start')
                 .interpolate()
                 .reset_index())
            X[self.time_feature] = X[self.time_feature].astype('int')
            return X
        else:
            return df


class SubSampleTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, step: int):
        self.step = step

    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
        return self

    def transform(self, df):
        return df.iloc[range(0, df.shape[0], self.step), :].copy()
