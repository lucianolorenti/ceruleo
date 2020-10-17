from sklearn.base import BaseEstimator, TransformerMixin


class ResamplerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, time_feature, time='40s', interpolation_method='linear'):
        self.time = time
        self.interpolation_method = interpolation_method
        self.time_feature = time_feature
        self.enabled = True

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        if self.enabled:
            return (df
                    .drop_duplicates(subset=self.time_feature)
                    .set_index(self.time_feature)
                    .resample(self.time, origin='start')
                    .interpolate(method=self.interpolation_method)
                    .reset_index(drop=True))
        else:
            return df.drop(columns=self.time_feature)
