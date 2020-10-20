from sklearn.base import BaseEstimator, TransformerMixin

class PandasToNumpy(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.values


class TargetIdentity(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if len(X.shape) > 1:            
            return X.iloc[:, -1].values            
        else:
            return X.values