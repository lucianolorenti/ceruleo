import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler

class IQROutlierRemover(BaseEstimator, TransformerMixin):
    """
    X = np.random.rand(500, 5) * np.random.randn(500, 5) * 15
    imput = IQRImputer(1.5)
    imput.fit(X)
    X_t = imput.transform(X)
    """
    def __init__(self, l=1.5, proportion_to_sample=0.1):
        self.l = l
        self.proportion_to_sammple = proportion_to_sample

    def fit(self, X):
        X = X[np.random.choice(X.shape[0], int(X.shape[0]*self.proportion_to_sammple), replace=False), :]
        Q = np.quantile(X, [0.25, 0.75], axis=0)
        self.Q1 = Q[0, :]
        self.Q3 = Q[1, :]        
        self.IQR = self.Q3 - self.Q1
        return self

    def transform(self, X):
        check_is_fitted(self, 'Q1')
        check_is_fitted(self, 'Q3')
        check_is_fitted(self, 'IQR')
        mask = (
            (X < (self.Q1 - self.l * self.IQR)) |
            (X > (self.Q3 + self.l * self.IQR))
        )
        X[mask] = np.nan
        return X


class ZScoreOutlierRemover(BaseEstimator, TransformerMixin):
    """
    X = np.random.rand(500, 5) * np.random.randn(500, 5) * 15
    imput = ZScoreImputer(1.5)
    imput.fit(X)
    X_t = imput.transform(X)
    """
    def __init__(self, number_of_std_allowed):
        self.number_of_std_allowed = number_of_std_allowed
        self.scaler = StandardScaler()

    def fit(self, X):
        self.scaler.fit(X)        
        return self

    def transform(self, X):
        X = self.scaler.transform(X)
        X[np.abs(X) > self.number_of_std_allowed] = np.nan
        return X



class EWMAOutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, lambda_=1.5):
        self.lambda_ = lambda_

    def fit(self, X, y=None):
        mean = np.mean(X, axis=0)
        s = np.sqrt(self.lambda_ /(2-self.lambda_))*np.std(X, axis=0)    
        self.UCL = mean + 3*s
        self.LCL = mean - 3*s
        return self

    def transform(self, X):
        mask = (
            (X < (self.LCL)) |
            (X > (self.UCL))
        )
        X[mask] = np.nan
        return X