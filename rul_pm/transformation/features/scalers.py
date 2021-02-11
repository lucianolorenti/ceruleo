
from typing import Optional

import pandas as pd
from rul_pm.transformation.transformerstep import TransformerStep


class PandasMinMaxScaler(TransformerStep):
    def __init__(self, range: tuple, name: Optional[str] = None, clip: bool = True):
        super().__init__(name)
        self.range = range
        self.min = range[0]
        self.max = range[1]
        self.data_min = None
        self.data_max = None
        self.clip = clip

    def partial_fit(self, df, y=None):
        partial_data_min = df.min()
        partial_data_max = df.max()
        if self.data_min is None:
            self.data_min = partial_data_min
            self.data_max = partial_data_max
        else:
            self.data_min = (pd
                             .concat([self.data_min, partial_data_min], axis=1)
                             .min(axis=1))
            self.data_max = (pd
                             .concat([self.data_max, partial_data_max], axis=1)
                             .max(axis=1))
        return self

    def fit(self, df, y=None):
        self.data_min = df.min()
        self.data_max = df.max()
        return self

    def transform(self, X):
        X = ((X-self.data_min)/(self.data_max-self.data_min)
             * (self.max - self.min)) + self.min
        if self.clip:
            X.clip(lower=self.min, upper=self.max, inplace=True)
        return X


class PandasStandardScaler(TransformerStep):
    def __init__(self,  name: Optional[str] = None):
        super().__init__(name)
        self.std = None
        self.mean = None

    def partial_fit(self, df, y=None):
        partial_data_mean = df.mean()
        partial_data_std = df.std()
        if self.mean is None:
            self.mean = partial_data_mean
            self.std = partial_data_std
        else:
            self.mean = (pd
                         .concat([self.mean, partial_data_mean], axis=1)
                         .mean(axis=1))
            self.std = (pd
                        .concat([self.std, partial_data_std], axis=1)
                        .mean(axis=1))
        return self

    def fit(self, df, y=None):
        self.mean = df.mean()
        self.std = df.std()
        return self

    def transform(self, X):
        return (X-self.mean)/(self.std)
