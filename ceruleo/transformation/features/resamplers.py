import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from ceruleo.transformation import TransformerStep


class Subsample(TransformerStep):
    def __init__(self, *, steps: int, **kwargs):
        super().__init__(**kwargs)
        self.steps = steps

    def transform(self, X: pd.DataFrame):
        return X.iloc[:: self.steps, :]


class IndexMeanResampler(TransformerStep):
    """Resample 


    When the index of the run-to-failure cycle is a time feature

    Parameters:

        rule: 
    """
    def __init__(self, *, rule,  **kwargs):
        super().__init__(**kwargs)
        self.rule = rule
    
    def transform(self, X: pd.DataFrame):
        return X.resample(self.rule).mean().dropna()
        


class SubsamplerTransformer(TransformerStep):
    """IntegerIndexResamplerTransformer

    Resample the time series with an integer index and interpolate linearly the values

    Parameters:

        time_feature: Time feature
        steps:  Number of steps
        drop_time_feature: Drop the time feature
    """

    def __init__(self, *args, time_feature: str, steps: int, drop_time_feature: bool):
        super().__init__(*args)
        self._time_feature_name = time_feature
        self._time_feature = None
        self.steps = steps
        self.drop_time_feature = drop_time_feature

    def partial_fit(self, X: pd.DataFrame):
        """Obtain the name of the feature used as time

        Parameters
        ----------
        X : pd.DataFrame
            The current time-series to be fitted

        Returns
        -------
        self

        """
        if self._time_feature is None:
            self._time_feature = self.find_feature(X, self._time_feature_name)
            if self._time_feature is None:
                raise ValueError("Time feature not found")

        return self

    def transform(self, X: pd.DataFrame):
        X = X.groupby(X[self._time_feature] // self.steps, sort=False).mean()
        if self.drop_time_feature:
            X = X.drop(columns=[self._time_feature])

        return X


class IntegerIndexResamplerTransformer(TransformerStep):
    """IntegerIndexResamplerTransformer

    Resample the time series with an integer index and interpolate linearly the values

    Parameters
    ----------
    time_feature : str
        Time feature
    steps : int
        Number of steps
    drop_time_feature: bool
        Drop the time feature
    """

    def __init__(self, *args, time_feature: str, steps: int, drop_time_feature: bool):
        super().__init__(*args)
        self._time_feature_name = time_feature
        self._time_feature = None
        self.steps = steps
        self.drop_time_feature = drop_time_feature

    def partial_fit(self, X: pd.DataFrame):
        """Obtain the name of the feature used as time

        Parameters
        ----------
        X : pd.DataFrame
            The current time-series to be fitted

        Returns
        -------
        self

        """
        if self._time_feature is None:
            self._time_feature = self.find_feature(X, self._time_feature_name)
        return self

    def transform(self, X: pd.DataFrame):
        Told = X[self._time_feature].values
        Tnew = np.arange(Told.min(), Told.max(), self.steps)
        new_columns = X.columns.values
        if self.drop_time_feature:
            new_columns = np.delete(
                new_columns, np.where(new_columns == self._time_feature)
            )

        new_X = pd.DataFrame(index=Tnew, columns=new_columns)
        for c in new_columns:
            try:
                F = interp1d(Told, X[c])
                new_X[c] = F(Tnew)
            except ValueError:
                # https://stackoverflow.com/questions/62015823/interpolating-categorical-data-in-python-nearest-previous-value/62016097#62016097
                y = X[c].values
                f = interp1d(
                    Told,
                    range(len(y)),
                    kind="nearest",
                    fill_value=(0, len(y) - 1),
                    bounds_error=False,
                )
                y_idx = f(Tnew)
                new_X[c] = [y[int(i)] for i in y_idx]
        return new_X
