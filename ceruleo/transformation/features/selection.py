import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from ceruleo.transformation import TransformerStep

logger = logging.getLogger(__name__)





class ByNameFeatureSelector(TransformerStep):
    """Select a subset of feature by name

    Parameters:
            features: Feature name or List of features name to select
    """
    def __init__(self, *, features:Union[str, List[str]]= [], name: Optional[str] = None):
        super().__init__(name=name)
        if isinstance(features, str):
            features = [features]
        self.features = features
        self.features_indices = None
        self.features_computed_ = []

    def partial_fit(self, df, y=None):
        if len(self.features) > 0:
            features = [f for f in self.features if f in set(df.columns)]
        else:
            features = list(set(df.columns))

        if len(self.features_computed_) == 0:
            self.features_computed_ = features
        else:
            self.features_computed_ = [
                f for f in self.features_computed_ if f in features
            ]
        return self

    def fit(self, df:pd.DataFrame, y=None):
        """ 
        Find the indices of the features to select

        Parameters:
            df: DataFrame containing the input life
        """
        if len(self.features) > 0:
            features = [f for f in self.features if f in set(df.columns)]
        else:
            features = list(set(df.columns))
        self.features_computed_ = sorted(features)
        return self
        return X.loc[:, self.features_computed_].copy()

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """ 
        Transform the input life

        Parameters:
            X: The input life to be transformed

        Returns:
            A new DataFrame containing only the selected features
        """
        return X.loc[:, self.features_computed_].copy()

    @property
    def n_features(self):
        return len(self.features_computed_)

    def description(self):
        name = super().description()
        return (name, self.features_computed_)

    def __str__(self):
        name, f = self.description()
        features = ', '.join(f)[:10]
        return f'{name} : [{features}]'
        


class PositionFeatures(TransformerStep):
    """
    Reorder the features of the input life

    Parameters:
        features: Dictionary containing the features to reorder and their new position
        name: Name of the step, by default None
    """
    def __init__(self, *, features: dict, name: Optional[str] = None):
        super().__init__(name=name)
        self.features = features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """ 
        Transform the input life by reordering the features

        Parameters:
            X: The input life to be transformed
        
        Returns:
            A new DataFrame containing the features in the order specified in the constructor
        """
        cols = list(X.columns)
        for name, pos in self.features.items():
            a, b = cols.index(name), pos
            cols[b], cols[a] = cols[a], cols[b]
            X = X[cols]
        return X


class DiscardByNameFeatureSelector(TransformerStep):
    """
    Remove a list of features from the input life

    Parameters:
        features: List of features to discard
        name: Name of the step, by default None
    """
    def __init__(self, *, features: List=[], name: Optional[str] = None):
        super().__init__(name=name)
        self.features = features
        self.features_indices = None

    def fit(self, df:pd.DataFrame, y=None):
        """
        Find the indices of the features to discard

        Parameters:
            df: DataFrame containing the set of features to discard
        """
        self.feature_columns = [f for f in df.columns if f not in self.features]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """ 
        Transform the input life

        Parameters:
            X: The input life to be transformed
        
        Returns:
            A new DataFrame containing only the features not in the list of features to discard
        """
        return X.loc[:, self.feature_columns]

    @property
    def n_features(self):
        return len(self.features)


class PandasVarianceThreshold(TransformerStep):
    """ 
    Remove features with variance lower than a variance threshold inserted in input

    Parameters:
        min_variance: Minimum variance threshold
        name: Name of the step, by default None
    """
    def __init__(self, *, min_variance: float, name: Optional[str] = None):
        super().__init__(name=name)
        self.min_variance = min_variance
        self.selected_columns_ = None

    def partial_fit(self, X:pd.DataFrame, y=None):
        """ 
        Find the indexes of the features with variance higher than the threshold

        Parameters:
            X: DataFrame containing the input life
        """
        variances_ = X.var(skipna=True)
        partial_selected_columns_ = X.columns[variances_ > self.min_variance]
        if (
            self.selected_columns_ is not None
            and len(partial_selected_columns_) < len(self.selected_columns_) * 0.5
        ):
            logger.warning(type(self).__name__)
            logger.warning(
                f"Life removed more than a half of the columns. Shape {X.shape}"
            )
            logger.warning(
                f"Current: {len(self.selected_columns_)}. New ones: {len(partial_selected_columns_)}"
            )
        if self.selected_columns_ is None:
            self.selected_columns_ = partial_selected_columns_
        else:
            self.selected_columns_ = [
                f for f in self.selected_columns_ if f in partial_selected_columns_
            ]
        if len(self.selected_columns_) == 0:
            logger.warning(type(self).__name__)
            logger.warning("All features were removed")
        return self

    def fit(self, X:pd.DataFrame, y=None):
        """ 
        Find the indexes of the features with variance higher than the threshold

        Parameters:
            X: DataFrame containing the input life
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input array must be a data frame")
        self.variances_ = X.var(skipna=True)
        self.selected_columns_ = X.columns[self.variances_ > self.min_variance]
        logger.debug(
            f"Dropped columns {[c for c in X.columns if c not in self.selected_columns_]}"
        )
        return self

    def transform(self, X:pd.DataFrame, y=None) -> pd.DataFrame:
        """ 
        Transform the input life

        Parameters:
            X: The input life to be transformed

        Returns:
            A new life containing only the features with variance higher than the threshold
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input array must be a data frame")
        return X[self.selected_columns_].copy()


class NullProportionSelector(TransformerStep):
    """ 
    Remove features with null proportion higher than a threshold inserted in input

    Parameters:
        max_null_proportion: Maximum null proportion threshold
        name: Name of the step, by default None
    """
    def __init__(self, *, max_null_proportion: float, name: Optional[str] = None):
        super().__init__(name=name)
        self.max_null_proportion = max_null_proportion
        self.selected_columns_ = None

    def partial_fit(self, X:pd.DataFrame, y=None):
        """ 
        Find the indexes of the features with null proportion lower than the threshold

        Parameters:
            X: DataFrame containing the input life
        """
        null_proportion = X.isnull().mean()

        partial_selected_columns_ = X.columns[
            null_proportion < self.max_null_proportion
        ]
        if (
            self.selected_columns_ is not None
            and len(partial_selected_columns_) < len(self.selected_columns_) * 0.5
        ):
            logger.warning(type(self).__name__)

        if self.selected_columns_ is None:
            self.selected_columns_ = partial_selected_columns_
        else:
            self.selected_columns_ = [
                f for f in self.selected_columns_ if f in partial_selected_columns_
            ]
        if len(self.selected_columns_) == 0:
            logger.warning(type(self).__name__)
            logger.warning("All features were removed")
        return self

    def fit(self, X:pd.DataFrame, y=None):
        """
        Find the indexes of the features with null proportion lower than the threshold

        Parameters:
            X: DataFrame containing the input life
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input array must be a data frame")
        self.null_proportion = X.isnull().mean()
        self.selected_columns_ = X.columns[
            self.null_proportion < self.max_null_proportion
        ]
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Transform the input life

        Parameters:
            X: The input life to be transformed
        
        Returns:
            A new life containing only the features with null proportion lower than the threshold
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input array must be a data frame")
        return X[self.selected_columns_].copy()


class  MatchFeatureSelector(TransformerStep):
    """Select all the features that match a pattern

    Parameters:
        pattern: Pattern to match
    """
    def __init__(self, *, pattern:str, name: Optional[str] = None):
        super().__init__(name=name)
        self.pattern = pattern
        self.selected_columns_ = None


    def partial_fit(self, df: pd.DataFrame, y=None):

        """ 
        Find the features matching the pattern

        Parameters:
            df: DataFrame containing the entire set of features 
        """

        if self.selected_columns_ is None:
            self.selected_columns_ = [f for f in df.columns if self.pattern in f ]

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Transform the input life

        Parameters:
            X: The input life to be transformed

        Returns:
            A new life with the same index as the input with the missing values replaced by the value in the succesive timestamp 
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input array must be a data frame")
        return X[self.selected_columns_].copy()


class ByTypeFeatureSelector(TransformerStep):
    """Select a subset of feature by type

    Parameters:
            type_: Data type to be selected, by default []
    """
    def __init__(self, *, type_:Union[str, List]= [], name: Optional[str] = None):
        super().__init__(name=name)

        self.features = []
        self.type = type_

    def partial_fit(self, df, y=None):
        if len(self.features) == 0:            
            self.features = df.select_dtypes(include=self.type).columns      
        return self

    def fit(self, df, y=None):
        if len(self.features) == 0:
            self.features = df.select_dtypes(include=self.type).columns

        return self

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input life

        Parameters:
            X: The input life to be transformed

        Returns:
            A new DataFrame containing only the features of the selected type
        """
        return X.loc[:, self.features].copy()

    @property
    def n_features(self):
        return len(self.features)

    def description(self):
        name = super().description()
        return (name, self.features)

    def __str__(self):
        name, f = self.description()
        features = ', '.join(f)[:10]
        return f'{name} : [{features}]'