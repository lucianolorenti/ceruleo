import logging

import numpy as np
from rul_gcd.transformation.feature_selection import (ByNameFeatureSelector,
                                                      NullProportionSelector,
                                                      UsefulFeatureSelector)
from rul_gcd.transformation.imputers import NaNRemovalImputer
from rul_gcd.transformation.outliers import IQROutlierRemover
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


class PandasToNumpy(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.values


def transformation_pipeline(outlier=IQROutlierRemover(), imputer=NaNRemovalImputer(),
                            scaler=RobustScaler(), resampler=None, features=[]):
    return Pipeline(steps=[
        ('initial_selection', ByNameFeatureSelector(features)),
        ('resampler', resampler if resampler is not None else 'passthrough'),
        ('to_numpy', PandasToNumpy()),
        ('outlier_removal', outlier if outlier is not None else 'passthrough'),
        ('NullProportionSelector', NullProportionSelector()),
        ('selector_1', VarianceThreshold(0.01)),
        ('scaler', scaler if scaler is not None else 'passthrough'),
        ('imputer', imputer if imputer is not None else 'passthrough'),
        ('selector_2', VarianceThreshold(0.01))
    ])



def transformer_info(transformer):
    if isinstance(transformer,  Pipeline):
        return [(name, transformer_info(step)) for name, step in transformer.steps]
    elif isinstance(transformer, TransformerMixin):
        return transformer.__dict__


class Transformer:
    def __init__(self, target_column, transformerX, transformerY=None):
        self.transformerX = transformerX
        self.transformerY = transformerY
        self.target_column = target_column
        self.features = None

    def _process_selected_features(self):
        if self.transformerX['selector'] is not None:
            selected_columns = (self.transformerX['selector']
                                    .get_support(indices=True))
            self.features = [self.features[i] for i in selected_columns]
        

    def fit(self, dataset):
        logger.info('Fitting Transformer')
        df = dataset.toPandas()
        if self.transformerY is not None:
            self.transformerY.fit(df[self.target_column].values.reshape(-1, 1))        
        self.transformerX.fit(df)

        X = self.transformerX.transform(df.head(n=2))
        self.number_of_features_ = X.shape[1]
        self.fitted_ = True
        return self

    def transform(self, df):     
        check_is_fitted(self, 'fitted_')
        return (self.transformX(df),  self.transformY(df))       

    def transformX(self, df):
        return self.transformerX.transform(df)

    def transformY(self, df):
        y = df[self.target_column].values.reshape(-1, 1)
        if self.transformerY is not None:
            y = self.transformerY.transform(y)
        return y


    @property
    def n_features(self):
        return self.number_of_features_

    def description(self):
        return {
            'target_column': self.target_column,
            'features': self.features,
            'transformerX': transformer_info(self.transformerX),
            'transformerY': transformer_info(self.transformerY),
        }
