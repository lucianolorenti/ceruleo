import logging

import numpy as np
from rul_gcd.transformation.feature_selection import (ByNameFeatureSelector,
                                                      NullProportionSelector)
from rul_gcd.transformation.imputers import NaNRemovalImputer
from rul_gcd.transformation.outliers import IQROutlierRemover
from rul_gcd.transformation.utils import PandasToNumpy, TargetIdentity
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)

RESAMPLER_STEP_NAME = 'resampler'


def simple_pipeline(features=[]):
    return Pipeline(steps=[
        ('initial_selection', ByNameFeatureSelector(features)),
        ('to_numpy', PandasToNumpy())
    ])


def transformation_pipeline(outlier=IQROutlierRemover(),
                            imputer=NaNRemovalImputer(),
                            scaler=RobustScaler(),
                            resampler=None,
                            features=[]):
    return Pipeline(steps=[
        ('initial_selection', ByNameFeatureSelector(features)),
        (RESAMPLER_STEP_NAME,
         resampler if resampler is not None else 'passthrough'),
        ('to_numpy', PandasToNumpy()),
        ('outlier_removal', outlier if outlier is not None else 'passthrough'),
        ('NullProportionSelector', NullProportionSelector()),
        ('selector', VarianceThreshold(0)),
        ('scaler', scaler if scaler is not None else 'passthrough'),
        ('imputer', imputer if imputer is not None else 'passthrough'),
    ])


def step_set_enable(transformer, step_name, enabled):
    if not (isinstance(transformer, Pipeline)):
        return
    for (name, step) in transformer.steps:
        if name == step_name and not isinstance(step, str):
            step.enabled = enabled


def transformer_info(transformer):
    if isinstance(transformer, Pipeline):
        return [(name, transformer_info(step))
                for name, step in transformer.steps]
    elif isinstance(transformer, TransformerMixin):
        return transformer.__dict__


class Transformer:
    """
    Transform each life

    Parameters
    ----------
    target_column : str
                    Column name with the target. Usually where the RUL resides         
    time_feature: str
                  Column name of the timestamp feature
    transformerX: TransformerMixin,
                  Transformer that will be applied to the life data
    transformerY: TransformerMixin default: TargetIdentity()
                  Transformer that will be applied to the target.
    disable_resampling_when_fitting: bool = True
                                     Wether to disable the resampling when the model is being fit.
                                     This can reduce the memory requirements when fitting
    """
    def __init__(self,
                 target_column: str,
                 time_feature: str,
                 transformerX: TransformerMixin,
                 transformerY: TransformerMixin = TargetIdentity(),
                 disable_resampling_when_fitting: bool = True):
        self.transformerX = transformerX
        self.transformerY = transformerY
        self.target_column = target_column
        self.features = None
        self.time_feature = time_feature
        self.disable_resampling_when_fitting = disable_resampling_when_fitting

    def _process_selected_features(self):
        if self.transformerX['selector'] is not None:
            selected_columns = (self.transformerX['selector'].get_support(
                indices=True))
            self.features = [self.features[i] for i in selected_columns]

    def fit(self, dataset):
        logger.info('Fitting Transformer')
        df = dataset.toPandas()
        self.fitX(df)
        self.fitY(df)

        X = self.transformerX.transform(df.head(n=2))
        self.number_of_features_ = X.shape[1]
        self.fitted_ = True
        return self

    def fitX(self, df):
        if self.disable_resampling_when_fitting:
            step_set_enable(self.transformerX, RESAMPLER_STEP_NAME, False)
        self.transformerX.fit(df)
        step_set_enable(self.transformerX, RESAMPLER_STEP_NAME, True)

    def fitY(self, df):
        if self.disable_resampling_when_fitting:
            step_set_enable(self.transformerY, RESAMPLER_STEP_NAME, False)
        self.transformerY.fit(df[[self.time_feature, self.target_column]])
        step_set_enable(self.transformerY, RESAMPLER_STEP_NAME, True)

    def transform(self, df):
        check_is_fitted(self, 'fitted_')
        return (self.transformX(df), self.transformY(df))

    def transformY(self, df):
        return np.squeeze(
            self.transformerY.transform(
                (df[[self.time_feature, self.target_column]])))

    def transformX(self, df):
        return self.transformerX.transform(df)

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


class SimpleTransformer(Transformer):
    def __init__(self, target_column: str, time_feature: str):
        super().__init__(target_column, time_feature, simple_pipeline(), TargetIdentity(), True)
                 