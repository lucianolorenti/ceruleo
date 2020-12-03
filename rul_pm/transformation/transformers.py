import copy
import logging

import numpy as np
from rul_pm.transformation.features.generation import OneHotCategoricalPandas
from rul_pm.transformation.features.selection import (
    ByNameFeatureSelector, DiscardByNameFeatureSelector,
    NullProportionSelector, PandasNullProportionSelector,
    PandasVarianceThreshold)
from rul_pm.transformation.imputers import (MedianImputer, NaNRemovalImputer,
                                            PandasMedianImputer)
from rul_pm.transformation.outliers import IQROutlierRemover
from rul_pm.transformation.utils import PandasToNumpy, TargetIdentity
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)

RESAMPLER_STEP_NAME = 'resampler'


def simple_pipeline(features=[], to_numpy: bool = True):
    return Pipeline(steps=[
        ('initial_selection', ByNameFeatureSelector(features)),
        ('to_numpy', PandasToNumpy() if to_numpy else 'passthrough')
    ])


def step_is_not_missing(step):
    return (step if step is not None else 'passthrough')


def step_if_argument_is_not_missing(cls, param):
    return (cls(param) if param is not None else 'passthrough')


def transformation_pipeline(outlier=None, numerical_imputer=None, scaler=None, resampler=None,
                            features=None, discard=None, locater=None,
                            final=None, categoricals=[], numerical_generator=None,
                            variance_threshold=0, min_null_proportion=0.5,
                            output_df=False, discard_at_end=[]):
    def categorial_pipeline():
        return Pipeline(
            steps=[
                ('imputer', PandasMedianImputer()),
                ('dummy', OneHotCategoricalPandas(categoricals))
            ])

    def numericals_pipeline():
        return Pipeline(
            steps=[
                ('generator', step_is_not_missing(numerical_generator)),
                ('outlier_removal', step_is_not_missing(outlier)),
                ('scaler', step_is_not_missing(scaler)),
                ('NullProportionSelector', step_if_argument_is_not_missing(
                    PandasNullProportionSelector, min_null_proportion)),
                ('selector', step_if_argument_is_not_missing(
                    PandasVarianceThreshold, variance_threshold)),
                ('imputer', numerical_imputer),
                ('final', step_is_not_missing(final))
            ])
    if features is not None and discard is not None:
        raise ValueError(
            'Features and discard cannot be setted at the same time')
    selector = 'passthrough'
    if features is not None:
        selector = ByNameFeatureSelector(features)
        if len(categoricals) > 0:
            categoricals = set(features).intersection(set(categoricals))
    if discard is not None:
        selector = DiscardByNameFeatureSelector(discard)

    if len(features) > 0 and len(categoricals) > 0:
        main_step = ColumnTransformer([
            ("numerical_transformation", numericals_pipeline(), features),
            ("categorical_transformation", categorial_pipeline(), categoricals)
        ])
    else:
        main_step = Pipeline(steps=[
            ('selector', selector),
            ('transform', numericals_pipeline())
        ])

    return Pipeline(steps=[
        (RESAMPLER_STEP_NAME, step_is_not_missing(resampler)),
        ('main_step', main_step),
        ('locater', step_is_not_missing(locater)),
        ('drop_life_column', DiscardByNameFeatureSelector(
            features=['life']+discard_at_end)),
        ('to_numpy', PandasToNumpy() if not output_df else 'passthrough')
    ])


def step_set_enable(transformer, step_name, enabled):
    if not (isinstance(transformer, Pipeline)):
        return
    for (name, step) in transformer.steps:
        if name == step_name and not isinstance(step, str) and step is not None:
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
                 transformerX: TransformerMixin,
                 time_feature: str = None,
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

    def clone(self):
        return copy.deepcopy(self)

    def fit(self, dataset, proportion=1.0):
        logger.info('Fitting Transformer')
        df = dataset.toPandas(proportion)
        self.fitX(df)
        self.fitY(df)

        self.minimal_df = df.head(n=2)
        X = self.transformerX.transform()
        self.number_of_features_ = X.shape[1]
        self.fitted_ = True
        return self

    def fitX(self, df):
        if self.disable_resampling_when_fitting:
            step_set_enable(self.transformerX, RESAMPLER_STEP_NAME, False)
        self.original_columns = df.columns
        self.transformerX.fit(df)
        step_set_enable(self.transformerX, RESAMPLER_STEP_NAME, True)

    def _target(self, df):
        if self.time_feature is not None:
            if isinstance(self.target_column, list):
                select_features = [self.time_feature] + self.target_column
            else:
                select_features = [self.time_feature,  self.target_column]
            return df[select_features]
        else:
            return df[[self.target_column]]

    def fitY(self, df):
        if self.disable_resampling_when_fitting:
            step_set_enable(self.transformerY, RESAMPLER_STEP_NAME, False)
        self.transformerY.fit(self._target(df))
        step_set_enable(self.transformerY, RESAMPLER_STEP_NAME, True)

    def transform(self, df):
        check_is_fitted(self, 'fitted_')
        return (self.transformX(df), self.transformY(df))

    def transformY(self, df):
        return np.squeeze(
            self.transformerY.transform(self._target(df)))

    def transformX(self, df):
        return self.transformerX.transform(df)

    def columns(self):
        pass

    @property
    def n_features(self):
        return self.number_of_features_

    def column_names(self):
        temp = self.transformerX.steps[-1]
        self.transformerX.steps[-1] = ('empty', 'passthrough')
        cnames = self.transformerX.transform(self.minimal_df).columns.values
        self.transformerX.steps[-1] = temp
        return cnames

    def description(self):
        return {
            'target_column': self.target_column,
            'features': self.features,
            'transformerX': transformer_info(self.transformerX),
            'transformerY': transformer_info(self.transformerY),
        }


class SimpleTransformer(Transformer):
    def __init__(self, target_column: str, time_feature: str = None, to_numpy: bool = True):
        super().__init__(target_column, simple_pipeline(to_numpy=to_numpy),
                         transformerY=TargetIdentity(), time_feature=time_feature, disable_resampling_when_fitting=True)
