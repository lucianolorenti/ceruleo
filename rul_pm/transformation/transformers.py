import copy
import logging

import numpy as np
from rul_pm.transformation.features.generation import OneHotCategoricalPandas
from rul_pm.transformation.features.selection import (
    ByNameFeatureSelector, PandasNullProportionSelector,
    PandasVarianceThreshold)
from rul_pm.transformation.utils import (PandasFeatureUnion, PandasToNumpy,
                                         TargetIdentity)
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

RESAMPLER_STEP_NAME = 'resampler'


class LivesPipeline(Pipeline):
    def partial_fit(self, X, y=None):
        args = [X, y]
        for name, est in self.steps:
            if est == 'passthrough':
                continue

            est.partial_fit(*args)

            X_transformed = est.transform(args[0])
            args = [X_transformed, y]
        return self


def simple_pipeline(features=[], to_numpy: bool = True):
    return LivesPipeline(steps=[
        ('initial_selection', ByNameFeatureSelector(features)),
        ('to_numpy', PandasToNumpy() if to_numpy else 'passthrough')
    ])


def step_is_not_missing(step):
    return (step if step is not None else 'passthrough')


def step_if_argument_is_not_missing(cls, param):
    return (cls(param) if param is not None else 'passthrough')


def numericals_pipeline(numerical_features: list = None,
                        numerical_generator=None, outlier=None, scaler=None,
                        min_null_proportion=0.3, variance_threshold=0, imputer=None, final=None) -> LivesPipeline:
    selector = 'passthrough'
    if numerical_features is not None:
        selector = ByNameFeatureSelector(numerical_features)
    return LivesPipeline(
        steps=[
            ('selector', selector),
            ('generator', step_is_not_missing(numerical_generator)),
            ('outlier_removal', step_is_not_missing(outlier)),
            ('scaler', step_is_not_missing(scaler)),
            ('NullProportionSelector', step_if_argument_is_not_missing(
                PandasNullProportionSelector, min_null_proportion)),
            ('variance_selector', step_if_argument_is_not_missing(
                PandasVarianceThreshold, variance_threshold)),
            ('imputer', imputer),
            ('NullProportionSelector_1', step_if_argument_is_not_missing(
                PandasNullProportionSelector, min_null_proportion)),
            ('variance_selector_1', step_if_argument_is_not_missing(
                PandasVarianceThreshold, variance_threshold)),
            ('final', step_is_not_missing(final))
        ])


def categorial_pipeline(categoricals: list) -> PandasFeatureUnion:
    return PandasFeatureUnion(
        [
            (f'dummy_{c}', OneHotCategoricalPandas(c)) for c in categoricals]
    )


def transformation_pipeline(resampler=None,
                            numericals_pipeline=None,
                            categorial_pipeline=None,
                            output_df=False) -> LivesPipeline:
    main_step = None
    if categorial_pipeline is not None:
        main_step = PandasFeatureUnion([
            ("numerical_transformation", numericals_pipeline),
            ("categorical_transformation", categorial_pipeline)
        ])
    else:
        main_step = numericals_pipeline

    return LivesPipeline(steps=[
        (RESAMPLER_STEP_NAME, step_is_not_missing(resampler)),
        ('main_step', main_step),
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
                 transformerX: LivesPipeline,
                 time_feature: str = None,
                 transformerY: LivesPipeline = TargetIdentity(),
                 disable_resampling_when_fitting: bool = True):

        self.transformerX = transformerX
        self.transformerY = transformerY
        self.target_column = target_column
        self.features = None
        self.time_feature = time_feature
        self.disable_resampling_when_fitting = disable_resampling_when_fitting
        if isinstance(self.target_column, str):
            self.target_column = [self.target_column]

    def _process_selected_features(self):
        if self.transformerX['selector'] is not None:
            selected_columns = (self.transformerX['selector'].get_support(
                indices=True))
            self.features = [self.features[i] for i in selected_columns]

    def clone(self):
        return copy.deepcopy(self)

    def fit(self, dataset, proportion=1.0):
        logger.debug('Fitting Transformer')
        for life in dataset:
            self.partial_fitX(life)
            self.partial_fitY(life)

        self.minimal_df = dataset[0].head(n=5)
        X = self.transformerX.transform(self.minimal_df)
        self.number_of_features_ = X.shape[1]
        self.fitted_ = True
        self.column_names = self._compute_column_names()
        return self

    def partial_fitX(self, df):
        self.transformerX.partial_fit(df)

    def partial_fitY(self, df):
        self.transformerY.partial_fit(self._target(df))

    def _target(self, df):
        if self.time_feature is not None:
            if isinstance(self.target_column, list):
                select_features = [self.time_feature] + self.target_column
            else:
                select_features = [self.time_feature,  self.target_column]
            return df[select_features]
        else:
            return df[self.target_column]

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

    @ property
    def n_features(self):
        return self.number_of_features_

    def _compute_column_names(self):
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
