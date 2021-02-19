import copy
import logging
from typing import Optional

import numpy as np
import pandas as pd
from rul_pm.transformation.features.generation import OneHotCategoricalPandas
from rul_pm.transformation.features.selection import (
    ByNameFeatureSelector, PandasNullProportionSelector,
    PandasVarianceThreshold)
from rul_pm.transformation.featureunion import PandasFeatureUnion
from rul_pm.transformation.pipeline import LivesPipeline
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

RESAMPLER_STEP_NAME = 'resampler'


def transformer_info(transformer):
    if isinstance(transformer, LivesPipeline):
        return [(name, transformer_info(step))
                for name, step in transformer.steps]
    elif isinstance(transformer, PandasFeatureUnion):
        return [
            ('name', 'FeatureUnion'),
            ('steps', [(name, transformer_info(step))
                       for name, step in transformer.transformer_list]),
            ('transformer_weights', transformer.transformer_weights)
        ]

    elif hasattr(transformer, 'get_params'):
        d = transformer.get_params()
        d.update({'name': type(transformer).__name__})
        return [(k, d[k]) for k in sorted(d.keys())]
    elif isinstance(transformer, str) and transformer == 'passthrough':
        return transformer
    else:
        print(type(transformer))

        raise ValueError('Pipeline elements must have the get_params method')


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
            ('imputer', step_is_not_missing(imputer)),
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
                            categorial_pipeline=None) -> LivesPipeline:
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
        ('main_step', main_step)
    ])


class Transformer:
    """
    Transform each life

    Parameters
    ----------   
    transformerX: LivesPipeline,
                  Transformer that will be applied to the life data
    transformerY: LivesPipeline 
                  Transformer that will be applied to the target.
    time_feature: str
                  Column name of the timestamp feature

    """

    def __init__(self,
                 transformerX: LivesPipeline,
                 transformerY: LivesPipeline,
                 time_feature: str = None,
                 transformerMetadata: Optional[LivesPipeline] = None):

        self.transformerX = transformerX
        self.transformerY = transformerY
        self.transformerMetadata = transformerMetadata
        self.features = None
        self.time_feature = time_feature

    def _process_selected_features(self):
        if self.transformerX['selector'] is not None:
            selected_columns = (self.transformerX['selector'].get_support(
                indices=True))
            self.features = [self.features[i] for i in selected_columns]

    def clone(self):
        return copy.deepcopy(self)

    def fit(self, dataset):
        logger.debug('Fitting Transformer')
        self.transformerX.fit(dataset)
        self.transformerY.fit(dataset)
        self.fitTransformerMetadata(dataset)

        self.minimal_df = dataset[0].head(n=5)
        X = self.transformerX.transform(self.minimal_df)
        self.number_of_features_ = X.shape[1]
        self.fitted_ = True
        self.column_names = self._compute_column_names()
        return self

    def fitTransformerMetadata(self, life: pd.DataFrame):
        if self.transformerMetadata is not None:
            if hasattr(self.transformerMetadata, 'partial_fit'):
                self.transformerMetadata.partial_fit(life)

    def transform(self, df: pd.DataFrame):
        check_is_fitted(self, 'fitted_')
        return (self.transformX(df), self.transformY(df), self.transformMetadata(df))

    def transformMetadata(self, df: pd.DataFrame) -> Optional[any]:
        if self.transformerMetadata is not None:
            return self.transformerMetadata.transform(df)
        else:
            return None

    def transformY(self, df):
        return np.squeeze(
            self.transformerY.transform(df).values
        )

    def transformX(self, df):
        return self.transformerX.transform(df).values

    def columns(self):
        return self.column_names

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
            'features': self.features,
            'transformerX': transformer_info(self.transformerX),
            'transformerY': transformer_info(self.transformerY),
        }

    def __str__(self):
        return str(self.description())
