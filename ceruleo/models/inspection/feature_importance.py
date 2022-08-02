from copy import deepcopy

import numpy as np
import pandas as pd

from ceruleo.models.model import TrainableModel
from ceruleo.transformation.utils import column_names_window
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm.auto import tqdm


def xgboost_feature_importance(dataset, window, step, transformer):
    from ceruleo.models.xgboost_ import XGBoostModel
    model = XGBoostModel(window, step, transformer)
    model.fit(dataset)
    names = transformer.column_names()
    pd.DataFrame({
        'names': names,
        'importance': model.feature_importances()
    })


class PermuteColumn(BaseEstimator, TransformerMixin):
    def __init__(self, idx: int):
        self.col_idx = idx

    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        shuffling_idx = np.arange(X.shape[0])
        np.random.shuffle(shuffling_idx)
        X = X.copy()
        X[:, self.col_idx] = X[shuffling_idx, self.col_idx]
        return X


def permtuation_feature_importance(model: TrainableModel, dataset, metric, n_repeats: int = 5):
    import gc
    y_true = model.true_values(dataset)
    y_pred = model.predict(dataset)
    baseline_score = metric(np.squeeze(y_true), np.squeeze(y_pred))

    scores_dict = {
        'baseline': baseline_score
    }
    original_pipeline = deepcopy(model.transformer.transformerX)
    model.transformer.transformerX.steps.append(
        ('permute', PermuteColumn(-1)))
    for col_idx, column in tqdm(list(enumerate(model.transformer.column_names))):
        scores = np.zeros(n_repeats)
        model.transformer.transformerX.steps[-1][1].col_idx = col_idx
        for n_round in range(n_repeats):
            y_pred = model.predict(dataset)
            score = metric(np.squeeze(y_true), np.squeeze(y_pred))
            scores[n_round] = score
        scores_dict[column] = scores
        gc.collect()
    model.transformer.transformerX = original_pipeline
    return scores_dict


def coefficient_table(model: TrainableModel) -> pd.DataFrame:
    if not hasattr(model.model, 'coef_'):
        raise ValueError('Model does not have coef_')
    cnames = column_names_window(model.transformer.columns(), model.window)

    return pd.DataFrame(zip(cnames, model.model.coef_),
                        columns=['Column', 'Coef'])
