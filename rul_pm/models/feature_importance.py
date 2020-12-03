from rul_pm.iterators.iterators import WindowedDatasetIterator
import pandas as pd
from rul_pm.models.xgboost_ import XGBoostModel

def xgboost_feature_importance(dataset, window, step, transformer):
    model = XGBoostModel(window, step, transformer)
    model.fit(dataset)
    names = transformer.column_names()
    pd.DataFrame({
        'names': names,
        'importance': model.feature_importances()
    })