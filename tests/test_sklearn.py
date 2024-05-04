from ceruleo.dataset.catalog.CMAPSS import CMAPSSDataset, sensor_indices
from ceruleo.transformation import Transformer
from ceruleo.transformation.functional.pipeline.pipeline import make_pipeline
from ceruleo.transformation.features.selection import ByNameFeatureSelector
from ceruleo.transformation.features.scalers import MinMaxScaler
from ceruleo.models.sklearn import (
    CeruleoRegressor,
    TimeSeriesWindowTransformer,
    CeruleoMetricWrapper
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

def test_gridsearch_cv():
    train_dataset = CMAPSSDataset(train=True, models='FD001')
    FEATURES = [train_dataset[0].columns[i] for i in sensor_indices]
    transformer = Transformer(
        pipelineX=make_pipeline(
            ByNameFeatureSelector(features=FEATURES), 
            MinMaxScaler(range=(-1, 1))

        ), 
        pipelineY=make_pipeline(
            ByNameFeatureSelector(features=['RUL']),  
        )
    )


    regressor_gs = CeruleoRegressor(
        TimeSeriesWindowTransformer(
            transformer,
            window_size=32,
            padding=True,
            step=1),   
        Ridge(alpha=15))

    grid_search = GridSearchCV(
        estimator=regressor_gs,
        param_grid={
            'ts_window_transformer__window_size': [5, 6],         
            'regressor': [Ridge(alpha=15), RandomForestRegressor(max_depth=5)]
        },
        scoring=CeruleoMetricWrapper('neg_mean_absolute_error'),
        cv=2,
        verbose=5
    )

    grid_search.fit(train_dataset)
    assert grid_search is not None
