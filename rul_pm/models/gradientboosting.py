from rul_pm.models.sklearn import SKLearnModel
from xgboost import XGBRegressor


class GradientBoostingModel(SKLearnModel):
    def __init__(self,
                 model_params={},
                 **kwargs):
        model = XGBRegressor( **model_params )
        super().__init__(
            model=model,
            **kwargs)
        
    
    def fit(self, train_dataset, validation_dataset, refit_transformer=True, **kwargs):
        if refit_transformer:
            self.transformer.fit(train_dataset)
        X, y, _ = self.get_data(train_dataset)
        X_val, y_val, _ = self.get_data(validation_dataset)
        self.model.fit(X, y.ravel(), eval_set=[(X_val, y_val)], **kwargs)
        return self