# RUL PM
Remaining useful life estimation utilities

[![Coverage Status](https://coveralls.io/repos/github/lucianolorenti/rul_pm/badge.svg?branch=main&t=dYuRdM)](https://coveralls.io/github/lucianolorenti/rul_pm?branch=main)
[![Documentation](https://img.shields.io/badge/documentation-dev-brightgreen)](https://lucianolorenti.github.io/rul_pm/)

## Content
The library contains functionalities for:

### Handling PM datasets
#### Life iterators
```python
from rul_pm.dataset.lives_dataset import AbstractLivesDataset

class Dataset(AbstractLivesDataset):
   ...

for life in dataset:
  # life is a pandas DataFrame
```
This dataset allows using the data transformation utilities provided
### Data transformation
The data transformation module provides a functional API and an API similar to the scikit-learn pipeline transformation to extract features and transform the life data.
```python
pipe = ByNameFeatureSelector(features)
pipe = RollingStatistics(25)(pipe)
pipe = PandasVarianceThreshold(0)(pipe)
pipe = PandasNullProportionSelector(0.1)(pipe)
pipe = PandasMinMaxScaler((-1,1))(pipe)
pipe = PandasRemoveInf()(pipe)
pipe = PandasTransformerWrapper(
           SimpleImputer(fill_value=-2, strategy='constant'))(pipe)

target_pipe = ByNameFeatureSelector(['RUL'])

transformer = Transformer(
    transformerX=pipe.build()    
    transformerY=target_pipe.build()
)
```
### Plotting utilities for displaying results of model evaluation
### Models
The model class is an abstract class that provides functionalities for fitting models.
These functionalities involves preparing the data for feeding the model, fit the model, and
predict. 
#### Keras
#### Scikit-Learn
The SKLearnModel class allows to fit a model that provides the Scikit-learn API. These includes also de XGBoost model.

```python
model = SKLearnModel(
           model=RandomForestRegressor(n_estimators=500),
           window=1,
           step = 3,
           transformer=transformer,
           shuffle='all')
model.fit(train_dataset)
```

## Literature Review
- [CMAPPS](research/CMAPPS.ipynb)
  * Remaining useful life estimation in prognostics using deep convolution neural networks    
      *  Xiang Li
      *  Qian Ding
      *  Jian-Qiao Sunc
  * [Remaining useful life prediction using multi-scale deep convolutional neural network](https://doi.org/10.1016/j.asoc.2020.106113)
      * Han Li
      * Wei Zhao
      * Yuxi Zhang
      * Enrico Zio
  * Temporal Convolutional Memory Networks forRemaining Useful Life Estimation of Industrial Machinery
      

## Example
- [Example of usage](examples/ExampleAircraftEngine.ipynb) 
