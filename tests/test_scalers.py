

import numpy as np
import pandas as pd

from ceruleo.transformation.features.scalers import (MinMaxScaler, RobustMinMaxScaler)
from sklearn.preprocessing import RobustScaler

class TestImputers():
    def test_MinMaxScaler_withNonde(self):
        scaler = MinMaxScaler(range=(-1, 1), clip=False)
       

        df1 = pd.DataFrame({
            'a': [None] * 6000,
            'b': np.random.randn(6000)*5 + 25
        })
        scaler.partial_fit(df1)


        df2 = pd.DataFrame({
            'a': [None] * 6000,
            'b': np.random.randn(6000)*3 + 88
        })
        scaler.partial_fit(df2)

        scaled_df1 = scaler.transform(df1)
        scaled_df2 = scaler.transform(df2)
        assert np.all(scaled_df1['b'] >= -1) and np.all(scaled_df1['b'] <= 1)


    def test_MinMaxScaler(self):
            
        scaler = MinMaxScaler(range=(-1, 1), clip=False)
       

        df1 = pd.DataFrame({
            'a': np.random.randn(6000)*5 + 25,
            'b': np.random.randn(6000)*5 + 25
        })
        scaler.partial_fit(df1)


        df2 = pd.DataFrame({
            'a': np.random.randn(6000)*3 + 15,
            'b': np.random.randn(6000)*3 + 88
        })
        scaler.partial_fit(df2)

        scaled_df1 = scaler.transform(df1)
        scaled_df2 = scaler.transform(df2)
        assert np.all(scaled_df1['a'] >= -1) and np.all(scaled_df1['a'] <= 1)
        assert np.all(scaled_df1['b'] >= -1) and np.all(scaled_df1['b'] <= 1)
        assert np.all(scaled_df2['a'] >= -1) and np.all(scaled_df2['a'] <= 1)
        assert np.all(scaled_df2['b'] >= -1) and np.all(scaled_df2['b'] <= 1)






    def test_RobustMinMaxScaler(self):

        scaler = RobustMinMaxScaler(range=(-1, 1), clip=False, lower_quantile=0.1, upper_quantile=0.9)
        sk_scaler = RobustScaler(with_centering=False, with_scaling=True, quantile_range=(10, 90))

        df1 = pd.DataFrame({
            'a': np.random.randn(6000)*5 + 25,
            'b': np.random.randn(6000)*5 + 25
        })
        scaler.partial_fit(df1)


        df2 = pd.DataFrame({
            'a': np.random.randn(6000)*3 + 15,
            'b': np.random.randn(6000)*3 + 88
        })
        scaler.partial_fit(df2)


        sk_scaler.fit(pd.concat((df1, df2)))

        sk_scaler.transform(df1)
      
  
