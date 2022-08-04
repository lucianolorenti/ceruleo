

import numpy as np
import pandas as pd

from ceruleo.transformation.features.scalers import (RobustMinMaxScaler)
from sklearn.preprocessing import RobustScaler

class TestImputers():

    def test_PandasRemoveInf(self):

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
      
