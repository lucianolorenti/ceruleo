

import numpy as np
import pandas as pd
from ceruleo.transformation.features.operations import Divide, Sum
from ceruleo.transformation.features.selection import ByNameFeatureSelector
from ceruleo.transformation.functional.transformers import Transformer



class TestImputers():

    def test_Divide(self):

        divider = Divide()
        
        df1 = pd.DataFrame({
            'a': np.random.randn(5),
            'b': np.random.randn(5)
        })
        true_result = pd.DataFrame({
            'a': np.ones(5)*5,
            'b': np.ones(5)*3
        })
        df2 = df1.copy()
        df2['a'] = df1['a'] * 5 
        df2['b'] = df1['b'] * 3
        result_transformed = divider.transform([df2, df1])
        error = (true_result - result_transformed).sum().sum()
        assert(error < 0.0000000000001)

    def test_Sum(self):
        divider = Sum()
        
        df1 = pd.DataFrame({
            'a': np.ones(5)*3,
            'b': np.ones(5)*4
        })
        df2 = pd.DataFrame({
            'a': np.ones(5)*2,
            'b': np.ones(5)*3
        })
        true_result = pd.DataFrame({
            'a': np.ones(5)*5,
            'b': np.ones(5)*7
        })
        result_transformed = divider.transform([df2, df1])
        error = (result_transformed - true_result).sum().sum()
        assert(error < 0.0000000000001)

    def test_opeartorOverloading(self):
        pipe = ByNameFeatureSelector(features=['a', 'b'])
        pipe1 = pipe / 2 

        pipe2 = pipe / 3

        pipe = pipe1 + pipe2

        df = pd.DataFrame({
            'a': np.random.rand(5),
            'b': np.random.rand(5),
            'c': np.random.rand(5),
        })
        df1 = df*6

        df_result = (df1/2 + df1/3)[['a', 'b']]

        test_transformer = Transformer(pipelineX=pipe)
        test_transformer.fit([df1])
        result = test_transformer.transform(df1)[0]

        error = (df_result - result).sum().sum()
        assert(error < 0.00000001)

        