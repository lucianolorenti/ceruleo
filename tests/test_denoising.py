import numpy as np
import pandas as pd
from ceruleo.transformation.features.denoising import (GaussianFilter,
    EWMAFilter, MeanFilter, MedianFilter, MultiDimensionalKMeans, OneDimensionalKMeans, SavitzkyGolayTransformer)


class TestDenoising:
    def test_SavitzkyGolayTransformer(self):

        remover = SavitzkyGolayTransformer(window=25)
        df = pd.DataFrame(
            {
                "a": np.random.rand(2000),
                "b": np.random.rand(2000),
            }
        )
        df_new = remover.fit_transform(df)
        
        assert df_new.shape[1] == 2 
        assert df_new['a'].max() <= df['a'].max()


    def test_MeanFilter(self):

        remover = MeanFilter(50, center=False)
        df = pd.DataFrame(
            {
                "a": np.random.rand(2000),
                "b": np.random.rand(2000),
            }
        )
        df_new = remover.fit_transform(df)
        
        assert df_new.shape[1] == 2 
        assert df_new['a'].max() <= df['a'].max()
        assert df_new['a'].min() >= df['a'].min()

    def test_MedianFilter(self):

        remover = MedianFilter(50)
        df = pd.DataFrame(
            {
                "a": np.random.rand(2000),
                "b": np.random.rand(2000),
            }
        )
        df_new = remover.fit_transform(df)
        
        assert df_new.shape[1] == 2 
        assert df_new['a'].max() <= df['a'].max()
        assert df_new['a'].min() >= df['a'].min()


    def test_OneDimensionalKMeans(self):
        remover = OneDimensionalKMeans(n_clusters=3)
        df = pd.DataFrame(
            {
                "a": np.random.rand(2000),
                "b": np.random.rand(2000),
            }
        )
        remover.partial_fit(df)
        df_new = remover.transform(df)
        
        assert df_new.shape[1] == 2 
        assert len(df_new['a'].unique()) == 3
        assert len(df_new['b'].unique()) == 3
        
    def test_MultiDimensionalKMeans(self):
        remover = MultiDimensionalKMeans(n_clusters=3)
        df = pd.DataFrame(
            {
                "a": np.random.rand(2000),
                "b": np.random.rand(2000),
            }
        )
        remover.partial_fit(df)
        df_new = remover.transform(df)
        
        assert df_new.shape[1] == 2 
        assert df_new[['a', 'b']].drop_duplicates().shape[0] == 3


    def test_EWMAFilter(self):
        remover = EWMAFilter(50)
        df = pd.DataFrame(
            {
                "a": np.random.rand(2000),
                "b": np.random.rand(2000),
            }
        )
        remover.partial_fit(df)
        df_new = remover.transform(df)
        
        assert df_new.shape[1] == 2 
        assert df_new['a'].max() <= df['a'].max()
        assert df_new['a'].min() >= df['a'].min()
        

    def test_GaussianFilter(self):
        remover = GaussianFilter(50, 5)
        df = pd.DataFrame(
            {
                "a": np.random.rand(2000),
                "b": np.random.rand(2000),
            }
        )
        remover.partial_fit(df)
        df_new = remover.transform(df)
        
        assert df_new.shape[1] == 2 
        assert df_new['a'].max() <= df['a'].max()
        assert df_new['a'].min() >= df['a'].min()
        


 