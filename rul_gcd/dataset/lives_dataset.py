class AbstractLivesDataset:
    def __getitem__(self, i):
        """

        Returns
        -------
        pd.DataFrame
            DataFrame with the data of the life i
        """
        raise NotImplementedError

    @property
    def nlives(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def toPandas(self, proportion=1.0):
        raise NotImplementedError
