from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold
from rul_pm.dataset.lives_dataset import AbstractLivesDataset
from tqdm.auto import tqdm


class FoldedDataset(AbstractLivesDataset):
    def __init__(self, dataset:AbstractLivesDataset, indices: list):
        self.dataset = dataset 
        self.indices = indices 

    @property
    def nlives(self):
        return len(self.indices)

    def __getitem__(self, i:int):
        """

        Returns
        -------
        pd.DataFrame
            DataFrame with the data of the life i
        """
        return self.dataset[self.indices[i]]


class RULGridSearchCV:
    def __init__(self, model, params:dict, folds_genereator:_BaseKFold):
        self.params = params
        self.model = model
        self.folds_genereator = folds_genereator
        self.param_list = []
        self.results = []


    def fit(self, dataset, verbose=1):
        self.param_list = []
        self.results = []

        for p in tqdm(list(ParameterGrid(self.params))):
            model = clone(self.model)
            model.reset()
            model.set_params(**p)
            params_results = []
            for train, validation in tqdm(self.folds_genereator.split(dataset)):
                train_dataset = FoldedDataset(dataset, train)
                validation_dataset = FoldedDataset(dataset, validation)
                r = model.fit(train_dataset, validation_dataset, verbose=verbose)
                params_results.append(r)
            self.param_list.append(p)
            self.results.append(params_results)

        


