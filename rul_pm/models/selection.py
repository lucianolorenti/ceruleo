import math

import numpy as np
from rul_pm.dataset.lives_dataset import AbstractLivesDataset
from tqdm.auto import tqdm

from sklearn.base import clone
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.model_selection._split import _BaseKFold


class FoldedDataset(AbstractLivesDataset):
    def __init__(self, dataset: AbstractLivesDataset, indices: list):
        self.dataset = dataset
        self.indices = indices

    @property
    def nlives(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        """

        Returns
        -------
        pd.DataFrame
            DataFrame with the data of the life i
        """
        return self.dataset[self.indices[i]]


class RULGridSearchCV:
    def __init__(self, model, params: dict, folds_genereator: _BaseKFold):
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
                r = model.fit(train_dataset, validation_dataset,
                              verbose=verbose)
                params_results.append(r)
            self.param_list.append(p)
            self.results.append(params_results)


class RULGridSearchPredefinedValidation:
    def __init__(self, model, params: dict):
        self.params = params
        self.model = model
        self.param_list = []
        self.results = []

    def fit(self, dataset_train, dataset_validation, verbose=1):
        self.param_list = []
        self.results = []

        for p in tqdm(list(ParameterGrid(self.params))):
            model = clone(self.model)
            model.reset()
            model.set_params(**p)
            r = model.fit(dataset_train, dataset_validation, verbose=verbose)
            self.param_list.append(p)
            self.results.append([r])


class GeneticAlgorithmFeatureSelection:
    def __init__(self, fitness_fun, population_size, max_iter: int):
        self.population_size = population_size
        self.fitness_fun = fitness_fun
        self.max_iter = max_iter

    def init_population(self, number_of_features):
        return np.array([
            [math.ceil(e) for e in pop]
            for pop in (np.random.rand(self.population_size, number_of_features)-0.5)]), np.zeros((2, number_of_features))-1

    def single_point_crossover(self, population):
        r, c, n = population.shape[0], population.shape[1], np.random.randint(
            1, population.shape[1])
        for i in range(0, r, 2):
            population[i], population[i+1] = np.append(
                population[i][0:n], population[i+1][n:c]), np.append(population[i+1][0:n], population[i][n:c])
        return population

    def flip_mutation(self, population):
        return population.max() - population

    def random_selection(self, population):
        r = population.shape[0]
        new_population = population.copy()
        for i in range(r):
            new_population[i] = population[np.random.randint(0, r)]
        return new_population

    def get_fitness(self, train_dataset, val_dataset, feature_list, population):
        fitness = []
        for i in range(population.shape[0]):
            columns = [feature_list[j]
                       for j in range(population.shape[1]) if population[i, j] == 1]
            fitness.append(self.fitness_fun(
                train_dataset, val_dataset, feature_list))
        return fitness

    def run(self, train_dataset, validation_dataset, feature_list, ):

        c = len(feature_list)

        population, memory = self.init_population(c)
        #population, memory = self.replace_duplicate(population, memory)

        fitness = self.get_fitness(
            train_dataset, validation_dataset, feature_list, population)

        optimal_value = max(fitness)
        optimal_solution = population[np.where(fitness == optimal_value)][0]

        for i in range(self.max_iter):
            population = self.random_selection(population)
            population = self.single_point_crossover(population)
            if np.random.rand() < 0.3:
                population = self.flip_mutation(population)

            #population, memory = replace_duplicate(population, memory)

            fitness = self.get_fitness(
                train_dataset, validation_dataset, feature_list, population)

            if max(fitness) > optimal_value:
                optimal_value = max(fitness)
                optimal_solution = population[np.where(
                    fitness == optimal_value)][0]
                print(optimal_solution, optimal_value)

        return optimal_solution, optimal_value
