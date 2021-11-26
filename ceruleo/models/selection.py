import math

import numpy as np
from ceruleo.dataset.lives_dataset import AbstractLivesDataset, FoldedDataset
from tqdm.auto import tqdm

from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection._split import _BaseKFold
import multiprocessing


class RULScorerWrapper:
    def __init__(self, scorer):
        from sklearn.metrics import get_scorer

        self.scorer = get_scorer(scorer)

    def __call__(self, estimator, X, y=None):
        y_true = estimator.true_values(X)
        return self.scorer(estimator, X, y_true)

class Fitter:
    def __init__(self, model, dataset, folds_genereator, fit_kwargs):
        self.model = model
        self.folds_genereator = folds_genereator
        self.dataset = dataset 
        self.fit_kwargs = fit_kwargs

    def __call__(self, params):
        i, params  = params
        model = clone(self.model)
        model.reset()
        model.set_params(**params)
        params_results = []
        for train, validation in self.folds_genereator.split(self.dataset):
            train_dataset = FoldedDataset(self.dataset, train)
            validation_dataset = FoldedDataset(self.dataset, validation)
            model.fit(train_dataset, validation_dataset, **self.fit_kwargs)
            y_pred = model.predict(validation_dataset)
            y_true = model.true_values(validation_dataset)

            params_results.append({"mse": np.mean((y_pred - y_true) ** 2)})
        return (params, params_results)


class RULGridSearchCV:
    def __init__(self, model, params: dict, folds_genereator: _BaseKFold):
        self.params = params
        self.model = model
        self.folds_genereator = folds_genereator
        self.param_list = []
        self.results = []

    def fit(self, dataset, **fit_kwargs):
        pool = multiprocessing.Pool(6)
        self.param_list, self.results = zip(
            *pool.map(
                Fitter(self.model, dataset, self.folds_genereator, fit_kwargs),
                enumerate(list(ParameterGrid(self.params))),
            )
        )


class RULGridSearchPredefinedValidation:
    def __init__(self, model, params: dict):
        self.params = params
        self.model = model
        self.param_list = []
        self.results = []

    def fit(self, dataset_train, dataset_validation, **fit_kwargs):
        self.param_list = []
        self.results = []

        for p in tqdm(list(ParameterGrid(self.params))):
            model = clone(self.model)
            model.reset()
            model.set_params(**p)
            r = model.fit(dataset_train, dataset_validation, **fit_kwargs)
            self.param_list.append(p)
            self.results.append([r])


class GeneticAlgorithmFeatureSelection:
    def __init__(self, fitness_fun, population_size, max_iter: int):
        self.population_size = population_size
        self.fitness_fun = fitness_fun
        self.max_iter = max_iter

    def init_population(self, number_of_features):
        return (
            np.array(
                [
                    [math.ceil(e) for e in pop]
                    for pop in (
                        np.random.rand(self.population_size, number_of_features) - 0.5
                    )
                ]
            ),
            np.zeros((2, number_of_features)) - 1,
        )

    def single_point_crossover(self, population):
        r, c, n = (
            population.shape[0],
            population.shape[1],
            np.random.randint(1, population.shape[1]),
        )
        for i in range(0, r, 2):
            population[i], population[i + 1] = (
                np.append(population[i][0:n], population[i + 1][n:c]),
                np.append(population[i + 1][0:n], population[i][n:c]),
            )
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
            # columns = [feature_list[j]
            #           for j in range(population.shape[1]) if population[i, j] == 1]
            fitness.append(self.fitness_fun(train_dataset, val_dataset, feature_list))
        return fitness

    def run(
        self,
        train_dataset,
        validation_dataset,
        feature_list,
    ):

        c = len(feature_list)

        population, memory = self.init_population(c)
        # population, memory = self.replace_duplicate(population, memory)

        fitness = self.get_fitness(
            train_dataset, validation_dataset, feature_list, population
        )

        optimal_value = max(fitness)
        optimal_solution = population[np.where(fitness == optimal_value)][0]

        for i in range(self.max_iter):
            population = self.random_selection(population)
            population = self.single_point_crossover(population)
            if np.random.rand() < 0.3:
                population = self.flip_mutation(population)

            # population, memory = replace_duplicate(population, memory)

            fitness = self.get_fitness(
                train_dataset, validation_dataset, feature_list, population
            )

            if max(fitness) > optimal_value:
                optimal_value = max(fitness)
                optimal_solution = population[np.where(fitness == optimal_value)][0]

        return optimal_solution, optimal_value
