import shelve
import uuid
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import pandas as pd
from ceruleo import CACHE_PATH
from ceruleo.dataset.ts_dataset import AbstractTimeSeriesDataset
from ceruleo.transformation.functional.graph_utils import (
    dfs_iterator,
    topological_sort_iterator,
)
from ceruleo.transformation.functional.pipeline.cache_store import CacheStoreType
from ceruleo.transformation.functional.pipeline.runner import CachedPipelineRunner
from ceruleo.transformation.functional.transformerstep import TransformerStep
from sklearn.base import BaseEstimator, TransformerMixin


class Pipeline(BaseEstimator, TransformerMixin):
    """Transformation pipeline

    Parameters:
        final_step: The final step of the transformation
        cache_type: Cache storage mode
    """

    def __init__(self, final_step, cache_type: CacheStoreType = CacheStoreType.MEMORY):
        self.final_step = final_step
        self.fitted_ = False
        self.cache_type = cache_type
        self.runner = CachedPipelineRunner(final_step, cache_type)

    def find_node(
        self, name: str
    ) -> Union[List[TransformerStep], TransformerStep, None]:
        """Find a transformation node given a name

        Parameters:
            name: Name of the step to find

        Returns:

            steps: Steps located in the pipeline.

        """
        matches = []
        for node in dfs_iterator(self.final_step):
            if node.name == name:
                matches.append(node)

        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            return matches
        else:
            return None

    def fit(
        self,
        dataset: Union[AbstractTimeSeriesDataset, pd.DataFrame],
        show_progress: bool = False,
    ):
        """Fit a pipeline using a dataset

        The CachedPipelineRunner is called to fit

        Parameters:

            dataset: A dataset of a run-to-failure cycle
            show_progress: Wether to show the progress when fitting

        Returns:
            s : Pipeline
        """
        if isinstance(dataset, pd.DataFrame):
            dataset = [dataset]
        c = self.runner.fit(dataset, show_progress=show_progress)
        self.column_names = c.columns
        self.fitted_ = True

        return self

    def partial_fit(
        self,
        dataset: Union[AbstractTimeSeriesDataset, pd.DataFrame],
        show_progress: bool = False,
    ):
        self.fit(dataset, show_progress=show_progress)

    def transform(self, df: Union[pd.DataFrame, Iterable[pd.DataFrame]]):
        """Transform a run-to-cycle failure or a dataset

        The CachedPipelineRunner is called to transform

        Parameters:

            df: A dataset of a run-to-failure cycle

        Returns:
            s : list of data frames
        """
        return self.runner.transform(df)

    def description(self):
        data = []
        for node in topological_sort_iterator(self):
            data.append(node.description())
        return data

    def get_params(self, deep: bool = False):
        params = {"cache_type": self.cache_type, "final_step": self.final_step}
        if deep:
            for node in topological_sort_iterator(self):
                p = node.get_params(deep)
                for k in p.keys():
                    params[f"{node.name}__{k}"] = p[k]
        return params


def make_pipeline(
    *steps, cache_type: CacheStoreType = CacheStoreType.MEMORY
) -> Pipeline:
    """Build a pipeline

    Example:

        make_pipeline(
            ByNameFeatureSelector(features=FEATURES),
            Clip(lower=-2, upper=2),
            IndexMeanResampler(rule='500s')
        )

    Parameters:

        steps: List of steps
        cache_type: Where to store the pipeline intermediate steps

    Returns:

        TemporisPipeline: The created pipeline
    """
    step = steps[0]
    for next_step in steps[1:]:
        step = next_step(step)

    return Pipeline(step, cache_type=cache_type)
