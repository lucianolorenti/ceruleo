import shelve
import uuid
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import pandas as pd

from sklearn.base import TransformerMixin
from temporis import CACHE_PATH
from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset
from temporis.transformation.functional.graph_utils import (
    dfs_iterator,
    edges,
    nodes,
    root_nodes,
    topological_sort_iterator,
)
from temporis.transformation.functional.pipeline.cache_store import CacheStoreType
from temporis.transformation.functional.pipeline.runner import CachedPipelineRunner
from temporis.transformation.functional.transformerstep import TransformerStep
from tqdm.auto import tqdm
import shutil








class TemporisPipeline(TransformerMixin):
    def __init__(self, final_step, cache_type:CacheStoreType = CacheStoreType.SHELVE):
        self.final_step = final_step
        self.fitted_ = False
        self.runner = CachedPipelineRunner(final_step, cache_type)

    def find_node(
        self, name: str
    ) -> Union[List[TransformerStep], TransformerStep, None]:
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
        if isinstance(dataset, pd.DataFrame):
            dataset = [dataset]
        c = self.runner.fit(dataset, show_progress=show_progress)
        self.column_names = c.columns
        self.fitted_ = True

        return self


    def partial_fit(self,
        dataset: Union[AbstractTimeSeriesDataset, pd.DataFrame],
        show_progress: bool = False):
        self.fit(dataset, show_progress=show_progress)

    def transform(self, df: Union[pd.DataFrame, Iterable[pd.DataFrame]]):
        return self.runner.transform(df)

    def description(self):
        data = []
        for node in topological_sort_iterator(self):
            data.append(node.description())
        return data

    


