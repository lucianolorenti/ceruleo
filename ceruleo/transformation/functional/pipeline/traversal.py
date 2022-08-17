from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from ceruleo import CACHE_PATH
from ceruleo.transformation.functional.pipeline.cache_store import (
    CacheStoreType, GraphTraversalCacheMemory, GraphTraversalCacheShelveStore)
from ceruleo.transformation.functional.pipeline.utils import (decode_tuple,
                                                              encode_tuple)
from ceruleo.transformation.functional.transformerstep import TransformerStep


class CachedGraphTraversal:
    """Iterator for a graph nodes. 


    The cache data structures has the following form
    Current Node -> Previous Nodes -> [Transformed Dataset]

    * cache[n]:
        contains a dict with one key for each previous node
    * cache[n][n.previous[0]]
        A list with each element of the dataset transformed in
        up to n.previous[0]


    Parameters:
        
        root_nodes: Initial nodes of the graph
        dataset: Each node visit the dataset
        cache_path: Where to store the cache
        cache_type: Mode for storing the intermediate steps

    """
    def __init__(
        self,
        root_nodes,
        dataset,
        cache_path: Optional[Path] = CACHE_PATH,
        cache_type: CacheStoreType = CacheStoreType.SHELVE,
    ):
        if cache_type == CacheStoreType.SHELVE:
            self.transformed_cache = GraphTraversalCacheShelveStore(cache_path)
        elif cache_type == CacheStoreType.MEMORY:
            self.transformed_cache = GraphTraversalCacheMemory()

        for r in root_nodes:
            for i, df in enumerate(dataset):
                self.transformed_cache[encode_tuple((r, None, i))] = df

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.transformed_cache.close()

    def clear_cache(self):
        self.transformed_cache.claer()

    def state_up_to(self, current_node: TransformerStep, dataset_element: int):

        previous_node = current_node.previous

        if len(previous_node) > 1:
            return [
                self.transformed_cache[encode_tuple((current_node, p, dataset_element))]
                for p in previous_node
            ]
        else:
            if len(previous_node) == 1:
                previous_node = previous_node[0]
            else:
                previous_node = None

            return self.transformed_cache[
                encode_tuple((current_node, previous_node, dataset_element))
            ]

    def clean_state_up_to(self, current_node: TransformerStep, dataset_element: int):

        previous_node = current_node.previous
        for p in previous_node:
            self.transformed_cache[
                encode_tuple((current_node, p, dataset_element))
            ] = None

    def store(
        self,
        next_node: Optional[TransformerStep],
        node: TransformerStep,
        dataset_element: int,
        new_element: pd.DataFrame,
    ):
        self.transformed_cache[
            encode_tuple((next_node, node, dataset_element))
        ] = new_element

    def remove_state(self, nodes: Union[TransformerStep, List[TransformerStep]]):
        if not isinstance(nodes, list):
            nodes = [nodes]
        for n in nodes:
            keys_to_remove = self.get_keys_of(n)
            for k in keys_to_remove:
                self.transformed_cache.pop(k)

    def get_keys_of(self, n):
        return [
            k
            for k in self.transformed_cache.keys()
            if decode_tuple(k)[0] == str(hash(n))
        ]
