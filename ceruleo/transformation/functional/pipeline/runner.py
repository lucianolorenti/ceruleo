import logging
from multiprocessing import JoinableQueue, Manager, Process, Queue
from typing import Iterable, Union

import pandas as pd
from ceruleo.transformation.functional.graph_utils import (
    root_nodes,
    topological_sort_iterator,
)
from ceruleo.transformation.functional.pipeline.cache_store import CacheStoreType
from ceruleo.transformation.functional.pipeline.traversal import CachedGraphTraversal
from ceruleo.transformation.functional.transformerstep import TransformerStep
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def _transform(node, old_element, dataset_element, queue):
    n = node.transform(old_element)
    queue.put((dataset_element, n))


class CachedPipelineRunner:
    """Performs an execution of the transformation graph caching the intermediate results

    Parameters:

        final_step: Last step of the graph
        cache_type: Mode for storing the cache
    """

    def __init__(
        self,
        final_step: TransformerStep,
        cache_type: CacheStoreType = CacheStoreType.SHELVE,
    ):

        self.final_step = final_step
        self.root_nodes = root_nodes(final_step)
        self.cache_type = cache_type

    def _run(
        self,
        dataset: Iterable[pd.DataFrame],
        fit: bool = True,
        show_progress: bool = False,
    ):
        dataset_size = len(dataset)

        with CachedGraphTraversal(
            self.root_nodes, dataset, cache_type=self.cache_type
        ) as cache:
            for node in topological_sort_iterator(self.final_step):
                if isinstance(node, TransformerStep) and fit:
                    if node.prefer_partial_fit:
                        for dataset_element in range(dataset_size):
                            d = cache.state_up_to(node, dataset_element)
                            node.partial_fit(d)
                    else:
                        data = pd.concat(
                            [
                                cache.state_up_to(node, dataset_element)
                                for dataset_element in range(dataset_size)
                            ]
                        )
                        node.fit(data)

                dataset_size = len(dataset)
                # if dataset_size > 1:
                # self._parallel_transform_step(cache, node, dataset_size, show_progress)
                # else:
                self._transform_step(cache, node, dataset_size, show_progress)

            last_state_key = cache.get_keys_of(None)[0]
            return cache.transformed_cache[last_state_key]

    def fit(self, dataset: Iterable[pd.DataFrame], show_progress: bool = False):
        return self._run(dataset, fit=True, show_progress=show_progress)

    def _update_step(self, cache, node, dataset_element, new_element):
        cache.clean_state_up_to(node, dataset_element)

        if len(node.next) > 0:
            for n in node.next:
                cache.store(n, node, dataset_element, new_element)
        else:
            cache.store(None, node, dataset_element, new_element)

    def _parallel_transform_step(
        self, cache: CachedGraphTraversal, node, dataset_size: int, show_progress: bool
    ):
        if show_progress:
            bar = tqdm(range(dataset_size))
            bar.set_description(node.name)
        else:
            bar = range(dataset_size)

        producers = []

        queue = JoinableQueue(dataset_size)
        for dataset_element in range(dataset_size):
            old_element = cache.state_up_to(
                node,
                dataset_element,
            )
            producers.append(
                Process(
                    target=_transform, args=(node, old_element, dataset_element, queue)
                )
            )
        for p in producers:
            p.start()

        for _ in bar:
            dataset_element, new_element = queue.get()
            queue.task_done()
            self._update_step(cache, node, dataset_element, new_element)
        cache.remove_state(node)
        for p in producers:
            p.join()
        queue.join()

    def _transform_step(
        self, cache: CachedGraphTraversal, node, dataset_size: int, show_progress: bool
    ):
        if show_progress:
            bar = tqdm(range(dataset_size))
            bar.set_description(node.name)
        else:
            bar = range(dataset_size)
        try:
            for dataset_element in bar:
                old_element = cache.state_up_to(node, dataset_element)
                new_element = node.transform(old_element)
                self._update_step(cache, node, dataset_element, new_element)
        except Exception as e:
            logger.error(f"There was an error when transforming with {node.name}")
            raise

    def transform(self, df: Union[pd.DataFrame, Iterable[pd.DataFrame]]):
        if isinstance(df, pd.DataFrame):
            return self._run([df], fit=False)
        else:
            return self._run(df, fit=False)
