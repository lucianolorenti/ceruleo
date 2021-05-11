from typing import Union
from rul_pm.iterators.batcher import Batcher
import numpy as np
from rul_pm.iterators.iterators import WindowedDatasetIterator
from rul_pm.transformation.pipeline import LivesPipeline

def true_values(dataset_iterator: Union[WindowedDatasetIterator, Batcher]) -> np.array:
    """Obtain the true RUL of the dataset after the transformation

    Parameters
    ----------
    dataset_iterator : Union[WindowedDatasetIterator, Batcher]
        Iterator of the dataset

    Returns
    -------
    np.array
         target values after the transformation
    """
    if isinstance(dataset_iterator, Batcher):
        dataset_iterator = dataset_iterator.iterator
    orig_transformer = dataset_iterator.transformer.transformerX
    dataset_iterator.transformer.transformerX = LivesPipeline(
        steps=[('empty', 'passthrough')])        
    d =  np.concatenate([y for _, y, _ in dataset_iterator])
    dataset_iterator.transformer.transformerX = orig_transformer
    return d