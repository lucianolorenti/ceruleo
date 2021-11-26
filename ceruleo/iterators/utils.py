

from numpy.lib.arraysetops import isin
from ceruleo.dataset.transformed import TransformedDataset
from ceruleo.dataset.ts_dataset import AbstractTimeSeriesDataset
from typing import Optional, Union

import numpy as np
from ceruleo.dataset.utils import iterate_over_target
from ceruleo.iterators.batcher import Batcher
from ceruleo.iterators.iterators import WindowedDatasetIterator

try:
    import tensorflow as tf 
    TENSORFLOW = True
except:
    TENSORFLOW = False


def true_values(
    dataset: Union[WindowedDatasetIterator, Batcher, AbstractTimeSeriesDataset],
    target_column: Optional[str] = None
) -> np.array:
    """Obtain the true RUL of the dataset after the transformation

    Parameters:

        dataset:  Iterator of the dataset

    Returns:
    
        true_RUL: target values after the transformation
    """
    from ceruleo.transformation.functional.transformers import TransformerIdentity
    if isinstance(dataset, Batcher):
        dataset = dataset.iterator

    elif isinstance(dataset, AbstractTimeSeriesDataset) and not isinstance(dataset, TransformedDataset):
        if target_column is None:
            if not hasattr(dataset, 'rul_column'):
                raise ValueError('Please provide a target column to access')
            else:
                target_column = dataset.rul_column
        return np.squeeze(np.concatenate([y for y in iterate_over_target(dataset)]))
    else:
        if TENSORFLOW:
            if isinstance(dataset, tf.data.Dataset):
                dataset = dataset.as_numpy_iterator()
        return np.squeeze(np.concatenate([y for _, y, _ in dataset]))
    raise ValueError("Ivalid dataset used")
