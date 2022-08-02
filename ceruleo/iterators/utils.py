

from numpy.lib.arraysetops import isin
from ceruleo.dataset.transformed import TransformedDataset
from ceruleo.dataset.ts_dataset import AbstractTimeSeriesDataset
from typing import Optional, Union

import numpy as np
from ceruleo.iterators.batcher import Batcher
from ceruleo.iterators.iterators import WindowedDatasetIterator

try:
    import tensorflow as tf 
    TENSORFLOW = True
except:
    TENSORFLOW = False


def true_values(
    dataset_iterator: Union[WindowedDatasetIterator, Batcher, AbstractTimeSeriesDataset],
    target_column: Optional[str] = None
) -> np.array:
    """Obtain the true RUL of the dataset after the transformation

    Parameters
    ----------
    dataset_iterator : Union[WindowedDatasetIterator, Batcher, AbstractTimeSeriesDataset]
        Iterator of the dataset

    Returns
    -------
    np.array
         target values after the transformation
    """
    from ceruleo.transformation.functional.transformers import TransformerIdentity
    if isinstance(dataset_iterator, Batcher):
        dataset_iterator = dataset_iterator.iterator

    elif isinstance(dataset_iterator, AbstractTimeSeriesDataset) and not isinstance(dataset_iterator, TransformedDataset):
        if target_column is None:
            raise ValueError('Please provide a target column to access')
        ti = TransformerIdentity(target_column)
        ti.fit([dataset_iterator[0]])
        dataset_iterator = WindowedDatasetIterator(
            dataset_iterator.map(ti), window_size=1
        )
    if TENSORFLOW:
        if isinstance(dataset_iterator, tf.data.Dataset):
            dataset_iterator = dataset_iterator.as_numpy_iterator()
    d = np.squeeze(np.concatenate([y for _, y, _ in dataset_iterator]))
    return d
