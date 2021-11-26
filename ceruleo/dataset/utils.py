from typing import Union

from ceruleo.dataset.transformed import TransformedDataset
from ceruleo.dataset.ts_dataset import AbstractLivesDataset


def iterate_over_features(ds: Union[TransformedDataset, AbstractLivesDataset]):
    """Helper function to iterate over the features in a dataset
    Example:
        for X, y, metadata in df:
            pass
        for X in iterate_over_features(ds):
            pass
    Parameters:
        ds: The dataset
    Returns:

        it: The iterator
    """
    if isinstance(ds, TransformedDataset):
        return map(lambda x: x[0], ds)
    else:
        return ds


def iterate_over_target(ds: Union[TransformedDataset, AbstractLivesDataset]):
    """Helper function to iterate over the RUL target in a dataset
    Example:

        for X, y, metadata in df:
            pass

        for y in iterate_over_target(ds):
            pass

    Parameters:
        ds: The dataset
        
    Returns:

        it: The iterator
    """
    if isinstance(ds, TransformedDataset):
        return map(lambda x: x[1], ds)
    elif hasattr(ds, 'rul_column'):
        return map(lambda x: x[ds.rul_column], ds)
    else:
        raise ValueError('Invalid dataset type used')
    

def iterate_over_features_and_target(ds: Union[TransformedDataset, AbstractLivesDataset]):
    """Helper function to iterate over the features and RUL target in a dataset
    Example:

        for X, y, metadata in df:
            pass

        for X, yy in iterate_over_features_and_target(ds):
            pass

    Parameters:
        ds: The dataset
        
    Returns:

        it: The iterator
    """
    if isinstance(ds, TransformedDataset):
        return map(lambda x: (x[0], x[1]), ds)
    elif hasattr(ds, 'rul_column'):
        return map(lambda x: (x, x[ds.rul_column]), ds)
    else:
        raise ValueError('Invalid dataset type used')