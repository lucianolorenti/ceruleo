import logging
import pickle
import random
from typing import Optional, Union

import numpy as np
import pandas as pd
from rul_pm.dataset.lives_dataset import AbstractLivesDataset
from rul_pm.iterators.utils import windowed_signal_generator
from rul_pm.transformation.transformers import Transformer
from rul_pm.utils.lrucache import LRUDataCache
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm

CACHE_SIZE = 30

logger = logging.getLogger(__name__)


class DatasetIterator:
    """

    Each life is stored in a LRU cache in-memory and should have
    an unique identifier. A number is sufficient.


    Parameters
    ---------
    dataset     : AbstractLivesDataset
                  Dataset with the lives data
    transformer : Transformer
                  Transformer to apply to each life
    shuffle     : Union[bool, str] default: False
                  If the data returned for the iterator should be shuffle.
                  The possible values depends on the iterator
    cache_size  : int. default: CACHE_SIZE
                  Size of the LRU cache where the lives are stored
    """

    def __init__(self,
                 dataset: AbstractLivesDataset,
                 transformer: Transformer,
                 shuffle: Union[bool, str] = False,
                 cache_size: int = CACHE_SIZE):

        self.dataset = dataset
        self.shuffle = shuffle
        self.transformer = transformer
        self.cache = LRUDataCache(cache_size)

        try:
            check_is_fitted(transformer)
        except NotFittedError:
            self.transformer.fit(dataset)

    def _load_data(self, life):
        """
        Return a DataFrame with the contents of the life

        Parameters
        ----------
        life : any
               The life identifiers
        """
        if life not in self.cache.data:
            data = self.dataset[life]
            X, y = self.transformer.transform(data)
            self.cache.add(life, (X, y))
        return self.cache.get(life)


class LifeDatasetIterator(DatasetIterator):
    """
    Iteratres over the whole set of lives.
    Each element returned by the iterator is the complete life of the equipment

    Parameters
    ----------
    dataset     : AbstractLivesDataset
                  Dataset with the lives data
    transformer : Transformer
                  Transformer to apply to each life
    shuffle     : default: False
                  If the data returned for the iterator should be shuffle.
                  The possible values depends on the iterator
    """

    def __init__(self,
                 dataset: AbstractLivesDataset,
                 transformer: Transformer,
                 shuffle: Union[bool, str] = False):
        super().__init__(dataset, transformer, shuffle)
        self.elements = list(range(0, len(self.dataset)))
        self.i = 0

    def __iter__(self):
        self.i = 0
        if self.shuffle:
            random.shuffle(self.elements)
        return self

    def __len__(self):
        return len(self.dataset)

    def __next__(self):
        if self.i == len(self):
            raise StopIteration
        current_life = self.elements[self.i]
        self.i += 1
        return self._load_data(current_life)


class WindowedDatasetIterator(DatasetIterator):
    """
    Iteratres over the whole set of lives.
    Each element returned by the iterator is the complete life of the equipment

    Parameters
    ----------
    dataset: AbstractLivesDataset
    window_size: int,
    transformer: Transformer
    step: int = 1
    shuffle : [False, 'signal', 'life', 'all', 'signal_life')
              How to shuffle the windows.

        * 'signal': Each point of the life is shuffled, but the lives
                    are kept in order

                    Iteration 1: | Life 1 | Life 1 | Life 1 | Life 2 | Life 2 | Life 2
                                    |   3    |  1     |  2     |   2    |   3    |   1
                    Iteration 2: | Life 1 | Life 1 | Life 1 | Life 2 | Life 2 | Life 2
                                    |   1    |  3     |  2     |   3    |   2    |   1


        * 'life': Lives are shuffled, but each point inside the life kept
                    its order

                    Iteration 1: | Life 1 | Life 1 | Life 1 | Life 2 | Life 2 | Life 2 |
                                    |   1    | 2      |  3     |   1    |   2    |   3    |
                    Iteration 2: | Life 2 | Life 2 | Life 2 | Life 1 | Life 1 | Life 1 |
                                    |   1    |  2     |  3     |   1    |   2    |   3    |

        * 'signal_life': Each point in the life is shuffled, and the life
                            order are shuffled also.

                    Iteration 1: | Life 1 | Life 1 | Life 1 | Life 2 | Life 2 | Life 2
                                    |   3    | 2      |  1     |   1    |   3    |   2
                    Iteration 2: | Life 2 | Life 2 | Life 2 | Life 1 | Life 1 | Life 1
                                    |   3    |  1     |  2     |   3    |   1    |   2


            * 'all': Everythin is shuffled

                        Iteration 1: | Life 1 | Life 2 | Life 2 | Life 1 | Life 1 | Life 2
                                     |   3    | 2      |  1     |   1    |   2    |   3

    cache_size: int = CACHE_SIZE
                Size of the LRU Cache. The size indicates the number of lives to store

    evenly_spaced_points: int
                Determine wether 
    """

    def __init__(self,
                 dataset: AbstractLivesDataset,
                 window_size: int,
                 transformer: Transformer,
                 step: int = 1,
                 output_size: int = 1,
                 shuffle: Union[str, bool] = False,
                 cache_size: int = CACHE_SIZE,
                 evenly_spaced_points: Optional[int] = None):
        super().__init__(dataset, transformer, shuffle, cache_size=cache_size)
        self.evenly_spaced_points = evenly_spaced_points
        self.window_size = window_size
        self.step = step
        self.shuffle = shuffle
        self.orig_lifes, self.orig_elements = self._windowed_element_list()
        self.lifes, self.elements = self.orig_lifes, self.orig_elements
        self.i = 0
        self.output_size = output_size

    def _windowed_element_list(self):
        def window_evenly_spaced(y, i):
            w = y[i-self.window_size:i+1].diff().dropna().abs()
            return np.all(w <= self.evenly_spaced_points)

        olifes = []
        oelements = []
        logger.info('Computing windows')
        for life in tqdm(range(self.dataset.nlives)):

            _, y = self._load_data(life)
            list_ranges = range(self.window_size-1, y.shape[0], self.step)
            if self.evenly_spaced_points is not None:
                is_valid_point = window_evenly_spaced
            else:
                def is_valid_point(y, i): return True
            list_ranges = [
                i for i in list_ranges if is_valid_point(y, i)
            ]

            for i in list_ranges:
                olifes.append(life)
                oelements.append(i)

        return olifes, oelements

    def _shuffle(self):
        """
        Shuffle the window elements
        """
        if not self.shuffle:
            return
        valid_shuffle = ((self.shuffle == False)
                         or (self.shuffle
                             in ('signal', 'life', 'all', 'signal_life', 'ordered')))
        df = pd.DataFrame({
            'life': self.orig_lifes,
            'elements': self.orig_elements
        })
        if not valid_shuffle:
            raise ValueError(
                "shuffle parameter invalid. Valid values are: False, 'signal', 'life', 'all' 'signal_life', 'ordered'"
            )
        if self.shuffle == 'signal':
            groups = [d.sample(frac=1, axis=0) for _, d in df.groupby('life')]
            df = pd.concat(groups).reset_index(drop=True)
        elif self.shuffle == 'life':
            groups = [d for _, d in df.groupby('life')]
            random.shuffle(groups)
            df = pd.concat(groups).reset_index(drop=True)
        elif self.shuffle == 'signal_life':
            groups = [d.sample(frac=1, axis=0) for _, d in df.groupby('life')]
            random.shuffle(groups)
            df = pd.concat(groups).reset_index(drop=True)
        elif self.shuffle == 'all':
            df = df.sample(frac=1, axis=0)

        elif self.shuffle == 'ordered':
            df = df.sort_values(by=['elements'])
        self.lifes = df['life'].values.tolist()
        self.elements = df['elements'].values.tolist()

    def __iter__(self):
        self.i = 0
        self._shuffle()
        return self

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, i: int):
        (life, timestamp) = (self.lifes[i], self.elements[i])
        X, y = self._load_data(life)
        return windowed_signal_generator(X, y, timestamp, self.window_size, self.output_size)

    def at_end(self):
        return self.i == len(self.elements)

    def __next__(self):
        if self.at_end():
            raise StopIteration
        ret = self.__getitem__(self.i)
        self.i += 1
        return ret

    def toArray(self):
        XX = []
        yy = []
        for X, y in tqdm(self):
            XX.append(np.expand_dims(X, axis=0))
            yy.append(y)
        return np.concatenate(XX, axis=0), np.array(yy)
