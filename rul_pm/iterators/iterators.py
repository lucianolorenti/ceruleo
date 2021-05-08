import logging
import random
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from pandas.core.algorithms import isin
from rul_pm.dataset.lives_dataset import AbstractLivesDataset
from rul_pm.transformation.transformers import Transformer
from rul_pm.utils.lrucache import LRUDataCache
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm

CACHE_SIZE = 30

logger = logging.getLogger(__name__)

SAMPLE_WEIGHT_PROPORTIONAL_TO_LENGTH = 'proportional_to_length'
SAMPLE_WEIGHT_RUL_INV = 'rul_inv'
SAMPLE_WEIGHT_EQUAL = 'equal'

VALID_SAMPLE_WEIGHTS = [SAMPLE_WEIGHT_PROPORTIONAL_TO_LENGTH,
                        SAMPLE_WEIGHT_RUL_INV,
                        SAMPLE_WEIGHT_EQUAL]


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
        if not transformer.fitted_:
            raise ValueError('Transformer not fitted')
        self.dataset = dataset
        self.shuffle = shuffle
        self.transformer = transformer
        self.cache = LRUDataCache(cache_size)

        try:
            check_is_fitted(transformer)
        except NotFittedError:
            self.transformer.fit(dataset)

    def _load_data(self, life) -> pd.DataFrame:
        """
        Return a DataFrame with the contents of the life

        Parameters
        ----------
        life : any
               The life identifiers
        """
        if life not in self.cache.data:
            data = self.dataset[life]
            X, y, metadata = self.transformer.transform(data)
            self.cache.add(life, (X, y, metadata))
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

    def __len__(self) -> int:
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
                Determine wether the window should include points in wich
                the RUL does not have gaps larger than the parameter

    sample_weight: str
                   Choose the weight of each sample. Possible values are
                   'equal', 'proportional_to_length'.
                   If 'equal' is choosed, each sample weights 1,
                   if 'proportional_to_length' is choosed, each sample
                   weight 1 / life length

    """

    def __init__(self,
                 dataset: AbstractLivesDataset,
                 window_size: int,
                 transformer: Transformer,
                 step: int = 1,
                 output_size: int = 1,
                 shuffle: Union[str, bool] = False,
                 cache_size: int = CACHE_SIZE,
                 evenly_spaced_points: Optional[int] = None,
                 sample_weight: Union[
                     str,
                     Callable[[pd.DataFrame], float]] = SAMPLE_WEIGHT_EQUAL,
                 add_last: bool = True,
                 discard_threshold: Optional[float] = None):
        super().__init__(dataset, transformer, shuffle, cache_size=cache_size)
        self.evenly_spaced_points = evenly_spaced_points
        self.window_size = window_size
        self.step = step
        self.shuffle = shuffle
        if isinstance(sample_weight, str):
            if sample_weight not in VALID_SAMPLE_WEIGHTS:
                raise ValueError(
                    f'Invalid sample_weight parameter. Valid values are {VALID_SAMPLE_WEIGHTS}')
        elif not callable(sample_weight):
            raise ValueError('sample_weight should be an string or a callable')

        self.sample_weight = sample_weight
        self.discard_threshold = discard_threshold
        self.orig_lifes, self.orig_elements, self.sample_weights = self._windowed_element_list()
        self.lifes, self.elements = self.orig_lifes, self.orig_elements
        self.i = 0
        self.output_size = output_size
        self.add_last = add_last

    def _sample_weight(self, y, i: int, metadata):
        if isinstance(self.sample_weight, str):
            if self.sample_weight == SAMPLE_WEIGHT_EQUAL:
                return 1
            elif self.sample_weight == SAMPLE_WEIGHT_RUL_INV:
                return ((1 / (y[i]+1)))
            elif self.sample_weight == SAMPLE_WEIGHT_PROPORTIONAL_TO_LENGTH:
                return (1 / y[0])
        elif callable(self.sample_weight):
            return self.sample_weight(y, i, metadata)

    def _windowed_element_list(self):
        def window_evenly_spaced(y, i):
            w = y[i-self.window_size:i+1].diff().dropna().abs()
            return np.all(w <= self.evenly_spaced_points)

        
        if self.discard_threshold is not None:
            def should_discard(y, i): 
                return y[i] > self.discard_threshold
        else:
            def should_discard(y, i): return False

        olifes = []
        oelements = []
        sample_weights = []
        logger.debug('Computing windows')
        for life in range(self.dataset.nlives):
            life_data = self.dataset[life]
            y = self.transformer.transformY(life_data)
            metadata = self.transformer.transformMetadata(life_data)

            list_ranges = range(self.window_size-1, y.shape[0], self.step)
            if self.evenly_spaced_points is not None:
                is_valid_point = window_evenly_spaced
            else:
                def is_valid_point(y, i): return True

            list_ranges = [
                i for i in list_ranges if is_valid_point(y, i) and not should_discard(y, i)
            ]

            for i in list_ranges:
                olifes.append(life)
                oelements.append(i)
                sample_weights.append(self._sample_weight(y, i, metadata))

        return olifes, oelements, sample_weights

    def _shuffle(self):
        """
        Shuffle the window elements
        """
        if not self.shuffle:
            return
        valid_shuffle = ((not self.shuffle)
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
            df = df.sort_values(by=['elements'], ascending=False)
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
        X, y, _ = self._load_data(life)
        window = windowed_signal_generator(
            X, y, timestamp, self.window_size, self.output_size, self.add_last)
        # return *window, [sample_weight]
        return window[0], window[1], [self.sample_weights[i]]

    def at_end(self):
        return self.i == len(self.elements)

    def __next__(self):
        if self.at_end():
            raise StopIteration
        ret = self.__getitem__(self.i)
        self.i += 1
        return ret

    def get_data(self):        
        N_points = len(self)
        dimension = self.window_size*self.transformer.n_features
        X = np.zeros(
            (N_points, dimension), 
            dtype=np.float32)
        y = np.zeros((N_points, self.output_size), dtype=np.float32)
        sample_weight = np.zeros(N_points, dtype=np.float32)

        for i, (X_, y_, sample_weight_) in enumerate(self):
            X[i, :] = X_.flatten()
            y[i, :] = y_.flatten()
            sample_weight[i] = sample_weight_[0]
        return X, y, sample_weight



def windowed_signal_generator(signal_X, signal_y, i: int, window_size: int, output_size: int = 1,  add_last: bool = True):
    """
    Return a lookback window and the value to predict.

    Parameters
    ----------
    signal_X:
             Matrix of size (life_length, n_features) with the information of the life
    signal_y:
             Target feature of size (life_length)
    i: int
       Position of the value to predict

    window_size: int
                 Size of the lookback window

    output_size: int
                 Number of points of the target

    add_last: bool


    Returns
    -------
    tuple (np.array, float)
    """
    initial = max(i - window_size+1, 0)
    signal_X_1 = signal_X[initial:i + (1 if add_last else 0), :]
    if len(signal_y.shape) == 1:

        signal_y_1 = signal_y[i:min(i+output_size, signal_y.shape[0])]

        if signal_y_1.shape[0] < output_size:
            padding = np.zeros(output_size - signal_y_1.shape[0])
            signal_y_1 = np.hstack((signal_y_1, padding))
        signal_y_1 = np.expand_dims(signal_y_1, axis=1)
    else:
        signal_y_1 = signal_y[i:min(i+output_size, signal_y.shape[0]), :]

        if signal_y_1.shape[0] < output_size:
            padding = np.zeros(
                ((output_size - signal_y_1.shape[0]), signal_y_1.shape[1]))
            signal_y_1 = np.concatenate((signal_y_1, padding), axis=0)

    if signal_X_1.shape[0] < window_size:

        signal_X_1 = np.vstack((
            np.zeros((
                window_size - signal_X_1.shape[0],
                signal_X_1.shape[1])),
            signal_X_1))

    return (signal_X_1, signal_y_1)
