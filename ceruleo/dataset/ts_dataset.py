from collections.abc import Iterable
from pathlib import Path

try: 
    from types import EllipsisType
except:
    EllipsisType = type(Ellipsis)
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import tensorflow as tf

    TENSORFLOW_ENABLED = True
except:
    TENSORFLOW_ENABLED = False
from tqdm.auto import tqdm
from abc import ABC, abstractmethod, abstractproperty


class DatasetIterator:
    def __init__(self, dataset):
        self.dataset = dataset
        self.i = 0

    def __next__(self):
        if self.i == self.dataset.n_time_series:
            raise StopIteration
        a = self.dataset[self.i]
        self.i += 1
        return a


class AbstractPDMDataset(ABC):
    def __init__(self):
        self._common_features = None
        self._durations = None

    def __iter__(self):
        return DatasetIterator(self)

    @abstractproperty
    @property
    def n_time_series(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_time_series(self, i: int) -> pd.DataFrame:
        """

        Returns:
            df: DataFrame with the data of the life i
        """
        raise NotImplementedError

    def number_of_samples_of_time_series(self, i: int) -> int:
        return self[i].shape[0]

    @abstractproperty
    def rul_column(self) -> str:
        raise NotImplementedError

    def duration(self, life: pd.DataFrame) -> float:
        return life[self.rul_column].max()

    def number_of_samples(self) -> List[int]:
        return [
            self.number_of_samples_of_time_series(i) for i in tqdm(range(len(self)))
        ]

    def duration(self, life: pd.DataFrame) -> float:
        """Obtain the duration of the time-series

        Parameters:
            life: The input life

        Returns:
            Duration of the life
        """
        v = life.index
        return v.max() - v.min()

    def durations(self, show_progress: bool = False) -> List[float]:
        """
        Obtain the length of each life

        Return:
            List of durations
        """
        if self._durations is None:
            if show_progress:
                iterator = tqdm(self)
            else:
                iterator = self
            self._durations = [self.duration(life) for life in iterator]
            # [self.rul_column].iloc[0]
        return self._durations

    def __call__(self, i):
        return self[i]

    def get_features_of_life(self, i: int) -> pd.DataFrame:
        return self[i]


    def __getitem__(
        self, i: Union[int, Iterable]
    ) -> Union[pd.DataFrame, "FoldedDataset"]:
        """Obtain a time-series or an splice of the dataset using a FoldedDataset

        Parameters:
            i: If the parameter is an int it will return a pd.DataFrame with the i-th time-series.
                If the parameter is a list of int it will return a FoldedDataset with the
                time-series whose id are present in the list

        Raises:
            ValueError: When the list does not contain integer parameters

        Returns:
            The i-th time-series
            An instance of class FoldedDataset containing the dataset with the lives specified by the list
        """
        if isinstance(i, slice):
            i = range(
                0 if i.start is None else i.start,
                len(self) if i.stop is None else i.stop,
                1 if i.step is None else i.step,
            )
        if TENSORFLOW_ENABLED and isinstance(i, tf.Tensor):
            return self.get_time_series(i.ref())
        


        if isinstance(i, Iterable):
            if not all(isinstance(item, (int, np.integer)) for item in i):
                if len(i) == 2:       
                    if not isinstance(i[1], EllipsisType):
                        raise ValueError("Invalid iterable index passed")
                    i = i[0]        

            return FoldedDataset(self, i)
        else:
            df = self.get_time_series(i)
            return df

    @property
    def shape(self) -> Tuple[int]:
        return (self.n_time_series,)

    def __len__(self) -> int:
        """
        Compute the number of lifes in the dataset

        Return:
            Number of time-series in the dataset
        """
        return self.n_time_series

    def to_pandas(
        self,
        proportion_of_lives: float = 1.0,
        subsample_proportion: float = 1.0,
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """
        Create a dataset with the time-series concatenated

        Parameters:
            proportion_of_lives: Proportion of lives to use, by default 1
            subsample_proportion: Proportion of samples to use, by default 1
            show_progress: Whether to show progress when concatenating the lives, by default False

        Returns:
            A DataFrame with all the lives concatenated
        """
        if show_progress:
            bar = tqdm
        else:
            bar = lambda x: x
        df = []

        features = list(
            self._compute_common_features(
                proportion_of_lives=proportion_of_lives, show_progress=show_progress
            )
        )

        for i in bar(range(self.n_time_series)):
            if proportion_of_lives < 1.0 and np.random.rand() > proportion_of_lives:
                continue

            current_life = self[i].loc[:, features]
            if subsample_proportion < 1.0:
                indices = range(
                    0,
                    current_life.shape[0],
                    int(current_life.shape[0] * subsample_proportion),
                )
                current_life = current_life.iloc[indices, :]
            df.append(current_life)
        return pd.concat(df)

    def _compute_common_features(
        self, proportion_of_lives: float = 1.0, show_progress: bool = False
    ) -> List[str]:
        common_features = []
        if show_progress:
            bar = tqdm
        else:
            bar = lambda x: x
        for i in bar(range(self.n_time_series)):
            if proportion_of_lives < 1.0 and np.random.rand() > proportion_of_lives:
                continue
            life = self.get_features_of_life(i)
            common_features.append(set(life.columns.values))
        return sorted(list(common_features[0].intersection(*common_features)))

    def common_features(
        self, show_progress: bool = False, proportion_of_lives: float = 1.0
    ) -> List[str]:
        """
        Compute the common features of the dataset among the different lives

        Parameters:
            proportion_of_lives: Proportion of lives to use, by default 1
            show_progress: Whether to show progress when computing the common features, by default False

        Returns:
            A list with the common features
        """
        if self._common_features is None:
            self._common_features = self._compute_common_features(
                proportion_of_lives, show_progress=show_progress
            )
        return self._common_features

    def map(
        self, transformer: "TransformedDataset", cache_size: int = None
    ) -> "TransformedDataset":
        """
        Apply a transformation to the dataset

        Parameters:
            transformer: The transformation to apply
            cache_size: The size of the cache to use, by default None

        Returns:
            The transformed dataset as an instance of class TransformedDataset
        """
        from ceruleo.dataset.transformed import TransformedDataset

        return TransformedDataset(self, transformer, cache_size=cache_size)

    def numeric_features(self, show_progress: bool = False) -> List[str]:
        """Obtain the list of the common numeric features in the dataset

        Parameters:
            show_progress: Whether to show progress when computing the common features, by default False

        Returns:
            List of columns containing the common numeric features
        """

        features = self.common_features(show_progress=show_progress)
        df = self.get_features_of_life(0)
        return list(
            df.loc[:, features]
            .select_dtypes(include=[np.number], exclude=["datetime", "timedelta"])
            .columns.values
        )

    def categorical_features(self, show_progress: bool = False) -> List[str]:
        """Obtain the list of the common categorical features in the dataset

        Parameters:
            show_progress: Whether to show progress when computing the common features

        Returns:
            List of columns containing the common numeric features
        """
        features = self.common_features(show_progress=show_progress)
        df = self.get_time_series(0)
        return list(
            df.loc[:, features]
            .select_dtypes(exclude=[np.number, "datetime", "timedelta"])
            .columns.values
        )


class FoldedDataset(AbstractPDMDataset):
    """
    Dataset containing a subset of the time-series. An instanc of this class can be obtained by slicing an AbstractTimeSeriesDataset with a list of indexes
    """

    def __init__(self, dataset: AbstractPDMDataset, indices: list):
        super().__init__()
        self.dataset = dataset
        self.indices = indices

    def __getattribute__(self, __name: str) -> Any:
        try:
            return super().__getattribute__(__name)
        except Exception as e:
            return self.dataset.__getattribute__(__name)

    @property
    def n_time_series(self) -> int:
        """
        Compute the number of lifes in the folded dataset

        Return:
            Number of time-series in the folded dataset

        """
        return len(self.indices)

    def get_time_series(self, i: int) -> pd.DataFrame:
        """
        Obtain the i-th time-series in the folded dataset

        Parameters:
            i: Index of the life

        Returns:
            The i-th time-series
        """
        return self.dataset[self.indices[i]]

    def _original_index(self, i: int) -> int:
        """
        Obtain the index of the i-th time-series in the original dataset

        Parameters:
            i: Index of the life

        Returns:
            The index of the i-th time-series in the original dataset
        """
        if isinstance(self.dataset, FoldedDataset):
            return self.dataset._original_index(self.indices[i])
        else:
            return self.indices[i]

    def original_indices(self) -> List[int]:
        """
        Obtain the original indices for all the time-series in the FoldedDataset

        Returns:
            The original indices for all the time-series in the FoldedDataset
        """
        return [self._original_index(i) for i in range(len(self.indices))]

    def number_of_samples_of_time_series(self, i: int) -> int:
        """
        Compute the number of samples of the i-th time-series in the FoldedDataset

        Parameters:
            i: Index of the life

        Returns:
            Number of samples of the i-th time-series in the FoldedDataset
        """
        return self[i][0].shape[0]

    def __reduce_ex__(self, __protocol) -> Union[str, Tuple[Any, ...]]:
        return (self.__class__, (self.dataset, self.indices))

    @property
    def rul_column(self):
        return self.dataset.rul_column


class PDMDataset(AbstractPDMDataset):
    dataset_path: Path 
    procesed_path: Path 
    cycles_table_filename: Path 
    cycles_metadata: pd.DataFrame


    def __init__(self, path: Path):
        super().__init__()
        self.dataset_path = path        
        self.procesed_path = self.dataset_path / "processed" / "cycles"
        self.cycles_table_filename = self.procesed_path / "cycles.csv"
        self._prepare_dataset()
        self.cycles_metadata = pd.read_csv(self.cycles_table_filename)

    def _prepare_dataset(self):
        pass

    @property
    def n_time_series(self) -> int:
        return len(self.cycles)

    @property
    def rul_column(self) -> int:
        return self._rul_column


class PDMInMemoryDataset(AbstractPDMDataset):
    cycles: List[pd.DataFrame]
    _rul_column: str
    cycles_metadata: Optional[pd.DataFrame]

    def __init__(
        self,
        cycles: List[pd.DataFrame],
        rul_column: str,
        cycles_metadata: Optional[pd.DataFrame] = None,
    ):
        super().__init__()
        self.cycles = cycles
        self._rul_column = rul_column
        self.cycles_metadata = cycles_metadata
        

    def get_time_series(self, i: int) -> pd.DataFrame:
        return self.cycles[i]

    @property
    def n_time_series(self) -> int:
        return len(self.cycles)

    @property
    def rul_column(self) -> int:
        return self._rul_column
