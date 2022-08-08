"""

The transformer is a high-level class that hold at least two transformation pipelines

  * One related to the transformation of the input of the model
  * The other related to the target of the model.

It allows accessing the information of the transformed data and is the object that uses the
dataset iterators to transform the data before feeding it to the model.
"""
import copy
import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from ceruleo.transformation.functional.graph_utils import \
    topological_sort_iterator
from ceruleo.transformation.functional.pipeline.cache_store import \
    CacheStoreType
from ceruleo.transformation.functional.pipeline.pipeline import \
    TemporisPipeline
from ceruleo.transformation.functional.transformerstep import TransformerStep

logger = logging.getLogger(__name__)



def transformer_info(transformer : Optional[TemporisPipeline]):
    """Obtains the transformer information in a serializable format

    Parameters:

        transformer: The transformer step, or pipeline to obtain their underlying information

    Returns:
        dict

    Raises:

        ValueError
            If the transformer passed as an argument doesn't have
            the get_params method.
    """
    if transformer is None:
        return 'Missing'

    data = []
    Q = topological_sort_iterator(transformer)
    for q in Q:
        data.append(q.description())
    return data








class Transformer:
    """Transform each life

    The transformer class is the highest level class of the transformer API.
    It contains Transformation Pipelines for the input data and the target,
    and provides mechanism to inspect the structure of the transformed data.

    Parameters:

        pipelineX: Pipeline that will be applied to the run-to-cycle data
        pipelineY: Pipeline that will be applied to the target.
        pipelineMetadata: Pipeline that will be used to extract additional
                            data from the lives information, by default None
    """

    def __init__(
        self,
        pipelineX: Union[TemporisPipeline, TransformerStep],
        pipelineY: Optional[Union[TemporisPipeline, TransformerStep]] = None,
        pipelineMetadata: Optional[Union[TemporisPipeline, TransformerStep]] = None,
        cache_type: CacheStoreType = CacheStoreType.MEMORY,

    ):
        def ensure_pipeline(x, cache_type: CacheStoreType):
            if isinstance(x, TemporisPipeline):
                return x
            return TemporisPipeline(x, cache_type=cache_type)

        self.pipelineX = ensure_pipeline(pipelineX, cache_type)
        if pipelineY is not None:
            self.pipelineY = ensure_pipeline(pipelineY, cache_type)
        else:
            self.pipelineY = None
        self.pipelineMetadata = (
            ensure_pipeline(pipelineMetadata, cache_type)
            if pipelineMetadata is not None
            else None
        )
        self.features = None
        self.fitted_ = False

    def _process_selected_features(self):
        if self.pipelineX["selector"] is not None:
            selected_columns = self.pipelineX["selector"].get_support(indices=True)
            self.features = [self.features[i] for i in selected_columns]

    def clone(self):
        return copy.deepcopy(self)

    def fit(self, dataset, show_progress: bool = False):
        """Fit the transformer with a given dataset.

        The transformer will fit the X transformer,
        the Y transformer and the metadata transformer

        Parameters
        ----------
        dataset : AbstractLivesDataset
            Dataset

        Returns
        -------
        self
        """
        logger.debug("Fitting Transformer")

        self.pipelineX.fit(dataset, show_progress=show_progress)
        if self.pipelineY is not None:
            self.pipelineY.fit(dataset, show_progress=show_progress)
        if self.pipelineMetadata is not None:
            self.pipelineMetadata.fit(dataset)

        if not isinstance(dataset, pd.DataFrame):
            self.minimal_df = dataset[0].head(n=20)
        else:
            self.minimal_df = dataset.head(n=20)
        X = self.pipelineX.transform(self.minimal_df)
        self.number_of_features_ = X.shape[1]
        self.fitted_ = True
        self.column_names = self._compute_column_names()
        return self

    def transform(self, life: pd.DataFrame):
        """Transform a life and obtain the input data, the target and the metadata

        Parameters
        ----------
        life : pd.DataFrame
            A life in a form of a DataFrame

        Returns
        -------
        Tuple[np.array, np.array, np.array]
            * The first element consists of the input transformed
            * The second element consits of the target transformed
            * The third element consists of the metadata
        """
        check_is_fitted(self, "fitted_")
        return (
            self.transformX(life),
            self.transformY(life),
            self.transformMetadata(life),
        )

    def fit_map(self, dataset, show_progress: bool = False ) -> "TransformedDataset":
        self.fit(dataset, show_progress=show_progress)
        return dataset.map(self)

    def transformMetadata(self, df: pd.DataFrame) -> Optional[any]:
        if self.pipelineMetadata is not None:
            return self.pipelineMetadata.transform(df)
        else:
            return None

    def transformY(self, life: pd.DataFrame) -> np.array:
        """Get the transformed target from a life

        Parameters
        ----------
        life : pd.DataFrame
            A life in a form of a DataFrame

        Returns
        -------
        np.array
            Target obtained from the life
        """
        if self.pipelineY is not None:
            return self.pipelineY.transform(life)
        else:
            return None

    def transformX(self, life: pd.DataFrame) -> np.array:
        """Get the transformer input data

        Parameters
        ----------
        life : pd.DataFrame
            A life i an form of a DataFrame

        Returns
        -------
        np.array
            Input data transformed
        """
        return self.pipelineX.transform(life)

    def columns(self) -> List[str]:
        """Columns names after transformation

        Returns
        -------
        List[str]
        """
        return self.column_names

    @property
    def n_features(self) -> int:
        """Number of features after transformation

        Returns
        -------
        int
        """
        return self.number_of_features_

    def _compute_column_names(self):
        return self.pipelineX.column_names

    def description(self):
        return {
            "features": self.features,
            "transformerX": transformer_info(self.pipelineX),
            "transformerY": transformer_info(self.pipelineY),
        }

    def __str__(self):
        return str(self.description())


def TransformerIdentity(rul_column: str = "RUL") -> Transformer:
    """Return the Transformer

    Parameters
    ----------
    rul_column : str, default, RUL
        Name of the RUL Column

    Returns
    -------
    Transformer
        [description]
    """
    from temporis.transformation.features.selection import \
        ByNameFeatureSelector
    from temporis.transformation.utils import IdentityTransformerStep

    return Transformer(IdentityTransformerStep(), ByNameFeatureSelector([rul_column]))
