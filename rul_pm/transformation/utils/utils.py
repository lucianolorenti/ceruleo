from typing import Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided
from rul_pm.transformation.transformerstep import TransformerStep
from sklearn.pipeline import FeatureUnion, _transform_one


class PandasToNumpy(TransformerStep):
    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.values


class IdentityTransformer(TransformerStep):
    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        if isinstance(input_array, pd.DataFrame):
            return input_array.copy()
        else:
            return input_array * 1


class PandasTransformerWrapper(TransformerStep):
    def __init__(self, transformer, name: Optional[str] = None):
        super().__init__(name)
        self.transformer = transformer

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input array must be a data frame")
        self.transformer.fit(X.values)
        return self

    def partial_fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input array must be a data frame")
        if hasattr(self.transformer, "partial_fit"):
            self.transformer.partial_fit(X.values)
        else:
            self.transformer.fit(X.values)
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input array must be a data frame")
        return pd.DataFrame(
            self.transformer.transform(X), columns=X.columns, index=X.index
        )


def column_names_window(columns: list, window: int) -> list:
    """

    Parameters
    ----------
    columns: list
             List of column names

    window: int
            Window size

    Return
    ------
    Column names with the format: w_{step}_{feature_name}
    """
    new_columns = []
    for w in range(1, window + 1):
        for c in columns:
            new_columns.append(f"w_{w}_{c}")
    return new_columns


def sliding_window_view(arr, window_shape, steps):
    """Produce a view from a sliding, striding window over `arr`.
    The window is only placed in 'valid' positions - no overlapping
    over the boundary.

    Parameters
    ----------
    arr : numpy.ndarray, shape=(...,[x, (...), z])
        The array to slide the window over.
    window_shape : Sequence[int]
        The shape of the window to raster: [Wx, (...), Wz],
        determines the length of [x, (...), z]
    steps : Sequence[int]
        The step size used when applying the window
        along the [x, (...), z] directions: [Sx, (...), Sz]

    Returns
    -------
    view of `arr`, shape=([X, (...), Z], ..., [Wx, (...), Wz])
        Where X = (x - Wx) // Sx + 1
        
    Notes
    -----
    In general, given
      `out` = sliding_window_view(arr,
                                  window_shape=[Wx, (...), Wz],
                                  steps=[Sx, (...), Sz])
       out[ix, (...), iz] = arr[..., ix*Sx:ix*Sx+Wx,  (...), iz*Sz:iz*Sz+Wz]
     Examples
     --------
     >>> import numpy as np
     >>> x = np.arange(9).reshape(3,3)
     >>> x
     array([[0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]])
     >>> y = sliding_window_view(x, window_shape=(2, 2), steps=(1, 1))
     >>> y
     array([[[[0, 1],
              [3, 4]],
             [[1, 2],
              [4, 5]]],
            [[[3, 4],
              [6, 7]],
             [[4, 5],
              [7, 8]]]])
    >>> np.shares_memory(x, y)
     True
    # Performing a neural net style 2D conv (correlation)
    # placing a 4x4 filter with stride-1
    >>> data = np.random.rand(10, 3, 16, 16)  # (N, C, H, W)
    >>> filters = np.random.rand(5, 3, 4, 4)  # (F, C, Hf, Wf)
    >>> windowed_data = sliding_window_view(data,
    ...                                     window_shape=(4, 4),
    ...                                     steps=(1, 1))
    >>> conv_out = np.tensordot(filters,
    ...                         windowed_data,
    ...                         axes=[[1,2,3], [3,4,5]])
    # (F, H', W', N) -> (N, F, H', W')
    >>> conv_out = conv_out.transpose([3,0,1,2])
    """

    in_shape = np.array(arr.shape[-len(steps) :])  # [x, (...), z]
    window_shape = np.array(window_shape)  # [Wx, (...), Wz]
    steps = np.array(steps)  # [Sx, (...), Sz]
    nbytes = arr.strides[-1]  # size (bytes) of an element in `arr`

    # number of per-byte steps to take to fill window
    window_strides = tuple(np.cumprod(arr.shape[:0:-1])[::-1]) + (1,)
    # number of per-byte steps to take to place window
    step_strides = tuple(window_strides[-len(steps) :] * steps)
    # number of bytes to step to populate sliding window view
    strides = tuple(int(i) * nbytes for i in step_strides + window_strides)

    outshape = tuple((in_shape - window_shape) // steps + 1)
    # outshape: ([X, (...), Z], ..., [Wx, (...), Wz])
    outshape = outshape + arr.shape[: -len(steps)] + tuple(window_shape)
    return as_strided(arr, shape=outshape, strides=strides, writeable=False)



