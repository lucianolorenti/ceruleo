from typing import Any, List, Union

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs


DataFrame = Union[pd.DataFrame, pl.DataFrame]
Series = Union[pd.Series, pl.Series]

def is_dataframe(df: DataFrame) -> bool:
    """
    Check if the DataFrame is a pandas or polars DataFrame

    Parameters:
        df: DataFrame to check

    Returns:
        True if the DataFrame is a pandas or polars DataFrame, False otherwise
    """
    return is_pandas(df) or is_polars(df)

def is_polars(df: DataFrame) -> bool:
    """
    Check if the DataFrame is a polars DataFrame

    Parameters:
        df: DataFrame to check

    Returns:
        True if the DataFrame is a polars DataFrame, False otherwise
    """
    return isinstance(df, pl.DataFrame) or isinstance(df, pl.Series)

def is_pandas(df: DataFrame) -> bool:
    """
    Check if the DataFrame is a pandas DataFrame

    Parameters:
        df: DataFrame to check

    Returns:
        True if the DataFrame is a pandas DataFrame, False otherwise
    """
    return isinstance(df, pd.DataFrame) or isinstance(df, pd.Series)

def dataframe_fillna(df: DataFrame, value: Any) -> DataFrame:
    """
    Fill NA/NaN values using the specified value

    Parameters:
        df: DataFrame to fill
        value: Value to fill NA/NaN values

    Returns:
        DataFrame with NA/NaN values filled
    """
    if is_pandas(df):
        return df.fillna(value)
    return df.fill_nan(value)


def get_numeric_features(df:DataFrame, features: List[str]) -> List[str]:
    if is_pandas(df):
        return list(
            df.loc[:, features]
            .select_dtypes(include=[np.number], exclude=["datetime", "timedelta"])
            .columns.values
        )
    else:
        return list(
           df.select(cs.by_dtype(pl.NUMERIC_DTYPES))
            .columns
        )
    

def dataframe_select_column(df: DataFrame, column: str) -> Series:
    """
    Select columns from a DataFrame

    Parameters:
        df: DataFrame to select columns from
        column: Columns to select

    Returns:
        DataFrame with selected columns
    """
    if is_pandas(df):
        return df[column]
    return df.get_column(column)


def dataframe_select_column_as_array(df: DataFrame, column: str) -> np.ndarray:
    """
    Select columns from a DataFrame

    Parameters:
        df: DataFrame to select columns from
        column: Columns to select

    Returns:
        DataFrame with selected columns
    """
    if is_pandas(df):
        return df[column].values
    return df.get_column(column)


def dataframe_min(df: DataFrame, skipna:bool=True) ->  pd.DataFrame:
    """
    Compute the minimum value of a DataFrame

    Parameters:
        df: DataFrame to compute the minimum value

    Returns:
        Minimum value of the DataFrame
    """
    if is_pandas(df):
        return df.min(skipna=skipna)
    return df.min().to_pandas()

def dataframe_max(df: DataFrame, skipna:bool=True) -> pd.DataFrame:
    """
    Compute the maximum value of a DataFrame

    Parameters:
        df: DataFrame to compute the maximum value

    Returns:
        Maximum value of the DataFrame
    """
    if is_pandas(df):
        return df.max(skipna=skipna)
    return df.max().to_pandas()