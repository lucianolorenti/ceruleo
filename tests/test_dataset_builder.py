from ceruleo.dataset.builder.builder import DatasetBuilder
from ceruleo.dataset.builder.cycles_splitter import (
    CyclesSplitter,
    FailureDataCycleSplitter,
    IncreasingFeatureCycleSplitter,
    LifeEndIndicatorCycleSplitter,
    LifeIdCycleSplitter,
)
import pandas as pd

from ceruleo.dataset.builder.output import InMemoryOutputMode
from ceruleo.dataset.builder.rul_column import CycleRULColumn, DatetimeRULColumn
from ceruleo.dataset.ts_dataset import AbstractPDMDataset, PDMDataset

def is_decreasing(s:pd.Series):
    return all(s[i] >= s[i+1] for i in range(len(s)-1))

def test_dataset_builder_one_file_increasing_feature_cycle_RUL():
    df = pd.DataFrame(
        {
            "Cycle": list(range(0, 12, ))*2,
            "feature_a": list(range(12))*2,
            "feature_b": list(range(12, 24))*2,
        }
    )
    dataset = (
        DatasetBuilder.one_file_format()
        .set_splitting_method(IncreasingFeatureCycleSplitter("Cycle"))
        .set_rul_column_method(CycleRULColumn("Cycle"))
        .set_output_mode(InMemoryOutputMode())
        .build_from_df(df)
    )
    assert isinstance(dataset, AbstractPDMDataset) 
    assert len(dataset) == 2
    assert isinstance(dataset[0], pd.DataFrame)
    assert "RUL" in dataset[0].columns.values.tolist()
    assert is_decreasing(dataset[0].RUL)


def test_dataset_builder_one_file_increasing_feature_datetime_RUL():
    df = pd.DataFrame(
        {
            "Cycle": list(range(0, 12, ))*2,
            "datetime": pd.date_range("2021-01-01", periods=24, freq="min").tolist(),
            "feature_a": list(range(12))*2,
            "feature_b": list(range(12, 24))*2,
        }
    )
    dataset = (
        DatasetBuilder.one_file_format()
        .set_splitting_method(IncreasingFeatureCycleSplitter("Cycle"))
        .set_rul_column_method(DatetimeRULColumn("datetime", "s"))
        .set_output_mode(InMemoryOutputMode())
        .build_from_df(df)
    )
    assert isinstance(dataset, AbstractPDMDataset) 
    assert len(dataset) == 2
    assert isinstance(dataset[0], pd.DataFrame)
    assert "RUL" in dataset[0].columns.values.tolist()
    assert is_decreasing(dataset[0].RUL)
    assert dataset[0].RUL.min() == 0
    assert dataset[0].RUL.max() == 11*60


def test_dataset_builder_one_file_life_id_cycle_datetime_RUL():
    df = pd.DataFrame(
        {
            "life_id": [1]*12 + [2]*12,
            "datetime": pd.date_range("2021-01-01", periods=24, freq="min").tolist(),
            "feature_a": list(range(12))*2,
            "feature_b": list(range(12, 24))*2,
        }
    )
    dataset = (
        DatasetBuilder.one_file_format()
        .set_splitting_method(LifeIdCycleSplitter("life_id"))
        .set_rul_column_method(DatetimeRULColumn("datetime", "s"))
        .set_output_mode(InMemoryOutputMode())
        .build_from_df(df)
    )
    assert isinstance(dataset, AbstractPDMDataset) 
    assert len(dataset) == 2
    assert isinstance(dataset[0], pd.DataFrame)
    assert "RUL" in dataset[0].columns.values.tolist()
    assert is_decreasing(dataset[0].RUL)
    assert dataset[0].RUL.min() == 0
    assert dataset[0].RUL.max() == 11*60


def test_dataset_builder_one_file_life_end_datetime_RUL():
    df = pd.DataFrame(
        {
            "life_end": [0]*11 + [1] + [0]*11 + [1],
            "datetime": pd.date_range("2021-01-01", periods=24, freq="min").tolist(),
            "feature_a": list(range(12))*2,
            "feature_b": list(range(12, 24))*2,
        }
    )
    dataset = (
        DatasetBuilder.one_file_format()
        .set_splitting_method(LifeEndIndicatorCycleSplitter("life_end"))
        .set_rul_column_method(DatetimeRULColumn("datetime", "s"))
        .set_output_mode(InMemoryOutputMode())
        .build_from_df(df)
    )
    assert isinstance(dataset, AbstractPDMDataset) 
    assert len(dataset) == 2
    assert isinstance(dataset[0], pd.DataFrame)
    assert "RUL" in dataset[0].columns.values.tolist()
    assert is_decreasing(dataset[0].RUL)
    assert dataset[0].RUL.min() == 0
    assert dataset[0].RUL.max() == 11*60







def test_dataset_builder_two_file_life_end_datetime_RUL():
    df = pd.DataFrame(
        {
           
            "datetime": pd.date_range("2021-01-01", periods=24, freq="min").tolist(),
            "feature_a": list(range(12))*2,
            "feature_b": list(range(12, 24))*2,
        }
    )
    failures = pd.DataFrame({
        "datetime_failure": [pd.Timestamp("2021-01-01 00:11:00"), pd.Timestamp("2021-01-01 00:23:00")],
        "failure_type": ["A", "B"]
    })
    dataset = (
        DatasetBuilder.one_file_format()
        .set_splitting_method(FailureDataCycleSplitter("datetime", "datetime_failure"))
        .set_rul_column_method(DatetimeRULColumn("datetime", "s"))
        .set_output_mode(InMemoryOutputMode())
        .build_from_data_fault_pair((df, failures))
    )
    assert isinstance(dataset, AbstractPDMDataset) 
    assert len(dataset) == 2
    assert isinstance(dataset[0], pd.DataFrame)
    assert "RUL" in dataset[0].columns.values.tolist()
    assert is_decreasing(dataset[0].RUL)
    assert dataset[0].RUL.min() == 0
    assert dataset[0].RUL.max() == 11*60



