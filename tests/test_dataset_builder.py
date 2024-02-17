from ceruleo.dataset.builder.builder import DatasetBuilder
from ceruleo.dataset.builder.cycles_splitter import (
    CyclesSplitter,
    IncreasingFeatureCycleSplitter,
)
import pandas as pd

from ceruleo.dataset.builder.output import InMemoryOutputMode
from ceruleo.dataset.ts_dataset import PDMDataset


def test_dataset_builder():
    df = pd.DataFrame(
        {
            "RUL": list(range(0, 12, ))*2,
            "feature_a": list(range(12))*2,
            "feature_b": list(range(12, 24))*2,
        }
    )
    dataset = (
        DatasetBuilder.one_file_format()
        .set_splitting_method(IncreasingFeatureCycleSplitter("RUL"))
        .set_output_mode(InMemoryOutputMode())
        .build_from_df(df)
    )
    isinstance(dataset, PDMDataset) 
