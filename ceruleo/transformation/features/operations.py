
from typing import List
from ceruleo.transformation.functional.transformerstep import TransformerStep
import pandas as pd
from functools import reduce 


class Sum(TransformerStep):
    def transform(self, X: List[pd.DataFrame]):
        return reduce(lambda x, y: x.add(y, fill_value=0), X)


class Divide(TransformerStep):
    def transform(self, X: List[pd.DataFrame]):
        return reduce(lambda x, y: x.divide(y, fill_value=0), X)