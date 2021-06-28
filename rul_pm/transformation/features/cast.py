
import logging
from typing import Optional

import pandas as pd
from rul_pm.transformation.transformerstep import TransformerStep

logger = logging.getLogger(__name__)


class CastTo(TransformerStep):
    def __init__(self, type:str, name:Optional[str]=None):
        super().__init__(name=name)
        self.type = type
        

    def transform(self, X: pd.DataFrame):
        return X.astype(self.type)
