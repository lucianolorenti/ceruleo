from typing import Callable, Optional
from temporis.transformation.functional.transformerstep import TransformerStep


class TransformerLambda(TransformerStep):
    def __init__(self, function:Callable,  name:Optional[str]=None):
        super().__init__(name=name)
        self.function = function

    def transform(self, X, y=None):
        return self.function(X)