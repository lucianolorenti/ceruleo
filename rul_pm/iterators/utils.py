import numpy as np
from rul_pm.transformation.transformers import Transformer


def dataset_map(fun, dataset, step, transformer, window):
    from rul_pm.iterators.batcher import get_batcher
    batcher = get_batcher(dataset,
                          window,
                          512,
                          transformer,
                          step,
                          shuffle=False,
                          restart_at_end=False)
    for X, y, w in batcher:
        fun(X, y)
