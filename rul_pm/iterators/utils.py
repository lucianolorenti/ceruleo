import numpy as np
from rul_pm.transformation.transformers import Transformer, simple_pipeline


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


def get_features(dataset,  features, step=1, window=1):
    t = simple_pipeline(features)
    data = {f: [] for f in features}

    def populate_data(X, y):
        for i, f in enumerate(features):
            data[f].extend(np.squeeze(X[:, i]).tolist())
    t = Transformer(
        features,
        t,
        transformerY=PandasToNumpy()
    )
    t.fit(dataset)
    dataset_map(populate_data, dataset, step, t, window)
    return data
