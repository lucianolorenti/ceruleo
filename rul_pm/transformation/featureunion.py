import numpy as np
import scipy.sparse as sparse
from numpy.lib.arraysetops import isin
from rul_pm.transformation.pipeline import LivesPipeline
from rul_pm.transformation.transformerstep import TransformerStepMixin
from sklearn.pipeline import FeatureUnion, _transform_one


class PandasFeatureUnion(FeatureUnion, TransformerStepMixin):

    def fit(self,  X, apply_before=None):
        from rul_pm.transformation.pipeline import transform_given_list
        self._validate_transformers()
        estimators = apply_before.copy() if apply_before is not None else []
        for _, trans, _ in self._iter():
            if isinstance(trans, LivesPipeline):
                trans.fit(X, apply_before=apply_before)
            else:
                for life in X:
                    trans.partial_fit(transform_given_list(life, estimators))

        return self

    def merge_dataframes_by_column(self, Xs):
        # indices = [X.index for X in Xs]
        # TODO: Check equal indices
        names = [name for name, _, _ in self._iter()]
        X = Xs[0]
        X.columns = [f'{names[0]}_{c}' for c in X.columns]
        for name, otherX in zip(names[1:], Xs[1:]):
            for c in otherX.columns:
                X[f'{name}_{c}'] = otherX[c]
        return X

    def transform(self, X):
        Xs = []
        for name, trans, weight in self._iter():
            Xs.append(_transform_one(
                transformer=trans,
                X=X,
                y=None,
                weight=weight)
            )
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        return params
