from sklearn.pipeline import Pipeline


class LivesPipeline(Pipeline):
    def partial_fit(self, X, y=None):
        args = [X, y]
        for name, est in self.steps:
            if est == 'passthrough':
                continue
            est.partial_fit(*args)
            X_transformed = est.transform(args[0])
            args = [X_transformed, y]
        return self
