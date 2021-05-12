import numpy as np
from rul_pm.results.results import regression_metrics


class TestResults():
    def test_metrics(self):
        data = {
            'Model 1': [{
                'predicted': np.array([3, 2, 1]),
                'true': np.array([3, 2, 1])
            }, {
                'predicted': np.array([4, 2, 1]),
                'true': np.array([3, 2, 1])
            }],
            'Model 2': [{
                'predicted': np.array([3, 2, 1]),
                'true': np.array([3, 2, 1])
            }, {
                'predicted': np.array([3, 2, 1]),
                'true': np.array([3, 2, 1])
            }],
        }
        r = regression_metrics(data)
        assert len(r) == 2
        assert 'Model 1' in r
        assert 'Model 2' in r
        assert 'mean' in r['Model 1']
        assert 'std' in r['Model 1']

        assert r['Model 2']['mean'] == 0
        assert r['Model 2']['std'] == 0

        assert r['Model 1']['mean'] > 0

        r = regression_metrics(data, 2)
        assert r['Model 2']['mean'] == 0
        assert r['Model 2']['std'] == 0

        assert r['Model 1']['mean'] == 0