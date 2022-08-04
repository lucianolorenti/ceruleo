import numpy as np 
from ceruleo.results.results import split_lives_indices
import pytest

class TestResults:
    def test_1(self):

        v1 = np.linspace(25, 0, 50)
        v2 = np.linspace(17, 5, 45)
        v3 = np.linspace(50, 7, 50)  
        y_true = np.hstack((v1,v2,v3))
        indices = split_lives_indices(y_true)
        assert (y_true[indices[0]] == v1).all()
        assert (y_true[indices[1]] == v2).all()
        assert (y_true[indices[2]] == v3).all()
        assert list(map(len, indices)) == [50, 45, 50]

        y_true = v1
        indices = split_lives_indices(y_true)
        assert (y_true[indices[0]] == v1).all()
        assert list(map(len, indices)) == [50]

        with pytest.raises(AssertionError):
            y_true = np.linspace(25, 0, 1)
            indices = split_lives_indices(y_true)
            assert (y_true[indices[0]] == v1).all()
            assert list(map(len, indices)) == []

