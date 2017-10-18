from __future__ import absolute_import

from mlcv.datasets import fetcher


class TestFetcher:

    def test_load_iris(self):
        X, y = fetcher.load('iris')

        assert(len(X) == len(y))
        assert(X.shape == (150, 4))
