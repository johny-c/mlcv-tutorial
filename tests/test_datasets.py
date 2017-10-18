from __future__ import absolute_import
import numpy as np


from mlcv.datasets import fetcher



class TestFetcher:

    def setUp(self):
        np.random.seed(0)

    def tearDown(self):
        pass

    def test_load_iris(self):
        X, y = fetcher.load('iris')
        self.assertTrue(len(X) == len(y))
        self.assertTrue(X.shape == (150, 4))
