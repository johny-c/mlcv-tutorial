import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator


class Solution(BaseEstimator):
    def __init__(self, **algorithm_params):
        pass

    def _validate_training_inputs(self, X, y=None):
        """Validate X and y and make sure they are compatible with the model
        parameters passed to __init__.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples, ?)

        Raises
        ------
        ValueError : If the inputs or the parameters do not match the expected
        format or their values are not compatible.

        """
        pass

    def fit(self, X, y=None):
        """

        Parameters
        ----------
        X : array, shape (n_samples_train, n_features)
            Training inputs.

        y : array, shape (n_samples_train, ?)
            Corresponding training targets.

        Returns
        -------
        solution : Solution
            A trained model.

        """
        self._validate_training_inputs(X, y)

        return self

    def _validate_testing_inputs(self, X):
        """Validate X and make sure it is compatible with the model parameters.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)

        Raises
        ------
        ValueError : If the inputs or the parameters do not match the expected
        format or their values are not compatible.

        """
        pass

    def predict(self, X):
        """

        Parameters
        ----------
        X : array, shape (n_samples_test, n_features)
            Testing inputs.

        Returns
        -------
        y : array, shape(n_samples_test, ?)
            A prediction for each testing input.

        """
        self._validate_testing_inputs(X)
        pass

    def predict_proba(self, X):
        """

        Parameters
        ----------
        X : array, shape (n_samples_test, n_features)
            Testing inputs.

        Returns
        -------
        y : array, shape(n_samples_test, n_classes)
            A class probability distribution for each testing input.

        """
        self._validate_testing_inputs(X)
        pass

    def score(self, y_pred, y_true):
        """

        Parameters
        ----------
        y_pred : array, shape(n_samples_test, ?)
            Predictions.

        y_true : array, shape(n_samples_test, ?)
            Groundtruth targets.
        Returns
        -------
        score : float
            An evaluation score, based on the discrepancy between
            predictions and groundtruth.

        """
        pass

    def visualize(self, X, *args, **kwargs):
        """

        Parameters
        ----------
        X :
        args :
        kwargs :

        Returns
        -------

        """
        pass

    def visualize_iter(self, X, *args, **kwargs):
        pass

    def print_progress(self, **kwargs):
        pass

    def preprocess_inputs(self, X, y):
        """

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input data.

        y : array, shape (n_samples, ?)
            Input targets.

        Returns
        -------
        X : array, shape (n_samples_, n_features_)
            Preprocessed input data.

        y : array, shape (n_samples_, ?)
            Preprocessed input targets.

        """
        pass


def main():
    ## 1. Read in the data
    # X_train = pd.read_csv('train_data.csv')
    # y_train = pd.read_csv('train_labels.csv')

    # X_test = pd.read_csv('test_data.csv')
    # y_test = pd.read_csv('test_labels.csv')

    ## 2. Do any necessary pre-processing (e.g. mean-centering)
    # X_train, y_train = preprocess_inputs(X_train, y_train)
    # X_test, y_test = preprocess_inputs(X_test, y_test)

    ## 3. Initialize your model (set the (hyper-) parameters)
    # model = Solution(param_1=value_1, param_2=value_2, ..., param_k=value_k)

    ## 4. Make sure your parameters and your inputs are valid and compatible.
    # model.validate_input(X_train, y_train)
    # model.validate_input(X_test, y_test)

    ## 5. Fit your model with the training data (That's the meat...)
    # model.fit(X_train, y_train)

    ## 6. Evaluate your model on some testing data (See what you learned.)
    # y_pred = model.predict(X_test)
    # model_score = model.score(y_pred, y_test)

    ## 7. Maybe plot something your model learned.
    # model.visualize(X_train, colors=model.param_5)
    pass

if __name__ == '__main__':
    main()