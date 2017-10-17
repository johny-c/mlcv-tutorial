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
