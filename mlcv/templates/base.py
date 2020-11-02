import six
from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator


class Solution(six.with_metaclass(ABCMeta, BaseEstimator)):
    """This is a core class, meant to be subclassed by specific estimators."""

    @abstractmethod
    def __init__(self):
        """
        You should pass any algorithm specific parameters to the constructor
        of your estimator (your estimator's __init__ method)
        """
        pass

    @abstractmethod
    def _validate_training_inputs(self, X, y=None):
        """Validate X and y.

        Make sure they have the shapes you expect and make sure they are
        compatible with the model parameters passed to __init__.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            User supplied training data.

        y : array, shape (n_samples, ?) (optional)
            User supplied training targets.

        Returns
        -------
        X_validated : array, shape (n_samples, n_features)
            Validated training data.

        y_validated : array, shape (n_samples, ?)
            Validated training targets.

        Raises
        ------
        ValueError : If the inputs or the parameters do not match the expected
        format or their values are not compatible.

        """
        pass

    @abstractmethod
    def fit(self, X, y=None):
        """Fit your model with the training data.

        Parameters
        ----------
        X : array, shape (n_samples_train, n_features)
            Training inputs.

        y : array, shape (n_samples_train, ?)
            Corresponding training targets.

        Returns
        -------
        self : Solution
            A trained model instance.

        """
        pass

    @abstractmethod
    def _validate_testing_inputs(self, X):
        """Validate X and make sure it is compatible with the model parameters.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            User supplied testing data.

        Returns
        -------
        X_validated : array, shape (n_samples, n_features)
            Validated testing data.
        Raises
        ------
        ValueError : If the inputs or the parameters do not match the expected
        format or their values are not compatible.

        """
        pass

    @abstractmethod
    def predict(self, X):
        """Predict with your trained model on unseen input data.

        Parameters
        ----------
        X : array, shape (n_samples_test, n_features)
            Testing inputs.

        Returns
        -------
        y : array, shape(n_samples_test, ?)
            A prediction for each testing input.

        Raises
        ------
        NotFittedError : If your model has not been trained before,
        you should not be able to predict with it.

        """
        pass

    def predict_proba(self, X):
        """Predict target probabilities for each testing input.

        Parameters
        ----------
        X : array, shape (n_samples_test, n_features)
            Testing inputs.

        Returns
        -------
        y : array, shape(n_samples_test, n_target_set)
            A probability distribution for each testing input.

        """
        self._validate_testing_inputs(X)
        pass

    @abstractmethod
    def score(self, y_pred, y_true):
        """Return a single number that reflects the quality of your model.

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
        pass

    def visualize_iteration(self, X, *args, **kwargs):
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
