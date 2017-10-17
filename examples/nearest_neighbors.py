import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

from sklearn.utils.validation import check_array, check_consistent_length
from sklearn.utils.validation import check_is_fitted
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

from mlcv.templates.base import Solution
from mlcv.visualization.utils import proba_to_rgba


class KNNClassifier(Solution):
    """K nearest neighbors classifier.

    Parameters
    ----------
    k : int, optional (default=1)
        The number of neighbors to consider.

    metric : str, optional (default='euclidean')
        The metric to use for distances computation.

    """

    def __init__(self, k=1, metric='euclidean'):
        super(KNNClassifier, self).__init__()
        self.k = k
        self.metric = metric

    def _validate_training_inputs(self, X, y=None):
        """Validate the parameters passed in __init__ and make sure they are
        consistent with X and y.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            There should be at least k + 1 samples.

        y : array, shape (n_samples, ?)
            There should be at least 2 distinct classes

        Raises
        ------
        ValueError : If the inputs or the parameters do not match the expected
        format or their values are not compatible.

        """

        check_consistent_length(X, y)

        # Check the number of neighbors is a positive integer
        if self.k < 1:
            raise ValueError("Number of neighbors must be at least 1.")


        n_samples, n_features = X.shape

        if n_samples < self.k + 1:
            raise ValueError("Only {} samples given, cannot fit {} nearest "
                             "neighbors.".format(n_samples, self.k))

        n_classes = len(np.unique(y))
        if n_classes < 2:
            raise ValueError("Only {} classes, cannot fit {} nearest "
                             "neighbors.".format(n_classes, self.k))

        if self.metric != 'euclidean':
            raise ValueError("Currently only euclidean metric is supported.")

    def fit(self, X, y=None):
        """

        Parameters
        ----------
        X : array, shape (n_samples_train, n_features)
            Training inputs.

        y : array, shape (n_samples_train,)
            Corresponding training targets.

        Returns
        -------
        solution : Solution
            A trained model.

        """
        self._validate_training_inputs(X, y)

        # Actually nearest neighbors does not learn anything.
        # It just memorizes the whole training set.
        self.X_ = X

        # Store a label encoder with the mapping y -> [0, n_classes)
        self.label_encoder_ = LabelEncoder()

        # Store the encoded labels
        self.y_ = self.label_encoder_.fit_transform(y)

        # Store a label binarizer to use if predicting class probabilities
        self.label_binarizer_ = LabelBinarizer()

        # Fit the label binarizer with the encoded labels
        self.label_binarizer_.fit(self.y_)

        return self

    def _validate_testing_inputs(self, X):
        """Make sure the testing inputs are compatible.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The number of features should be the same as in the training set.

        Raises
        ------
        ValueError : If the inputs or the parameters do not match the expected
        format or their values are not compatible.

        """

        # Make sure the model has been trained first
        check_is_fitted(self, ['X_', 'y_'])

        # Make sure the testing inputs are in a valid format
        check_array(X)


        n_features_train = self.X_.shape[1]
        n_features_test = X.shape[1]
        if n_features_test != n_features_train:
            raise ValueError("The testing set has {} features, while the "
                             "training set has {} features.".format(
                n_features_test, n_features_train))

    def predict(self, X):
        """

        Parameters
        ----------
        X : array, shape (n_samples_test, n_features)
            Testing inputs.

        Returns
        -------
        y : array, shape(n_samples_test,)
            A prediction for each testing input.

        """

        self._validate_testing_inputs(X)

        # Find the nearest neighbors in the training set
        idx_nn = _find_nearest_neighbors(X, self.X_, self.k,
                                         return_distance=False)

        # Find the (encoded) labels of the nearest neighbors
        y_nn = self.y_[idx_nn]

        # For each test sample, find the most frequent label amongst neighbors
        y, n_votes = scipy.stats.mode(y_nn, axis=1)

        # Inverse transform the predicted labels
        y = self.label_encoder_.inverse_transform(y)

        return y

    def predict_proba(self, X):
        """Estimate the class probabilities for each input in X.

        Parameters
        ----------
        X : array, shape (n_samples_test, n_features)
            Testing inputs.

        Returns
        -------
        proba : array, shape(n_samples_test, n_classes)
            Class probabilities for each testing input.

        """

        self._validate_testing_inputs(X)

        # Find the nearest neighbors in the training set
        idx_nn, dist_nn = _find_nearest_neighbors(X, self.X_, self.k,
                                                  return_distance=True,
                                                  squared=False)

        # Probabilities can be obtained by using inverse distances as scores
        scores = np.exp(-dist_nn)

        # Convert scores to valid probabilities (softmax)
        p_neighbors = scores / scores.sum(axis=1)[:, None]

        # Find the (encoded) labels of the nearest neighbors
        y_nn = self.y_[idx_nn]

        # For each row in y_nn sum the one-hot encoded labels
        n_samples = X.shape[0]
        n_classes = len(self.label_encoder_.classes_)

        proba = np.zeros((n_samples, n_classes))
        for i, (y_row, p_row) in enumerate(zip(y_nn, p_neighbors)):
            one_hot_labels = self.label_binarizer_.transform(y_row)
            proba[i] = (one_hot_labels * p_row[:, None]).sum(axis=0)

        # Make sure the rows still sum up to 1.0 (numerical errors)
        proba = proba / proba.sum(axis=1)[:, None]

        return proba

    def score(self, y_pred, y_true):
        """Classification accuracy is the ratio of correct predictions.

        Parameters
        ----------
        y_pred : array, shape (n_samples, )
            Predicted labels.

        y_true : array, shape (n_samples, )
            Groundtrith labels.

        Returns
        -------
        score : The true positives rate.

        """
        check_consistent_length(y_pred, y_true)

        return np.equal(y_pred.ravel(), y_true.ravel()).sum() / y_pred.size

    def visualize(self, X, proba=None, **kwargs):
        """Scatter plot with color intensity proportional to class probability.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Data samples.

        proba : array, shape (n_samples, n_classes)
            Class probabilities for each sample.

        **kwargs : keyword arguments
            Other Parameters.

        """
        self._validate_testing_inputs(X)

        # We only plot the first two dimensions
        X = X[:, :2]

        n_classes = proba.shape[1]

        # Get rgba colors with tuned intensity (alpha)
        rgba, y_pred = proba_to_rgba(proba, return_most_likely=True)

        # Scatter the points with color
        class_names = self.label_encoder_.classes_

        class_handles = []
        for i in range(n_classes):
            class_mask = y_pred == i
            X_class = X[class_mask]
            ch = plt.scatter(X_class[:, 0], X_class[:, 1], c=rgba[class_mask])
            class_handles.append(ch)

        plt.xlabel('dim 1')
        plt.ylabel('dim 2')
        plt.title('KNN classification')
        plt.legend(class_handles, class_names, scatterpoints=1,
                   loc='upper right', ncol=1, fontsize=8)


def _find_nearest_neighbors(X, Y, k, return_distance=True, squared=False):
    """Find the nearest neighbors in Y of each element in X.

    Parameters
    ----------
    X : array, shape (n_samples_a, n_features)
        The reference set.

    Y : array, shape (n_samples_b, n_features)
        The query set.

    k : int
        The number of neighbors to find.

    return_distance : bool
        Whether to return the distances or not.

    squared : bool
        Whether to return the squared distances or the true distances.

    Returns
    -------
    idx_nn : array, shape (n_samples_a, k)
        The indices of the nearest neighbors in Y.

    distances : array, shape (n_samples_a, k)
        The distances to the nearerst neighbors in Y.

    """

    # Compute (squared) euclidean distances using the binomial formula:
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x^T*y
    X_norms_sq = np.sum(np.square(X), axis=1)
    Y_norms_sq = np.sum(np.square(Y), axis=1)
    X_times_Y = np.dot(X, Y.T)

    # Use numpy's broadcasting
    distances = X_norms_sq[:, None] - 2*X_times_Y + Y_norms_sq[None, :]

    # Numerical issues could allow negative distances
    np.maximum(distances, 0, out=distances)

    # distances has shape (n_samples_a, n_samples_b)
    # -> We only need the k smallest distances per row
    # -> We need to sort the rows
    # -> Actually we only need to sort the smallest k elements per row
    # -> It's more efficient to partition the k elements and sort only them
    idx = np.argpartition(distances, kth=k-1)

    # idx are indices referring to the query set [0, n_samples_b)
    # idx has shape (n_samples_a, n_samples_b)
    # Drop all columns after the k-th column
    idx = idx[:, :k]

    # Keep the k smallest distances per row
    sample_range = np.arange(len(idx))[:, None]
    distances = distances[sample_range, idx]

    # Now sort each row
    ind = np.argsort(distances)

    # ind refers to the position in distances namely it is in [0, k)
    # ind has shape (n_samples_a, k)
    # We can now sort the indices referring to the query set
    idx = idx[sample_range, ind]

    # If we are to return the distances we should sort them too.
    if return_distance:
        distances = distances[sample_range, ind]
        if squared:
            return idx, distances
        else:
            return idx, np.sqrt(distances)
    else:
        return idx


def main():

    # # 1. Read in the data
    X, y = load_iris(return_X_y=True)

    # 1a. Split in training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, train_size=0.7, random_state=42)

    # # 2. Do any necessary pre-processing (e.g. mean-centering)
    # from sklearn.preprocessing import StandardScaler
    # preprocessor = StandardScaler(with_mean=True, with_std=False)
    # preprocessor.fit(X_train)
    # X_train = preprocessor.transform(X_train)
    # X_test = preprocessor.transform(X_test)

    # # 3. Initialize your model (set the (hyper-) parameters)
    model = KNNClassifier(k=3)
    print(model)

    # # 4. Fit your model with the training data (This is the meat.)
    model.fit(X_train, y_train)

    # # 5. Evaluate your model on testing data (See what you learned.)
    y_pred = model.predict(X_test)
    test_accuracy = model.score(y_pred, y_test)
    print('Accuracy on {} test samples: {:5.2f}%.'
          .format(len(y_test), 100 * test_accuracy))

    # # 6. Optionally visualize something your model learned.
    # 6a. Scatter plot with color-coded confidences
    proba = model.predict_proba(X_test)
    model.visualize(X_test, proba)
    plt.show()


if __name__ == '__main__':
    main()