import numpy as np
import pandas as pd
from sklearn.utils import check_array, check_consistent_length
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from mlcv.templates.solution_template import Solution



class KNNClassifier(Solution):
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
        super(KNNClassifier, self).fit(X, y)

        self.X_ = X
        self.y_ = y

        return self

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

        check_array(X)



        pass


# def failing_case():
#
#     X_train = np.array([[1., 2.], [3., 4.]])
#     y_train = np.array([1, 0])
#
#     model = KNNClassifier(k=3)
#     print(model)
#
#     model.fit(X_train, y_train)

def main():

    ## 1. Read in the data
    # X_train = pd.read_csv('train_data.csv')
    # y_train = pd.read_csv('train_labels.csv')

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, train_size=0.7)

    # X_test = pd.read_csv('test_data.csv')
    # y_test = pd.read_csv('test_labels.csv')

    ## 2. Do any necessary pre-processing (e.g. mean-centering)
    # X_train, y_train = preprocess_inputs(X_train, y_train)
    # X_test, y_test = preprocess_inputs(X_test, y_test)

    # 3. Initialize your model (set the (hyper-) parameters)
    model = KNNClassifier(k=3)
    print(model)

    # 4. Make sure your parameters and your inputs are valid and compatible.
    # model._validate_training_inputs(X_train, y_train)
    # model.validate_input(X_test, y_test)

    ## 5. Fit your model with the training data (That's the meat...)
    model.fit(X_train, y_train)

    ## 6. Evaluate your model on some testing data (See what you learned.)
    # y_pred = model.predict(X_test)
    # model_score = model.score(y_pred, y_test)

    ## 7. Maybe plot something your model learned.
    # model.visualize(X_train, colors=model.param_5)


if __name__ == '__main__':
    main()