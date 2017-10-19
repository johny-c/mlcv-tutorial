import numpy as np
import os
from sklearn import datasets

DATA_DIR = os.path.split(os.path.realpath(__file__))[0]
DATASETS = ['iris', 'banknote', 'old_faithful', 'greetings']


def load_iris():
    """Load a data set for multi-class classification.

    Returns
    -------
    X : array (150, 4)
        The data samples.

    y : array (150,)
        The class labels.
    """
    return datasets.load_iris(return_X_y=True)


def load_banknote():
    """Load a data set for binary classification.

    Returns
    -------
    X : array (1348, 4)
        The data samples.

    y : array (1348,)
        The (binary) class labels.
    """

    data_path = os.path.join(DATA_DIR, 'banknote_auth_data.csv')
    label_path = os.path.join(DATA_DIR, 'banknote_auth_labels.csv')

    X = np.loadtxt(data_path, delimiter=',')
    y = np.loadtxt(label_path, dtype=str)

    return X, y


def load_old_faithful():
    """Load a data set for clustering.

    Returns
    -------
    X : array (272, 2)

    y : None

    """
    data_path = os.path.join(DATA_DIR, 'old_faithful.txt')
    X = np.loadtxt(data_path, skiprows=1, usecols=(1, 2))

    return X, None


def load_greetings():
    """Load a welcoming data set.

    Returns
    -------
    X : array (11324, 3)

    """

    data_path = os.path.join(DATA_DIR, 'greetings.txt')
    X = np.loadtxt(data_path, delimiter=',')

    return X, None


def load(dataset_name):
    """Load a data set.

    Parameters
    ----------
    dataset_name : str
        Name of the data set to load.

    Returns
    -------
    X : array, shape (n_samples, n_features)
        The data samples.

    y : array, shape (n_samples,) (optional)
        The data targets if there are any.

    """

    dataset_name = dataset_name.lower()
    if dataset_name not in DATASETS:
        raise ValueError('Dataset {} unknown.\nSupproted datasets:\n{}'
                         .format(dataset_name, DATASETS))

    if dataset_name == 'iris':
        return load_iris()
    elif dataset_name == 'banknote':
        return load_banknote()
    elif dataset_name == 'old_faithful':
        return load_old_faithful()
    elif dataset_name == 'greetings':
        return load_greetings()
