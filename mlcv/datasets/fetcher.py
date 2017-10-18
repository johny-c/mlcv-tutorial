import numpy as np
import os
from sklearn import datasets

DATA_DIR = os.path.split(os.path.realpath(__file__))[0]
DATASETS = ['iris', 'banknote', 'old_faithful', 'greetings']


def load_iris():
    return datasets.load_iris(return_X_y=True)


def load_banknote():

    data_path = os.path.join(DATA_DIR, 'banknote_auth_data.csv')
    label_path = os.path.join(DATA_DIR, 'banknote_auth_labels.csv')

    X = np.loadtxt(data_path, delimiter=',')
    y = np.loadtxt(label_path, dtype=str)

    return X, y


def load_old_faithful():
    data_path = os.path.join(DATA_DIR, 'old_faithful.txt')
    X = np.loadtxt(data_path, skiprows=1, usecols=(1, 2))

    return X


def load_greetings():
    data_path = os.path.join(DATA_DIR, 'greetings.txt')
    X = np.loadtxt(data_path, delimiter=',')

    return X


def load(dataset_name):

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
