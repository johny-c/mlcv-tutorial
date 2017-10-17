import numpy as np
import os

DATA_DIR = os.path.split(__file__)[0]


def load_banknote():

    data_path = os.path.join(DATA_DIR, 'banknote_auth_data.csv')
    label_path = os.path.join(DATA_DIR, 'banknote_auth_labels.csv')

    X = np.loadtxt(data_path, delimiter=',')
    y = np.loadtxt(label_path, dtype=str)

    return X, y


def load_old_faithful():
    data_path = os.path.join(DATA_DIR, 'old_faithful.txt')

    print('Loading dataset from {}...'.format(data_path))
    X = np.loadtxt(data_path, skiprows=1, usecols=(1, 2))

    return X