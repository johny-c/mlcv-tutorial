import numpy as np
from matplotlib import pyplot as plt
from mlcv.datasets.fetcher import load_greetings
import os

def main():

    # # 1. Read in the data
    X, y = load_greetings()
    # y is None in this case

    # plot the data as scatter plot
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()


if __name__ == '__main__':
    main()