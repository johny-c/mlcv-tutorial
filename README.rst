mlcv-tutorial
===============

Assisting library for the ML4CV tutorial.

It is recommended to use Python 3.6 in a virtual environment and install the 
latest stable versions of the following packages:

* numpy
* scipy
* scikit-learn
* pandas
* requests
* matplotlib
* seaborn

Usage
-----

1. Create a virtual environment:

    `python3 -m venv /path/to/new/virtual/environment_name` (if you use pip)
    
    `conda create -n environment_name python=3.6 anaconda` (if you use conda)

2. Enter the virtual environment:

    `source activate environment_name`

3. Install or update the package:

    `pip install --upgrade git+http://github.com/johny-c/mlcv-tutorial.git --user`


4. Import the package and use it in some of your work.

.. code-block:: python

    from mlcv.datasets import fetcher
    from mlcv.examples.nearest_neighbors import KNNClassifier
    from sklearn.model_selection import train_test_split

    # Load a data set
    X, y = fetcher.load_iris()

    # Split in train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    # Instantiate a model you want to use (learn)
    knn = KNNClassifier(k=3)

    # Train the model
    knn.fit(X_train, y_train)

    # See the parameters of the model
    print(knn)

    # Predict with the trained model
    y_pred = knn.predict(X_test)

    # Evaluate the trained model
    test_acc = knn.score(y_pred, y_test)
    print('3-nearest neighbors test accuracy: {:5.2f}%.'.format(test_acc*100))


5. Exit the virtual environment:

    `source deactivate`