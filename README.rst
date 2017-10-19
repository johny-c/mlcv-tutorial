.. -*- mode: rst -*-

|Travis|_

.. |Travis| image:: https://api.travis-ci.org/johny-c/mlcv-tutorial.svg?branch=master
.. _Travis: https://travis-ci.org/johny-c/mlcv-tutorial

mlcv-tutorial
===============

Assisting library for the ML4CV tutorial based on scikit-learn.

It is recommended to use Python 3.6 in a virtual environment and install the 
latest stable versions of the dependencies. If not present,
`mlcv-tutorial` will attempt to install them automatically.

Installation
------------

Dependencies
~~~~~~~~~~~~

mlcv-tutorial requires:

- numpy (>= 1.13.3)
- scipy (>= 0.19.1)
- scikit-learn (>=0.19.0)
- requests (>=2.14.2)
- matplotlib (>=2.0.2)


User installation
~~~~~~~~~~~~~~~~~


1. Create a virtual environment. If you use ``pip``::

        python3 -m venv /path/to/new/virtual/environment_name

   or if you use ``conda``::

        conda create -n environment_name python=3.6 anaconda

2. Enter the virtual environment::

    source activate environment_name

3. Install or upgrade the package::

    pip install --upgrade git+https://github.com/johny-c/mlcv-tutorial.git

4. To exit the virtual environment::

    source deactivate

Usage
~~~~~

Enter the virtual environment you created. Upgrade regularly to get the latest
version. Open a python script, import the package and use it in your own work!

.. code-block:: python

    from mlcv.templates.base import Solution

    class MyEstimator(Solution):

        def __init__(param1=3, param2='gaussian'):
            # Store the passed parameters in your estimator instance
            self.param1 = param1
            self.param2 = param2

        def fit(X, y):
            # Train your estimator on the training inputs X and training targets y
            return self

        def predict(X):
            # Predict targets for the given testing inputs X.
            return y_pred

        def score(y_pred, y_true):
            # Evaluate your model
            return accuracy

