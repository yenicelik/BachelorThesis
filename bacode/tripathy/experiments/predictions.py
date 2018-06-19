"""
    Each one of the following functions have the following parameters
:param X_train: The X values on which we train the Gaussian Process
:param Y_train: The Y values on which we trian the Gaussian Process
:param X_test: The X values on which we evaluate some loss function
:param Y_test: The Y values on which we evaluate some loss function
:param X_project: The X values for which we want to receive predictions
:return:
"""
import numpy as np

from bacode.tripathy.src.rembo.rembo_algorithm import RemboAlgorithm
from bacode.tripathy.src.boring.boring_algorithm import BoringGP
from bacode.tripathy.src.tripathy__ import TripathyGP

class BasePrediction:

    def train(self, X_train, Y_train):
        raise NotImplementedError

    def evaluate_numerical(self, X_test, Y_test):
        raise NotImplementedError

    def predict(self, X_project):
        raise NotImplementedError


####################################
# PREDICT GAUSSIAN FOR IND. MODELS #
####################################
# class PredictVanillaUCB(BasePrediction):
#     """
#         This should return the results that you get when predicting using real UCB
#     """
#
#     def __init__(self, domain):
#         self.algo = UCB()
#         self.algo.initialize(domain=domain)
#         self.data_added = False

class PredictRembo(BasePrediction):

    def __init__(self, domain):
        self.algo = RemboAlgorithm()
        self.algo.initialize(domain=domain)
        self.data_added = False

    def train(self, X_train, Y_train):
        assert not self.data_added, "Data was already added before! You should create a new Object!"
        self.algo.add_data({
            'x': X_train,
            'y': Y_train
        })
        self.data_added = True

    def predict(self, X_project):
        Y_hat = self.algo.gp.mean(X_project)
        return Y_hat

    def evaluate_numerical(self, X_test, Y_test):
        # Predict, then compare to Y_test
        Y_hat = self.predict(X_test)
        return .0

class PredictStiefelSimple(BasePrediction):

    def __init__(self, domain):
        self.algo = TripathyGP(domain, calculate_always=True)
        self.data_added = False

    def train(self, X_train, Y_train):
        assert not self.data_added, "Data was already added before! You should create a new Object!"
        self.algo.add_data(X_train, Y_train)
        self.data_added = True

    def predict(self, X_project):
        Y_hat = self.algo.mean(X_project)
        return Y_hat

    def evaluate_numerical(self, X_test, Y_test):
        # Predict, then compare to Y_test
        Y_hat = self.predict(X_test)
        return .0

class PredictBoring(BasePrediction):

    def __init__(self, domain):
        self.algo = BoringGP(domain, always_calculate=True)
        self.data_added = False

    def train(self, X_train, Y_train):
        assert not self.data_added, "Data was already added before! You should create a new Object!"
        self.algo.add_data(X_train, Y_train)
        self.data_added = True

    def predict(self, X_project):
        Y_hat = self.algo.mean(X_project)
        return Y_hat

    def evaluate_numerical(self, X_test, Y_test):
        # Predict, then compare to Y_test
        Y_hat = self.predict(X_test)
        return .0


def train_and_predict_all_models(X_train, Y_train, X_test, Y_test, domain):
    # Rembo
    rembo = PredictRembo(domain)
    rembo.train(X_train, Y_train)
    rembo_yhat = rembo.predict(X_test)

    # Vanilla Tripathy # Does not contain the "kernel" bug
    tripathy = PredictStiefelSimple(domain)
    tripathy.train(X_train, Y_train)
    tripathy_yhat = tripathy.predict(X_test)

    # Boring # I think the error of UserWarning:Your kernel has a different input dimension 1 then the given X dimension 2. Be very sure this is what you want and you have not forgotten to set the right input dimenion in your kernel
    boring = PredictBoring(domain)
    boring.train(X_train, Y_train)
    boring_yhat = boring.predict(X_test)
    # boring_yhat = np.random.rand(X_test.shape[0], 1)

    return rembo_yhat, tripathy_yhat, boring_yhat
