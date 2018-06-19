"""
    Each one of the following functions have the following parameters
:param X_train: The X values on which we train the Gaussian Process
:param Y_train: The Y values on which we trian the Gaussian Process
:param X_test: The X values on which we evaluate some loss function
:param Y_test: The Y values on which we evaluate some loss function
:param X_project: The X values for which we want to receive predictions
:return:
"""
from bacode.tripathy.src.rembo.rembo_algorithm import RemboAlgorithm

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

    def __init__(self):
        pass

    def train(self, X_train, Y_train):
        raise NotImplementedError

    def evaluate_numerical(self, X_test, Y_test):
        raise NotImplementedError

    def predict(self, X_project):
        raise NotImplementedError

class PredictBoring(BasePrediction):

    def __init__(self):
        pass

    def train(self, X_train, Y_train):
        raise NotImplementedError

    def evaluate_numerical(self, X_test, Y_test):
        raise NotImplementedError

    def predict(self, X_project):
        raise NotImplementedError


def train_and_predict_all_models(X_train, Y_train, X_test, Y_test, domain):
    # Visualize Rembo
    rembo = PredictRembo(domain)
    rembo.train(X_train, Y_train)
    rembo_yhat = rembo.predict(X_test)
    return rembo_yhat
