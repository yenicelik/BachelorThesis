import sys
sys.path.append("/Users/davidal/GoogleDrive/BachelorThesis/bacode/tripathy")
print(sys.path)
import numpy as np
from bacode.tripathy.src.bilionis_refactor.t_kernel import TripathyMaternKernel
from febo.environment.benchmarks.functions import Parabola
from GPy.models.gp_regression import GPRegression

class TestFunctions(object):

    def init(self):
        self.no_samples = 20
        self.function = Parabola()
        self.real_dim = 2
        self.active_dim = 1
        self.function._set_dimension(self.active_dim)
        self.kernel = TripathyMaternKernel(self.real_dim, self.active_dim)

        # Hide the matrix over here!
        # self.real_W = np.asarray([
        #     [0, 0],
        #     [0, 1],
        #     [1, 0]
        # ])

        self.real_W = np.asarray([
            [1],
            [1]
        ])
        self.real_W = self.real_W / np.linalg.norm(self.real_W)

        self.X = np.random.rand(self.no_samples, self.real_dim)
        Z = np.dot(self.X, self.real_W)
        assert Z.shape == (self.no_samples, self.active_dim)
        self.Y = self.function.f(Z.T).reshape(self.no_samples,1)
        #assert self.Y.shape == (self.no_samples,)

        self.sn = 0.8

    def test_gp_regression(self):
        """
            The prediction of GPRegression sohuld be 1D!
        :return:
        """
        self.init()

        test_samples = 10

        Xrand = np.random.rand(test_samples, self.real_dim)

        # Check shape of GP
        gp_reg = GPRegression(
            self.X,
            self.Y,
            kernel=self.kernel,
            noise_var=self.sn
        )

        y_hat = gp_reg.predict(Xrand)[0] # Apparently, this predicts both mean and variance...

        assert y_hat.shape == (test_samples,1), y_hat.shape


    def test_function_returns_non_zeros(self):
        self.init()
        pass

        # X = np.random.rand(self.no_samples, self.real_dim)
        # Z = np.dot(X, self.real_W)
        # print("Input is: ")
        # print(Z.T)
        # Y = self.function._f(Z.T)
        #
        # print("Output is: ")
        # print(Y)
        #
        # assert not np.all(self.Y == 0), str(Y)

