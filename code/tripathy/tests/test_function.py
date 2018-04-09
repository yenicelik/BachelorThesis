import sys
sys.path.append("/Users/davidal/GoogleDrive/BachelorThesis/code/tripathy")
print(sys.path)
import numpy as np
from src.t_kernel import TripathyMaternKernel
from src.t_optimization_functions import t_WOptimizer
from febo.environment.benchmarks.functions import Rosenbrock

class TestFunctions(object):

    def init(self):
        self.no_samples = 20
        self.function = Rosenbrock()
        self.real_dim = 2
        self.active_dim = 1
        self.function._set_dimension(self.active_dim)

        # Hide the matrix over here!
        # self.real_W = np.asarray([
        #     [0, 0],
        #     [0, 1],
        #     [1, 0]
        # ])

        self.real_W = np.asarray([
            [0,],
            [1,],
        ])

        self.sn = 0.8

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

