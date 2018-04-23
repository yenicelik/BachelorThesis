import sys

from GPy.models import GPRegression
from febo.environment.benchmarks import Rosenbrock

sys.path.append("/Users/davidal/GoogleDrive/BachelorThesis/code/tripathy")
print(sys.path)
import numpy as np
import warnings
from src.t_kernel import TripathyMaternKernel

from src.t_optimization_functions import t_ParameterOptimizer
from src.t_loss import loss

class TestParameterOptimization(object):

    def init(self):
        self.real_dim = 3
        self.active_dim = 2
        self.no_samples = 5
        self.kernel = TripathyMaternKernel(self.real_dim, self.active_dim)

        self.real_sn = 0.01
        self.real_s = 0.1
        self.real_l = 0.1 * np.ones(self.active_dim)

        self.fix_W = self.kernel.sample_W()

        self.function = Rosenbrock()
        self.fix_W = np.asarray([
            [0, 0],
            [0, 1],
            [1, 0]
        ])
        self.function = Rosenbrock()
        self.X = np.random.rand(self.no_samples, self.real_dim)
        Z = np.dot(self.X, self.fix_W)
        self.Y = self.function._f(Z.T).reshape(-1, 1)

        print("Shape are: ", (self.X.shape, self.Y.shape))

        self.parameter_optimizer = t_ParameterOptimizer(
            self.fix_W,
            self.kernel,
            self.X,
            self.Y.T
        )

    def test_parameter_opt_does_not_err(self):
        self.init()

        # Have some initial guesses for sn, s, l
        s = float( np.random.rand(1) )
        sn = float( np.random.rand(1) )
        l = np.random.rand(self.active_dim)

        GPRegression(self.X, self.Y)

        self.kernel.update_params(W=self.fix_W, s=s, l=l)
        print("Got this far!")
        print("Previous parameters")
        print(s)
        print(l)
        print(sn)
        s, l, sn = self.parameter_optimizer.optimize_s_sn_l(sn, s, l, n=100)
        print("Got new parameters: ")
        print(s)
        print(l)
        print(sn)

    def test_parameter_opt_chages_parameters(self):
        self.init()
        pass


