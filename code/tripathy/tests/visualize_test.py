import sys
sys.path.append("/Users/davidal/GoogleDrive/BachelorThesis/code/tripathy")
print(sys.path)
import numpy as np
from src.t_kernel import TripathyMaternKernel
from src.t_optimization_functions import t_WOptimizer
from febo.environment.benchmarks.functions import Parabola

from GPy.models.gp_regression import GPRegression

class VisualizedTestingTau:

    def __init__(self):
        self.real_dim = 2
        self.active_dim = 1

        self.no_samples = 5
        self.kernel = TripathyMaternKernel(self.real_dim, self.active_dim)

        # Parameters
        self.sn = 2.
        self.W = self.kernel.sample_W()

        self.function = Parabola()
        self.real_W = np.asarray([
            [1],
            [1]
        ])
        self.real_W = self.real_W / np.linalg.norm(self.real_W, axis=1)

        # TODO: clean up code a little bit (take awayy variables that
        # are defined twice and all that stuff)

        self.X = np.random.rand(self.no_samples, self.real_dim)
        Z = np.dot(self.X, self.real_W)
        self.Y = self.function._f(Z.T)

        self.w_optimizer = t_WOptimizer(
            self.kernel, # TODO: does the kernel take over the W?
            self.sn,
            np.asscalar(self.kernel.inner_kernel.variance),
            self.kernel.inner_kernel.lengthscale,
            self.X, self.Y
        )

    def visualize_tau_trajectory_for_random_W(self):
        pass


    def visualize_tau_trajectory_for_identity_W(self):
        pass

class VisualizedTestingW:

    def __init__(self):
        self.real_dim = 3
        self.active_dim = 2

    def visualize_quadratic_function(self):
        pass


