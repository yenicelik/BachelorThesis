import sys
sys.path.append("/Users/davidal/GoogleDrive/BachelorThesis/code/tripathy")
print(sys.path)
import numpy as np
from src.t_kernel import TripathyMaternKernel

from src.t_optimization_functions import t_WOptimizer

class TestIndividualFunctions(object):

    def init(self):
        self.real_dim = 3
        self.active_dim = 2
        self.no_samples = 5
        self.kernel = TripathyMaternKernel(self.real_dim, self.active_dim)

        self.sn = 2.

        self.W = np.random.rand(self.real_dim, self.active_dim)

        self.X = np.random.rand(self.no_samples, self.real_dim)
        self.Y = np.random.rand(self.no_samples)

        self.w_optimizer = t_WOptimizer(
            self.kernel,
            self.sn,
            np.asscalar(self.kernel.inner_kernel.variance),
            self.kernel.inner_kernel.lengthscale,
            self.X, self.Y
        )

    def test__A_fnc(self):
        self.init()

        print("X value is: ", self.w_optimizer.X.shape)
        print(self.w_optimizer)

        res = self.w_optimizer._A(self.W)

        # TODO: check what dimension this is supposed to have!
        assert res.shape == (self.real_dim, self.real_dim)

    def test_gamma_returns_correct_dimensions(self):
        self.init()

        res = self.w_optimizer._gamma(1e-3, self.W)

        assert res.shape == (self.W.shape[0], self.W.shape[1])


class TestSmallProcesses(object):

    def init(self):
        self.real_dim = 3
        self.active_dim = 2
        self.no_samples = 5
        self.kernel = TripathyMaternKernel(self.real_dim, self.active_dim)

        self.W = self.kernel.sample_W()
        self.sn = 2.

        self.X = np.random.rand(self.no_samples, self.real_dim)
        self.Y = np.random.rand(self.no_samples)


        self.w_optimizer = t_WOptimizer(
            self.kernel,
            self.sn,
            np.asscalar(self.kernel.inner_kernel.variance),
            self.kernel.inner_kernel.lengthscale,
            self.X, self.Y
        )

    def test_find_best_tau_maximizes_loss(self):
        self.init()

        self.w_optimizer._find_best_tau(self.W)

        # Maximizing/Increasing the loss is successful
        # TODO: think about how to have a good measure to see if it actually decreases!
        print("All losses: ", self.w_optimizer.all_losses)
        # maybe do np.sum( np.diff(self.w_optimizer.all_losses) ) > 0 ? to check if it generally increased?
        assert self.w_optimizer.all_losses[0] <= self.w_optimizer.all_losses[-1] + 1e-6, self.w_optimizer.all_losses

    def test_optimize_stiefel_manifold_doesnt_err(self):

        self.init()

        # For the sake of testing, do 10 iterations
        self.w_optimizer.optimize_stiefel_manifold(self.W, 100)


