import sys

from febo.environment.benchmarks import Rosenbrock

sys.path.append("/Users/davidal/GoogleDrive/BachelorThesis/bacode/tripathy")
print(sys.path)
import numpy as np
from bacode.tripathy.src.bilionis_refactor.t_kernel import TripathyMaternKernel

from bacode.tripathy.src.bilionis_refactor.t_optimizer import TripathyOptimizer
from bacode.tripathy.src.bilionis_refactor.t_loss import loss

class TestIndividualFunctions(object):

    def init(self):
        self.real_dim = 3
        self.active_dim = 2
        self.no_samples = 5
        self.kernel = TripathyMaternKernel(self.real_dim, self.active_dim)

        self.sn = 2.

        self.W = self.kernel.sample_W()

        self.function = Rosenbrock()
        self.real_W = np.asarray([
            [0, 0],
            [0, 1],
            [1, 0]
        ])
        self.function = Rosenbrock()
        self.X = np.random.rand(self.no_samples, self.real_dim)
        Z = np.dot(self.X, self.real_W)
        self.Y = self.function.f(Z.T)

        self.optimizer = TripathyOptimizer()

    def test_two_step_optimization(self):
        self.init()

        L0 = loss(
            self.kernel,
            self.W,
            self.sn,
            self.kernel.inner_kernel.variance,
            self.kernel.inner_kernel.lengthscale,
            self.X,
            self.Y
        )

        W, sn, l, s = self.optimizer.run_two_step_optimization(
            t_kernel=self.kernel,
            sn=self.sn,
            X=self.X,
            Y=self.Y
        )

        L1 = loss(
            self.kernel,
            W,
            sn,
            s,
            l,
            self.X,
            self.Y
        )

        assert L0 < L1, (L0, L1)

    def test_finding_active_subspace_does_not_crash(self):
        self.init()

        # TODO: loss before, should be bigger than loss after!
        W_hat, sn, l, s, d = self.optimizer.find_active_subspace(self.X, self.Y)
        assert self.optimizer.dim_losses[0] >= self.optimizer.dim_losses[-1]