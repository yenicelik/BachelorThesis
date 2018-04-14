import sys

from febo.environment.benchmarks import Rosenbrock

sys.path.append("/Users/davidal/GoogleDrive/BachelorThesis/code/tripathy")
print(sys.path)
import numpy as np
import warnings
from src.t_kernel import TripathyMaternKernel

from src.t_optimization_functions import t_WOptimizer
from src.t_loss import loss

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
        self.Y = self.function._f(Z.T)

        self.w_optimizer = t_WOptimizer(
            self.kernel,
            self.sn,
            np.asscalar(self.kernel.inner_kernel.variance),
            self.kernel.inner_kernel.lengthscale,
            self.X, self.Y
        )

    def test__A_fnc(self):
        self.init()

        res = self.w_optimizer._A(self.W)

        # TODO: check what dimension this is supposed to have!
        assert res.shape == (self.real_dim, self.real_dim)

    def test_gamma_returns_correct_dimensions(self):
        self.init()

        res = self.w_optimizer._gamma(1e-3, self.W)

        assert res.shape == (self.W.shape[0], self.W.shape[1])


class TestTauProcesses(object):

    def init(self):
        self.real_dim = 3
        self.active_dim = 2
        self.no_samples = 5
        self.kernel = TripathyMaternKernel(self.real_dim, self.active_dim)

        self.W = self.kernel.sample_W()
        self.sn = 2.

        self.function = Rosenbrock()
        self.real_W = np.asarray([
            [0, 0],
            [0, 1],
            [1, 0]
        ])
        self.function = Rosenbrock()
        self.X = np.random.rand(self.no_samples, self.real_dim)
        Z = np.dot(self.X, self.real_W)
        self.Y = self.function._f(Z.T)

        self.w_optimizer = t_WOptimizer(
            self.kernel,
            self.sn,
            np.asscalar(self.kernel.inner_kernel.variance),
            self.kernel.inner_kernel.lengthscale,
            self.X, self.Y
        )

    def test_tau_trajectory_determines_W(self):
        self.init()
        no_samples = 20

        W_init = self.kernel.sample_W()
        all_Ws = []

        tau_delta = 1e-5 #1e-4 is optimal!
        tau_0 = 0
        for i in range(no_samples):
            inp_tau = min( tau_0+i*tau_delta, self.w_optimizer.tau_max)
            new_W = self.w_optimizer._gamma(inp_tau, W_init)
            all_Ws.append(new_W)

        for i in range(no_samples-1):
            assert ((all_Ws[i] - all_Ws[i+1])**2).mean() >= 1e-16, str((i, all_Ws[i], all_Ws[i+1]))

    def test_tau_trajectory_determines_W_static(self):
        """
            Not changing tau keeps W the same
        :return:
        """
        self.init()
        no_samples = 20

        tau_0 = np.random.rand(1) * self.w_optimizer.tau_max

        for i in range(no_samples):
            W_init = self.kernel.sample_W()
            new_W1 = self.w_optimizer._gamma(tau_0, W_init)
            new_W2 = self.w_optimizer._gamma(tau_0, W_init)
            assert (new_W1 == new_W2).all()

    def test__find_best_tau_finds_a_better_tau(self):
        # TODO: semi-flaky test!
        self.init()

        W_init = self.kernel.sample_W()

        init_tau = 5e-5
        W_init = self.w_optimizer._gamma(init_tau, W_init)

        new_tau = self.w_optimizer._find_best_tau(np.copy(W_init))
        new_W = self.w_optimizer._gamma(new_tau, W_init)

        old_loss = loss(
            self.kernel,
            W_init,
            self.sn,
            self.kernel.inner_kernel.variance,
            self.kernel.inner_kernel.lengthscale,
            self.X,
            self.Y
        )
        new_loss = loss(
            self.kernel,
            new_W,
            self.sn,
            self.kernel.inner_kernel.variance,
            self.kernel.inner_kernel.lengthscale,
            self.X,
            self.Y
        )

        if old_loss == new_loss:
            warnings.warn("Old loss equals new loss! The W has not improved!")
            print("WARNING: Old loss equals new loss! The W has not improved!")
            print("Changes in W!")
            print(W_init)
            print(new_W)
            print("Losses")
            print(old_loss - new_loss)
#        assert abs(new_loss - old_loss) > 1e-12
        # TODO: This is a flaky test!
        assert new_loss >= old_loss # TODO: is this ok? It looks like it heavily depends on how W is initialized!

    def test_optimize_stiefel_manifold_doesnt_err(self):

        self.init()

        # For the sake of testing, do 10 iterations
        self.w_optimizer.optimize_stiefel_manifold(self.W)


class TestAProcesses(object):

    def init(self):
        self.real_dim = 3
        self.active_dim = 2
        self.no_samples = 5
        self.kernel = TripathyMaternKernel(self.real_dim, self.active_dim)

        self.W = self.kernel.sample_W()
        self.sn = 2.

        self.function = Rosenbrock()
        self.real_W = np.asarray([
            [0, 0],
            [0, 1],
            [1, 0]
        ])
        self.function = Rosenbrock()
        self.X = np.random.rand(self.no_samples, self.real_dim)
        Z = np.dot(self.X, self.real_W)
        self.Y = self.function._f(Z.T)

        self.w_optimizer = t_WOptimizer(
            self.kernel,
            self.sn,
            np.asscalar(self.kernel.inner_kernel.variance),
            self.kernel.inner_kernel.lengthscale,
            self.X, self.Y
        )

    def test__A_returns_correct_values(self):
        """
            Calculate this example by hand
        :return:
        """
        pass

    def test__gamma_returns_correct_values(self):
        """
            Calculate this example by hand
        :return:
        """
        pass


