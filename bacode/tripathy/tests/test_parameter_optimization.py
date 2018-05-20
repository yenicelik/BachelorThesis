import sys

from GPy.models import GPRegression
from febo.environment.benchmarks import Rosenbrock

sys.path.append("/Users/davidal/GoogleDrive/BachelorThesis/code/tripathy")
print(sys.path)
import numpy as np
import warnings
from bacode.tripathy.src.t_kernel import TripathyMaternKernel

from bacode.tripathy.src.t_optimization_functions import t_ParameterOptimizer
from bacode.tripathy.src.t_loss import loss

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
        self.Y = self.function.f(Z.T).reshape(-1, 1)

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

        new_s, new_l, new_sn = self.parameter_optimizer.optimize_s_sn_l(sn, s, l)

        print(s, new_s)
        print(sn, new_sn)
        print(l, new_l)

        assert not np.isclose(new_s, s), (new_s, s)
        assert not np.isclose(new_l, l).all(), (new_l, l)
        assert not np.isclose(new_sn, sn), (new_sn, sn)

    def test_parameter_optimizes_loss(self):
        self.init()

        s_init = float( np.random.rand(1) )
        sn_init = float( np.random.rand(1) )
        l_init = np.random.rand(self.active_dim)

        old_loss = loss(
            self.kernel,
            self.fix_W,
            sn_init,
            s_init,
            l_init,
            self.X,
            self.Y
        )

        new_s, new_l, new_sn = self.parameter_optimizer.optimize_s_sn_l(sn_init, s_init, l_init)

        new_loss = loss(
            self.kernel,
            self.fix_W,
            new_sn,
            new_s,
            new_l,
            self.X,
            self.Y
        )

        # print("Old loss, new loss ", (old_loss, new_loss))
        # TODO: Should this be a new smaller value, or a value toward zero, or a new bigger value?
        # assert new_loss <= old_loss
        assert new_loss != old_loss

    def test_parameters_change(self):
        self.init()

        s_init = float( np.random.rand(1) )
        sn_init = float( np.random.rand(1) )
        l_init = np.random.rand(self.active_dim)

        old_loss = loss(
            self.kernel,
            self.fix_W,
            sn_init,
            s_init,
            l_init,
            self.X,
            self.Y
        )

        new_s, new_l, new_sn = self.parameter_optimizer.optimize_s_sn_l(sn_init, s_init, l_init)

        assert s_init != new_s, (s_init, new_s)
        assert not np.isclose(l_init, new_l).all(), (l_init, new_l)
        assert sn_init != new_sn, (sn_init, new_sn)



