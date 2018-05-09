
# Constraint function (tau may be only between 0 and tau_max)

import math
import numpy as np
import scipy

from GPy.core.parameterization import Param
from GPy.models import GPRegression

from .t_loss import loss, dloss_dW, dK_dW

from .config import config

class t_ParameterOptimizer:
    """
        This class includes all the logic to opimize both
        the hyperparameters of the kernel, (s, l)
        and the hyperparameter of the GP-regression (sn, )
    """

    def __init__(self, fix_W, kernel, X, Y):
        ###############################
        #    PARAMETER-OPTIMIZATION   #
        ###############################
        self.fix_W = fix_W
        self.kernel = kernel
        self.kernel.update_params(
            W=self.fix_W,
            s=self.kernel.inner_kernel.variance,
            l=self.kernel.inner_kernel.lengthscale
        )
        self.X = X
        self.Y = Y

    def optimize_s_sn_l(self, sn, s, l):
        assert (isinstance(sn, float))
        assert (l.shape == (self.fix_W.shape[1],))

        # Create a GP
        self.kernel.update_params(W=self.fix_W, s=s, l=l)
        gp_reg = GPRegression(self.X, self.Y.reshape(-1, 1), self.kernel, noise_var=sn)
        try:
            gp_reg.optimize(optimizer="lbfgs", max_iters=config['max_iter_parameter_optimization'])
        except Exception as e:
            print(e)
            print(gp_reg.kern.K(gp_reg.X))
            print("Error above!")


        # TODO: does this optimization work in the correct direction?

        new_variance = gp_reg.kern.inner_kernel.variance
        new_lengthscale = gp_reg.kern.inner_kernel.lengthscale
        new_sn = gp_reg['Gaussian_noise.variance']

        assert gp_reg.kern.inner_kernel.lengthscale is not None
        assert gp_reg.kern.inner_kernel.variance is not None
        # assert not np.isclose(np.asarray(new_lengthscale), np.zeros_like(new_lengthscale) ).all(), new_lengthscale

        return float(new_variance), new_lengthscale.copy(), float(new_sn)


class t_WOptimizer:
    """
        This class includes all the logic
    """

    def __init__(self, kernel, fix_sn, fix_s, fix_l, X, Y):
        self.kernel = kernel
        self.fix_sn = fix_sn
        self.fix_s = fix_s
        self.fix_l = fix_l
        self.X = X
        self.Y = Y

        # TAKEN FROM CONFIG
        self.tau_max = config['tau_max'] #0.1 # 1e-3 :: this value finally seems to considerably change the loss!
        # TODO: check if any tau_delta does not depend on tau_max!
        self.stol = config['eps_alg3']

        self.tau = np.asscalar(np.random.rand(1)) * self.tau_max

        # FOR THE SAKE OF INCLUDING THIS WITHIN THE CLASS
        self.W = None
        self.all_losses = []
        self.M_s = config['max_iter_alg3'] # 500 # 10000

        self.no_taus = config['no_taus']

        # self.tau_arr = list( np.append(
        #     np.linspace(0., self.tau_max, num=self.no_taus),
        #     np.logspace(0., self.tau_max / 2, num=self.no_taus // 2)
        # ) )
        # assert len(self.tau_arr) == (self.no_taus + self.no_taus // 2), self.tau_arr

        self.tau_arr = np.linspace(0., self.tau_max, num=self.no_taus)

        self.tau_arr = [max(0., x) for x in self.tau_arr]
        self.tau_arr = [min(self.tau_max, x) for x in self.tau_arr]

        assert len(self.tau_arr) == self.no_taus

    #########################################
    #                                       #
    # All the derivative-specific functions #
    #                                       #
    #########################################


    ###############################
    #      STIEFEL-OPTIMIZATION   #
    ###############################
    def optimize_stiefel_manifold(self, W):

        self.W = W

        F_1 = loss(self.kernel, self.W, self.fix_sn, self.fix_s, self.fix_l, self.X, self.Y)

        for i in range(self.M_s):
            if i % 1 == 0:
                print("Alg. 3 Progress: " + str((i * 100) / self.M_s) + "%")
                print("Alg. 3: ", (i, self.M_s) )
            self.tau = self._find_best_tau(self.W)
            self.W = self._gamma(self.tau, self.W)

            self.kernel.update_params(W=self.W, l=self.fix_l, s=self.fix_s)

            F_0 = F_1
            F_1 = loss(self.kernel, self.W, self.fix_sn, self.fix_s, self.fix_l, self.X, self.Y)

            if abs( (F_1 - F_0) / F_0) < self.stol:
                break
        return self.W

    ###############################
    #          BRANCH 1           #
    ###############################
    def _gamma(self, tau, W):
        # print("Tau is: ", tau)
        assert W is not None, W
        assert (tau >= 0 - self.tau_max / 10.), (tau, self.tau_max)
        assert (tau <= self.tau_max + self.tau_max / 10.), (tau, self.tau_max)

        real_dim = W.shape[0]
        active_dim = W.shape[1]

        AW = 0.5 * tau * self._A(W)
        lhs = np.eye(real_dim) - AW
        rhs = np.eye(real_dim) + AW
        rhs = np.dot(rhs, W)
        out = np.linalg.solve(lhs, rhs)
        # out = np.dot(out, W)

        return out

    def _A(self, W):
        dL_dW = dloss_dW(self.kernel, W, self.fix_sn, self.fix_s, self.fix_l, self.X, self.Y)

        out = np.dot(dL_dW, W.T) # TODO: this does not fully conform with the equation!
        out -= np.dot(W, dL_dW.T)

        assert out.shape == (W.shape[0], W.shape[0])

        return out

    ###############################
    #          BRANCH 2           #
    ###############################
    #  Take this definition out of here
    def tau_manifold(self, tau, W):
        assert (not math.isnan(tau))
        assert W is not None
        W_tau = self._gamma(tau, W)
        loss_val = loss(self.kernel, W_tau, self.fix_sn, self.fix_s, self.fix_l, self.X, self.Y)
        self.all_losses.append(loss_val)
        return loss_val

    def _find_best_tau(self, W):

        # self.tau_manifold = np.vectorize(self.tau_manifold, excluded=1)

        assert isinstance(self.fix_s, float) or isinstance(self.fix_s, Param), type(self.fix_s)
        assert self.fix_l.shape == (W.shape[1],)

        assert (self.tau >= 0 - self.tau_max / 100.)
        assert (self.tau <= self.tau_max + self.tau_max / 100.)

        # tau_arr[tau_arr > self.tau_max] = self.tau_max

        # assert len(tau_arr) >= 5, len(tau_arr)

        # losses = np.asarray( list(map( lambda x: self.tau_manifold(x, W) , self.tau_arr)) )
        losses = [self.tau_manifold(x, W) for x in self.tau_arr]
        assert len(losses) == len(self.tau_arr)

        best_index = int( np.argmax( losses ) )
        best_tau = self.tau_arr[best_index]

        assert (not math.isnan(best_tau))

        return best_tau
