
# Constraint function (tau may be only between 0 and tau_max)

import math
import numpy as np
import scipy

from GPy.core.parameterization import Param
from .t_loss import loss, dloss_dW, dK_dW

class t_ParameterOptimizer:
    """
        This class includes all the logic to opimize both
        the hyperparameters of the kernel, (s, l)
        and the hyperparameter of the GP-regression (sn, )
    """

    def __init__(self, fix_W, kernel):
        ###############################
        #    PARAMETER-OPTIMIZATION   #
        ###############################
        self.fix_W = fix_W
        self.kernel = kernel

    def optimize_sn_l(self, sn, s, l, X, Y, n):
        assert (isinstance(sn, float))
        assert (l.shape == (self.fix_W.shape[1],))

        # The Function we want to optimize
        # TODO: jacobian could be added for even better values
        def fnc(x):
            if x.shape != (4,):
                print("Shape does not conform!!", x.shape)
                assert (x.shape == (4,))
            x = x.flatten()
            return self._loss(W, x[0], x[1], x[2:], X, Y)

        x0 = np.insert(l, 0, s, axis=0).reshape((-1))
        x0 = np.insert(x0, 0, sn, axis=0).reshape((-1))

        res = scipy.optimize.minimize(
            fnc, x0, method="BFGS", options={
                "maxiter": n,
                "disp": False
            }
        )

        return res.x.flatten()[0], res.x.flatten()[1:]


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
        self.tau_max = 1 #0.1 # 1e-3 :: this value finally seems to considerably change the loss!
        # TODO: check if any tau_delta does not depend on tau_max!
        self.stol = 1e-16

        self.tau = np.asscalar(np.random.rand(1)) * self.tau_max

        # FOR THE SAKE OF INCLUDING THIS WITHIN THE CLASS
        self.W = None
        self.all_losses = []
        self.M_s = 10000

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
            self.tau = self._find_best_tau(self.W)
            self.W = self._gamma(self.tau, self.W)

            self.kernel.update_params(W=self.W, l=self.fix_l, s=self.fix_s)

            F_0 = F_1
            F_1 = loss(self.kernel, self.W, self.fix_sn, self.fix_s, self.fix_l, self.X, self.Y)

            if (F_1 - F_0) / F_0 < self.stol:
                break
        return self.W

    ###############################
    #          BRANCH 1           #
    ###############################
    def _gamma(self, tau, W):
            # print("Tau is: ", tau)
            assert (tau >= 0 - self.tau_max / 10.), (tau, self.tau_max)
            assert (tau <= self.tau_max + self.tau_max / 10.), (tau, self.tau_max)

            real_dim = W.shape[0]
            active_dim = W.shape[1]

            AW = self._A(W)
            lhs = np.eye(real_dim) - 0.5 * tau * AW
            rhs = np.eye(real_dim) + 0.5 * tau * AW
            out = np.linalg.solve(lhs, rhs)
            out = np.dot(out, W)

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
    def _find_best_tau(self, W):

        assert isinstance(self.fix_s, float) or isinstance(self.fix_s, Param), type(self.fix_s)
        assert self.fix_l.shape == (W.shape[1],)

        def tau_manifold(tau):
            assert (not math.isnan(tau))
            W_tau = self._gamma(tau, W)
            loss_val = loss(self.kernel, W_tau, self.fix_sn, self.fix_s, self.fix_l, self.X, self.Y)
            self.all_losses.append(loss_val)
            return loss_val

        assert (self.tau >= 0 - self.tau_max / 100.)
        assert (self.tau <= self.tau_max + self.tau_max / 100.)

        tau_arr = np.append(
            np.linspace(0., self.tau_max, num=50),
            np.logspace(0., self.tau_max, num=100)
        )
        tau_arr[tau_arr > self.tau_max] = self.tau_max
        tau_arr = np.unique(tau_arr)

        assert len(tau_arr) > 20

        best_loss = -np.inf
        best_tau = 0.
        for cur_tau in tau_arr:
            cur_loss = tau_manifold(cur_tau)
            if cur_loss > best_loss:
                best_loss = cur_loss
                best_tau = cur_tau

        print("New best tau and loss are: ", (best_tau, best_loss))

        assert (not math.isnan(best_tau))

        return best_tau
