
# Constraint function (tau may be only between 0 and tau_max)

import math
import numpy as np
import scipy

from .t_loss import loss, dloss_dK, dloss_dW, dK_dW

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
        self.tau_max = 1e-3
        self.gtol = 1e-10

        self.tau = np.asscalar(np.random.rand(1)) * self.tau_max

        # FOR THE SAKE OF INCLUDING THIS WITHIN THE CLASS
        self.W = None
        self.all_losses = []

    #########################################
    #                                       #
    # All the derivative-specific functions #
    #                                       #
    #########################################


    ###############################
    #      STIEFEL-OPTIMIZATION   #
    ###############################
    def optimize_stiefel_manifold(self, W, m):

        self.W = W

        F_1 = loss(self.kernel, self.W, self.fix_sn, self.fix_s, self.fix_l, self.X, self.Y)

        for i in range(m):
            self.tau = self._find_best_tau(self.W)
            self.W = self._gamma(self.tau, self.W)

            self.kernel.update_params(W=self.W, l=self.fix_l, s=self.fix_s)

            F_0 = F_1
            F_1 = loss(self.kernel, self.W, self.fix_sn, self.fix_s, self.fix_l, self.X, self.Y)

            if (F_1 - F_0) / F_0 < self.gtol:
                break
        return self.W

    ###############################
    #          BRANCH 1           #
    ###############################
    def _gamma(self, tau, W):
            # print("Tau is: ", tau)
            assert (tau >= 0 - self.tau_max / 10.)
            assert (tau <= self.tau_max + self.tau_max / 10.)

            real_dim = W.shape[0]
            active_dim = W.shape[1]

            AW = self._A(W)
            lhs = np.eye(real_dim) - 0.5 * tau * AW
            rhs = np.eye(real_dim) + 0.5 * tau * AW
            out = np.linalg.solve(lhs, rhs)
            out = np.dot(out, W)

            return out

    def _A(self, W):
        # TODO: the dimensions of X and Y are weird! (for dK_dW)
        dK_grad_W = dloss_dW(self.kernel, W, self.fix_sn, self.fix_s, self.fix_l, self.X, self.Y)

        out = np.dot(dK_grad_W, W.T) # TODO: this does not fully conform with the equation!
        out -= np.dot(W, dK_grad_W.T)

        assert out.shape == (W.shape[0], W.shape[0])

        return out

    ###############################
    #          BRANCH 2           #
    ###############################
    def _find_best_tau(self, W):

        assert isinstance(self.fix_s, float)
        assert self.fix_l.shape == (W.shape[1],)  # TODO: what do I change this to?

        def tau_manifold(tau):
            # TODO: do i have to take the negative of the output?
            # TODO: check if the loss decreases!
            # TODO: should there be a "-1 * "or not?

            assert (not math.isnan(tau))
            W_tau = self._gamma(tau, W)
            loss_val = loss(self.kernel, W_tau, self.fix_sn, self.fix_s, self.fix_l, self.X, self.Y)
            self.all_losses.append(loss_val)
            return -1 * loss_val # -1 because scipy minimizes by convention (we want to maximize!)

        assert (self.tau >= 0 - self.tau_max / 100.)
        assert (self.tau <= self.tau_max + self.tau_max / 100.)

        res = scipy.optimize.minimize(
            tau_manifold, self.tau, method='SLSQP', options={
                'maxiter': 20,  # TODO: because we don't use the EGO scheme, we use this one...
                'disp': False,
                'ftol': 1e-24
            },
            bounds=((0, self.tau_max),)
        )

        assert (not math.isnan(res.x))

        return res.x
