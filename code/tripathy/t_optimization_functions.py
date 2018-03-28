
# Constraint function (tau may be only between 0 and tau_max)

import math
import numpy as np
import scipy

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
        assert (l.shape == (2,))

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
                "disp": True
            }
        )

        return res.x.flatten()[0], res.x.flatten()[1:]


class t_WOptimizer:
    """
        This class includes all the logic
    """

    def __init__(self, fix_sn, fix_s, fix_l, X, Y):
        self.fix_sn = fix_sn
        self.fix_s = fix_s
        self.fix_l = fix_l
        self.X = X
        self.Y = Y

        # TAKEN FROM CONFIG
        self.tau_max = 1e-3
        self.gtol = 1e-6

        # FOR THE SAKE OF INCLUDING THIS WITHIN THE CLASS
        self.W = None

    #########################################
    #                                       #
    # All the derivative-specific functions #
    #                                       #
    #########################################


    ###############################
    #      STIEFEL-OPTIMIZATION   #
    ###############################
    def optimize_stiefel_manifold(self, W, m):
        F_1 = loss(W, self.fix_sn, self.fix_s, self.fix_l, self.X, self.Y)

        for i in range(m):
            tau = self._find_best_tau(W)
            self.W = self._gamma(tau, W)
            F_0 = F_1
            F_1 = loss(self.W, self.fix_sn, self.fix_s, self.fix_l, self.X, self.Y)

            if np.abs((F_1 - F_0) / F_0) < self.gtol:
                break
        return W

    ###############################
    #          BRANCH 1           #
    ###############################
    def _gamma(self, tau, W):
            print("Tau is: ", tau)
            assert (tau >= 0)
            assert (tau <= self.tau_max)

            AW = self._A(W)
            lhs = np.eye(self.real_dim) - 0.5 * tau * AW
            rhs = np.eye(self.real_dim) + 0.5 * tau * AW
            out = np.linalg.solve(lhs, rhs)
            out = np.dot(out, W)
            return out

    def _A(self, W):
        derivative = dloss_W(W, self.fix_sn, self.fix_s, self.fix_l, self.X, self.Y)
        return np.dot(derivative, W) - np.dot(W, derivative.T)

    ###############################
    #          BRANCH 2           #
    ###############################
    def _find_best_tau(self, W):

        assert (isinstance(self.fix_s, float))
        assert (self.fix_l.shape == (2,))  # TODO: what do I change this to?


        def tau_manifold(self, tau):
            # TODO: do i have to take the negative of the output?
            # TODO: check if the loss decreases!
            # TODO: should there be a "-1 * "or not?

            assert (not math.isnan(tau))
            W_tau = self._gamma(tau, W, self.fix_sn, self.fix_s, self.fix_l, self.X, self.Y)
            return -1 * self._loss(W_tau, self.fix_sn, self.fix_s, self.fix_l, self.X, self.Y)


        # Randomly sample tau!
        tau_0 = np.random.random_sample() * self.tau_max # TODO: only sample this the first time

        assert (tau_0 >= 0)
        assert (tau_0 <= self.tau_max)

        res = scipy.optimize.minimize(
            tau_manifold, tau_0, method='BFGS', options={
                'maxiter': 50,  # TODO: because we don't use the EGO scheme, we use this one...
                'disp': False
            },
            bounds=((0, self.tau_max),)
        )

        print(res.message)

        assert (not math.isnan(res.x))

        return res.x
