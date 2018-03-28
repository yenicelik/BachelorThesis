"""
    This file includes the loss functions.
    It also includes the logic which optimizes all parameters (sn, s, l, W)
    that occur within the process of the confidence-bounded model that wraps this
"""
import numpy as np
import math

from .t_loss import loss
from .t_optimization_functions import t_ParameterOptimizer, t_WOptimizer

class TripathyOptimize:

    def __init__(self):
        # PARAMETERS
        self.d_max = 10
        self.max_iter = 100

        self.stiefel_max_steps = 10
        self.param_max_steps = 10

        self.btol = 1e-6
        self.ftol = 1e-6

    ###############################
    #      GENERAL-OPTIMIZATION   #
    ###############################
    def find_active_subspace(self, init_W, init_sn, init_s, init_l, X, Y):
        BIC1 = -100000
        for d in range(self.d_max):

            BIC0 = BIC1
            BIC1 = self.bic(d, init_W, init_sn, init_s, init_l, X, Y)

            # Run entire optimize-code
            self.run_two_step_optimization_once(d)

            if BIC1 - BIC0 / BIC0 < self.btol:
                print("Best found dimension is: ", d, BIC1, BIC0)
                break

    def run_two_step_optimization(self, d):
        L0 = loss(W, sn, s, l, X, Y)
        L1 = L0 + self.ftol * 1e5

        for i in range(self.max_iter):
            print("Step " + str(i) + " within the two-step-optimizer ")

            # Optimize over W (within the stiefel manifold)
            w_optimizer = t_WOptimizer(
                fix_sn=fix_sn,
                fix_s=fix_s,
                fix_l=fix_l,
                X=X,
                Y=Y)
            W = w_optimizer.optimize_stiefel_manifold(
                W=W,
                m=self.stiefel_max_steps
            )

            # TODO: here, we could simply call GP.optimize (with the correct kernel!)
            # TODO: we can then retrieve the variance and lengthscales using .variance, .lengthscales
            # (instead of updating the regression paramateres, we call
            # Optimize over all other parameters ()

            # TODO: Possibly just call .optimize?
            # In that case, we have to have W saved somewhere as a fixed variable within the GP objetc (the kernel object)

            theta_optimizer = t_ParameterOptimizer(
                fix_W=fix_W,
                kernel=kernel
            )
            sn, s, l = theta_optimizer.optimize_sn_l(
                sn=sn,
                s=s,
                l=l,
                X=X,
                Y=Y,
                n=self.param_max_steps
            )

            L0 = L1
            L1 = loss(self.W, self.sn, self.s, self.l, self.gp.X, self.gp.Y)
            if (np.abs(L1 - L0) / L0) < self.ftol:
                break

        return self.W, self.sn, self.l


    ###############################
    #        METRIC-FUNCTIONS     #
    ###############################
    def bic(self, d, W, sn, s, l, X, Y):
        s1 = self._loss(self.W, self.sn, self.s, self.l, self.gp.X, self.gp.Y)
        s2 = self.real_dim * d + self.l.shape[0] + 1
        return s1 + s2