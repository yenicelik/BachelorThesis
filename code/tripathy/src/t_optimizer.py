"""
    This file includes the loss functions.
    It also includes the logic which optimizes all parameters (sn, s, l, W)
    that occur within the process of the confidence-bounded model that wraps this
"""
import numpy as np
import math

from .t_loss import loss
from .t_optimization_functions import t_ParameterOptimizer, t_WOptimizer
from GPy.core.parameterization import Param


class TripathyOptimizer:

    def __init__(self):
        # PARAMETERS
        self.d_max = 10
        self.M_l = 10000

        self.leps = 10.e-16
        self.m = 1

    ###############################
    #      GENERAL-OPTIMIZATION   #
    ###############################
    # TODO: instead of taking sn, s, and l, simply take the gaussian process.
    # then we can also simply call the 'optimize' function over it!
    # def find_active_subspace(self, init_W, init_sn, init_s, init_l, X, Y):
    #     BIC1 = -100000
    #     for d in range(self.d_max):
    #
    #         BIC0 = BIC1
    #         BIC1 = self.bic(d, init_W, init_sn, init_s, init_l, X, Y)
    #
    #         # Run entire optimize-code
    #         self.run_two_step_optimization_once(d)
    #
    #         if BIC1 - BIC0 / BIC0 < self.btol:
    #             print("Best found dimension is: ", d, BIC1, BIC0)
    #             break

    def run_two_step_optimization(self, t_kernel, sn, X, Y):

        for i in range(self.M_l):
            print("Alg. 1 Progress: ", str((i*100)/self.M_l) + "%")

            #################################################################################
            # PERFORM m ITERATIONS TOWARDS THE SOLUTION OF THE STIEFEL OPTIMIZATION PROBLEM #
            #################################################################################
            w_optimizer = t_WOptimizer(
                kernel=t_kernel,
                fix_sn=sn,
                fix_s=t_kernel.inner_kernel.variance,
                fix_l=t_kernel.inner_kernel.lengthscale,
                X=X,
                Y=Y)

            W = w_optimizer.kernel.W

            L0 = loss(
                w_optimizer.kernel,
                W,
                sn,
                w_optimizer.kernel.inner_kernel.variance,
                w_optimizer.kernel.inner_kernel.lengthscale,
                X,
                Y
            )

            for i in range(self.m):
                # TODO: the following optimizer should return the W for which the Loss is optimized.
                # NOT the W which was found at last
                W = w_optimizer.optimize_stiefel_manifold(W=W)
                t_kernel.update_params(W=W, l=t_kernel.inner_kernel.lengthscale, s=t_kernel.inner_kernel.variance)
                w_optimizer.kernel.update_params(
                    W=W,
                    l=w_optimizer.kernel.inner_kernel.lengthscale,
                    s=w_optimizer.kernel.inner_kernel.variance
                )

            #################################################################################
            #  PERFORM n ITERATIONS TOWARDS THE SOLUTION OF PARAMETER OPTIMIZATION PROBLEM  #
            #################################################################################
            L1 = loss(
                w_optimizer.kernel,
                W,
                sn,
                w_optimizer.kernel.inner_kernel.variance,
                w_optimizer.kernel.inner_kernel.lengthscale,
                X,
                Y
            )

            if abs(L1 - L0) / L0 < self.leps:
                print("Break Alg. 1", (L1, L0))
                break

        return W, sn, t_kernel.inner_kernel.lengthscale, t_kernel.inner_kernel.variance

    #
    #         # TODO: here, we could simply call GP.optimize (with the correct kernel!)
    #         # TODO: we can then retrieve the variance and lengthscales using .variance, .lengthscales
    #         # (instead of updating the regression paramateres, we call
    #         # Optimize over all other parameters ()
    #
    #         # TODO: Possibly just call .optimize?
    #         # In that case, we have to have W saved somewhere as a fixed variable within the GP objetc (the kernel object)
    #
    #         theta_optimizer = t_ParameterOptimizer(
    #             fix_W=fix_W,
    #             kernel=kernel
    #         )
    #         sn, s, l = theta_optimizer.optimize_sn_l(
    #             sn=sn,
    #             s=s,
    #             l=l,
    #             X=X,
    #             Y=Y,
    #             n=self.param_max_steps
    #         )
    #
    #         L0 = L1
    #         L1 = loss(self.W, self.sn, self.s, self.l, self.gp.X, self.gp.Y)
    #         if (np.abs(L1 - L0) / L0) < self.ftol:
    #             break
    #
    #     return self.W, self.sn, self.l


    ###############################
    #        METRIC-FUNCTIONS     #
    ###############################
    def bic(self, d, W, sn, s, l, X, Y):
        s1 = self._loss(self.W, self.sn, self.s, self.l, self.gp.X, self.gp.Y)
        s2 = self.real_dim * d + self.l.shape[0] + 1
        return s1 + s2