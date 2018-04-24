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
        self.M_l = 1000

        self.leps = 10.e-3
        self.m = 1
        self.n = 1

        # For debugging / testing only
        self.losses = []

        # Everything should work through these values
        self.W = None
        self.sn = None
        self.l = None
        self.s = None

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

        # Initial values
        self.W = t_kernel.W
        self.sn = sn
        self.s = t_kernel.inner_kernel.variance
        self.l = t_kernel.inner_kernel.lengthscale

        for i in range(self.M_l):
            print("Alg. 1 Progress: ", str((i*100)/self.M_l) + "%")

            #################################################################################
            # PERFORM m ITERATIONS TOWARDS THE SOLUTION OF THE STIEFEL OPTIMIZATION PROBLEM #
            #################################################################################
            # Create the optimizers
            t_kernel.update_params(W=self.W, s=self.s, l=self.l)
            w_optimizer = t_WOptimizer(
                kernel=t_kernel,
                fix_sn=self.sn,
                fix_s=self.s,
                fix_l=self.l,
                X=X,
                Y=Y
            )

            L0 = loss(
                t_kernel,
                self.W,
                self.sn,
                self.s,
                self.l,
                X,
                Y
            )

            for i in range(self.m):
                # TODO: the following optimizer should return the W for which the Loss is optimized.
                # NOT the W which was found at last
                print("Old W: ", self.W)
                W = w_optimizer.optimize_stiefel_manifold(W=self.W.copy())
                print("New W: ", W)
                self.W = W
                t_kernel.update_params(
                    W=self.W,
                    l=self.l,
                    s=self.s
                )
            #################################################################################
            #  INTERMEDIATE LOSS
            ##################################################################################
            L01 = loss(
                t_kernel,
                self.W,
                self.sn,
                self.s,
                self.l,
                X,
                Y
            )

            #################################################################################
            #  PERFORM n ITERATIONS TOWARDS THE SOLUTION OF PARAMETER OPTIMIZATION PROBLEM  #
            #################################################################################

            parameter_optimizer = t_ParameterOptimizer(
                fix_W=self.W,
                kernel=t_kernel,
                X=X,
                Y=Y
            )

            # TODO: Check if instances (not that sth is copied wrongly etc. comply!
            for i in range(self.n):
                print("\n\n\nOld s, l, sn", (self.s, self.l, self.sn))
                self.s, self.l, self.sn = parameter_optimizer.optimize_s_sn_l(
                    sn=self.sn,
                    s=self.s,
                    l=self.l.copy()
                )
                print("\n\n\nNew s, l, sn", (self.s, self.l, self.sn))
                print("self.l is: ", self.l)
                t_kernel.update_params(
                    W=self.W,
                    s=self.s,
                    l=self.l
                )


            t_kernel.update_params(W=self.W, s=self.s, l=self.l)
            L1 = loss(
                t_kernel,
                self.W,
                self.sn,
                self.s,
                self.l,
                X,
                Y
            )

            print("Tuples is: ", (self.W, self.s, self.l, self.sn))

            self.losses.append(L1)

            # assert L0 < L01, (L0, L01)
            # assert L01 > L1, (L01, L1)
            # if len(self.losses) > 1:
            #     assert self.losses[-2] != self.losses[-1]

            print(L0, L01, L1)

            #assert L0 != L01

            if abs( (L1 - L0) / L0) < self.leps:
                print("Break Alg. 1", abs(L1 - L0) / L0)
                break

        return self.W, self.sn, self.l, self.s


    ###############################
    #        METRIC-FUNCTIONS     #
    ###############################
    def bic(self, d, W, sn, s, l, X, Y):
        s1 = self._loss(self.W, self.sn, self.s, self.l, self.gp.X, self.gp.Y)
        s2 = self.real_dim * d + self.l.shape[0] + 1
        return s1 + s2