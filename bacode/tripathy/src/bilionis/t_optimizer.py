"""
    This file includes the loss functions.
    It also includes the logic which optimizes all parameters (sn, s, l, W)
    that occur within the process of the confidence-bounded model that wraps this
"""
import numpy as np
import math
import copy
import dill
# import multiprocess
import multiprocessing
import pathos

from .t_loss import loss
from .t_optimization_functions import t_ParameterOptimizer, t_WOptimizer
from .t_kernel import TripathyMaternKernel
from GPy.core.parameterization import Param

from bacode.tripathy.src.config import config


class TripathyOptimizer:

    def __init__(self):
        # PARAMETERS Algorithm 1
        self.d_max = config['max_dimensions']
        self.M_l = config['max_iter_alg1'] # 20 # 200 # 1000

        self.leps = config['eps_alg1'] #10.e-3
        self.m = config['max_iter_W_optimization']
        self.n = config['max_iter_parameter_optimization']

        self.no_of_restarts = config['no_restarts'] # 50

        # PARAMETERS Algorithm 4
        self.btol = config['eps_alg4']



        # For debugging / testing only
        self.losses = []
        self.dim_losses = []

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
    def find_active_subspace(self, X, Y):
        # Input dimension is always constant!
        D = X.shape[1]

        # TODO: Iteration by one might be a lot. Potentiall make the stepsize a function of the maximum dimensions

        BIC1 = -10000 # Don't make if -inf, otherwise division is not possible
        for d in range(1, min(D, self.d_max + 1)):

            print("Testing for dimension: ", d)

            self.kernel = TripathyMaternKernel(real_dim=D, active_dim=d)

            BIC0 = BIC1

            # Run entire optimize-code
            W_hat, sn, l, s = self.try_two_step_optimization_with_restarts(
                t_kernel=self.kernel,
                X=X,
                Y=Y
            )

            # Update the kernel with the new values
            self.kernel.update_params(W=W_hat, l=l, s=s)

            BIC1 = self.bic(
                kernel=self.kernel,
                W=W_hat,
                sn=sn,
                s=s,
                l=l,
                X=X,
                Y=Y
            )
            self.dim_losses.append(BIC1)

            print("Dimension loss is: ", BIC1)

            if (BIC1 - BIC0) / BIC0 < self.btol:
                print("Best found dimension is: ", d, BIC1, BIC0)
                break

        return W_hat, sn, l, s, d

    # Function to be run in parallel by multiple actors
    def single_run(self, t_kernel, X, Y):

        # Output
        W = None
        sn = None
        l = None
        s = None
        cur_loss = -np.inf

        try:

            # We first sample new weights and hyperparameters
            W_init = t_kernel.sample_W()
            l_init = np.random.rand(t_kernel.active_dim)
            s_init = float(np.random.rand(1))
            sn = float(np.random.rand(1))

            # print("Restarting...", (j, self.no_of_restarts))
            # TODO: At this point, just create a new kernel!

            t_kernel.update_params(W=W_init, l=l_init, s=s_init)
            # TODO: we need to copy the kernel before we apply any further operations!
            # TODO: Currently, the kernel is shared, so all the parameters are shared!

            # We do a deepcopy, because each time we start from the initial values
            W, sn, l, s = self.run_two_step_optimization(
                t_kernel=t_kernel,
                sn=sn,
                X=X,
                Y=Y
            )

            cur_loss = loss(
                t_kernel,
                W,
                sn,
                s,
                l,
                X,
                Y
            )

            print("Loss: ", cur_loss)

        except Exception as e:
            print(e)
            with open("./errors.txt", "a") as myfile:
                myfile.write(str(e))

        return W, sn, l, s, cur_loss

    def try_two_step_optimization_with_restarts(self, t_kernel, X, Y):

        losses = []
        configs = []

        # Define the throw-aways function:
        def wrapper_singlerun(_):
            return self.single_run(t_kernel, X, Y)

        # Defining the process pool
        # Count how many processors we have:
        number_processes = multiprocessing.cpu_count()
        # print("Number of processes found: ", number_processes)

        if config['restict_cores']:
            number_processes = min(number_processes - 1, config['max_cores'])
        number_processes = max(number_processes, 1)

        # print("Number of processes found: ", number_processes)

        pool = pathos.multiprocessing.Pool(number_processes)
        all_responses = pool.map(wrapper_singlerun, range(self.no_of_restarts))
        pool.close()
        pool.join()

        losses = [x[4] for x in all_responses]
        configs = [(x[0], x[1], x[2], x[3]) for x in all_responses]

        # print("Losses are: ", losses)

        best_index = int(np.argmax( losses ))
        best_config = configs[best_index]

        return best_config[0], best_config[1], best_config[2], best_config[3] #W, sn, l, s



    def run_two_step_optimization(self, t_kernel, sn, X, Y):

        # TODO: Here, we simply copy the W, and work on the same W all the time. This is a major bug!
        # Initial values
        self.W = t_kernel.W
        self.sn = sn
        self.s = t_kernel.inner_kernel.variance
        self.l = t_kernel.inner_kernel.lengthscale

        for i in range(self.M_l):

            if i % max(self.M_l//50, 1) == max(self.M_l//50, 1) // 2:
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
                # print("Old W: ", self.W)
                W = w_optimizer.optimize_stiefel_manifold(W=self.W.copy())
                # print("New W: ", W)
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

            for i in range(self.n):
                # print("\n\n\nOld s, l, sn", (self.s, self.l, self.sn))
                self.s, self.l, self.sn = parameter_optimizer.optimize_s_sn_l(
                    sn=self.sn,
                    s=self.s,
                    l=self.l.copy()
                )
                # print("\n\n\nNew s, l, sn", (self.s, self.l, self.sn))
                # print("self.l is: ", self.l)
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

            # print("Tuples is: ", (self.W, self.s, self.l, self.sn))

            self.losses.append(L1)

            # assert L0 < L01, (L0, L01)
            # assert L01 > L1, (L01, L1)
            # if len(self.losses) > 1:
            #     assert self.losses[-2] != self.losses[-1]

            # print(L0, L01, L1)

            #assert L0 != L01 TODO: check if there are any such conditions!

            if abs( (L1 - L0) / L0) < self.leps:
                print("Break Alg. 1", abs(L1 - L0) / L0)
                break

        return self.W, self.sn, self.l, self.s


    ###############################
    #        METRIC-FUNCTIONS     #
    ###############################
    def bic(self, kernel, W, sn, s, l, X, Y):
        s1 = loss(kernel, W, sn, s, l, X, Y)
        s2 = W.shape[0] * W.shape[1] + l.shape[0] + 1 # This counts how many parameters we have (is some form of regularization
        return s1 + s2