"""
    This file includes the loss functions.
    It also includes the logic which optimizes all parameters (sn, s, l, W)
    that occur within the process of the confidence-bounded model that wraps this
"""
import numpy as np
import os
import math
import copy
import dill
# import multiprocess
import multiprocessing
import pathos
import time
import gc

from .t_loss import loss
from .t_optimization_functions import t_ParameterOptimizer, t_WOptimizer
from .t_kernel import TripathyMaternKernel
from GPy.core.parameterization import Param

from bacode.tripathy.src.bilionis_refactor.config import config
from copy import deepcopy

# We try not to use class functions, as we don't want to have shared memory!
# Function to be run in parallel by multiple actors
def single_run(self, t_kernel, X, Y):
    # t_kernel = deepcopy(t_kernel)

    # Output
    # W = None
    # sn = None
    # l = None
    # s = None
    # cur_loss = -np.inf

    print("Initializing single-run...")

    try:

        # We first sample new weights and hyperparameters
        W_init = t_kernel.sample_W()
        l_init = np.random.rand(t_kernel.active_dim)
        s_init = float(np.random.rand(1))
        sn = float(np.random.rand(1))

        # print("Restarting...", (j, self.no_of_restarts))

        # t_kernel.update_params(W=W_init, l=l_init, s=s_init)
        # TODO: we need to copy the kernel before we apply any further operations!
        # TODO: Currently, the kernel is shared, so all the parameters are shared!
        # t_kernel = None
        real_dim = t_kernel.real_dim
        active_dim = t_kernel.active_dim
        t_kernel = None
        t_kernel = TripathyMaternKernel(
            real_dim=real_dim,
            active_dim=active_dim,
            W=W_init,
            variance=s_init,
            lengthscale=l_init
        )

        # We do a deepcopy, because each time we start from the initial values
        W, sn, l, s = run_two_step_optimization(
            self,
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

        # except Exception as e:
        #     print("We encountered an error!")
        #     print(str(e))
        #     with open("./errors.txt", "a") as myfile:
        #         myfile.write(str(e))

        return W, sn, l, s, cur_loss

    except Exception as e:
        print(e)
        print("Some error happened!")
        return None, None, None, None, -1e13

def run_two_step_optimization(self, t_kernel, sn, X, Y, save_Ws=False, save_best_config=False):
    # First of all, deepcopy the kernel, such that any modifications are not thread-overlapping!

    # TODO: Here, we simply copy the W, and work on the same W all the time. This is a major bug!
    # Initial values
    W = t_kernel.W
    sn = sn
    s = t_kernel.inner_kernel.variance
    l = t_kernel.inner_kernel.lengthscale

    all_Ws = []
    self.losses = []

    best_config = (W, sn, s, l)

    for i in range(self.M_l):

        if i % max(self.M_l//5, 1) == max(self.M_l//20, 1):
            print("Alg. 1 Progress: ", str((i*100)/self.M_l) + "%")

        #################################################################################
        # PERFORM m ITERATIONS TOWARDS THE SOLUTION OF THE STIEFEL OPTIMIZATION PROBLEM #
        #################################################################################
        # Create the optimizers
        L0 = loss(
            t_kernel,
            W,
            sn,
            s,
            l,
            X,
            Y
        )

        t_kernel.update_params(W=W, s=s, l=l)
        w_optimizer = t_WOptimizer(
            kernel=t_kernel,
            fix_sn=sn,
            fix_s=s,
            fix_l=l,
            X=X,
            Y=Y
        )

        for j in range(self.m):
            # NOT the W which was found at last
            # print("Old W: ", self.W)
            print("Older W: ", W)
            W = w_optimizer.optimize_stiefel_manifold(W=W.copy())
            print("Newer W: ", W)
            # print("Newer W: ", W)
            # exit(0)
            self.W = W # TODO: Very weird!
            t_kernel.update_params(
                W=W,
                l=l,
                s=s
            )
        #################################################################################
        #  INTERMEDIATE LOSS
        ##################################################################################
        # L01 = loss(
        #     t_kernel,
        #     W,
        #     sn,
        #     s,
        #     l,
        #     X,
        #     Y
        # )

        #################################################################################
        #  PERFORM n ITERATIONS TOWARDS THE SOLUTION OF PARAMETER OPTIMIZATION PROBLEM  #
        #################################################################################
        parameter_optimizer = t_ParameterOptimizer(
            fix_W=W,
            kernel=t_kernel,
            X=X,
            Y=Y
        )

        # for j in range(self.n):
            # print("\n\n\nOld s, l, sn", (self.s, self.l, self.sn))
        print("Old parameters: ", (s, sn))
        s, l, sn = parameter_optimizer.optimize_s_sn_l(
            sn=sn,
            s=s,
            l=l.copy()
        )
        print("New parameters: ", (s, sn))
        # print("\n\n\nNew s, l, sn", (self.s, self.l, self.sn))
        # print("self.l is: ", self.l)
        t_kernel.update_params(
            W=W,
            s=s,
            l=l
        )

        t_kernel.update_params(W=W, s=s, l=l)
        L1 = loss(
            t_kernel,
            W,
            sn,
            s,
            l,
            X,
            Y
        )

        # print("Tuples is: ", (self.W, self.s, self.l, self.sn))

        # print("Because we cannot believe that the same W is chosen every time...")
        # print("Found W is: ", W)

        if save_Ws:
            all_Ws.append((W, sn, l, s))
        self.losses.append(L1)

        # assert L0 < L01, (L0, L01)
        # assert L01 > L1, (L01, L1)
        # if len(self.losses) > 1:
        #     assert self.losses[-2] != self.losses[-1]

        # print(L0, L01, L1)

        if  ( ((L1 - L0) / L0) < self.leps ) and ( i > 1 ):
            if L1 > L0 + 2*self.leps:
                best_config = (W, sn, l, s)
                continue
            print("Break Alg. 1", (L1 - L0) / L0)
            print("Break Alg. 1", L0, L1)
            print("Break Alg. 1", L1 - L0, L0)
            print("Breaking with i value: ", i)
            break

        best_config = (W, sn, l, s)

    if save_Ws and save_best_config:
        return all_Ws, best_config
    elif save_Ws:
        return all_Ws
    else:
        return W, sn, l, s

class TripathyOptimizer:

    def __init__(self):
        # PARAMETERS Algorithm 1
        self.d_max = config['max_dimensions']
        self.M_l = config['max_iter_alg1'] # 20 # 200 # 1000

        self.leps = config['eps_alg1'] #10.e-3
        self.m = config['max_iter_W_optimization']
        self.n = 1 #config['max_iter_parameter_optimization']

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

        # Initialize the pool here, so we can use it globally
        # Defining the process pool
        # Count how many processors we have:
        number_processes = multiprocessing.cpu_count()
        # print("Number of processes found: ", number_processes)

        if config['restict_cores']:
            number_processes = min(number_processes, config['max_cores'])
        number_processes = max(number_processes, 1)
        print("Number of cores in use: ", number_processes)
        print("The current process id is: ", os.getpid())

        self.pool = pathos.multiprocessing.ProcessingPool(number_processes)

    ###############################
    #      GENERAL-OPTIMIZATION   #
    ###############################
    # TODO: instead of taking sn, s, and l, simply take the gaussian process.
    # then we can also simply call the 'optimize' function over it!
    def find_active_subspace(self, X, Y, load=False):
        # Input dimension is always constant!
        if load:
            data = np.load(config['projection_datapath'] + "03_sinusoidal_ucb_hidden2d.npz")
            return data['W'], data['noise_var'], data['l'], data['var'], data['d']

        D = X.shape[1]

        # Output:
        W_hat = None
        sn = None
        l = None
        s = None
        d = None

        # TODO: Iteration by one might be a lot. Potentiall make the stepsize a function of the maximum dimensions

        BIC1 = -1e12 # Don't make if -inf, otherwise division is not possible
        for d in [config['active_dimension']]:
#        for d in range(1, max(2, min(D, self.d_max + 1))):

            print("Testing for dimension: ", d)

            self.kernel = TripathyMaternKernel(real_dim=D, active_dim=d)

            BIC0 = BIC1

            # Run entire optimize-code
            W_hat, sn, l, s = self.try_two_step_optimization_with_restarts(
                t_kernel=self.kernel,
                X=X,
                Y=Y
            )

            print("Returned W, sn, l values are: ", W_hat, sn, l, s)
            assert W_hat is not None
            assert sn is not None
            assert l is not None
            assert s is not None

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

            if abs( (BIC1 - BIC0) / BIC0) < self.btol:
                print("Best found dimension is: ", d, BIC1, BIC0)
                break

        return W_hat, sn, l, s, d

    def try_two_step_optimization_with_restarts(self, t_kernel, X, Y):

        losses = []
        configs = []

        # Define the throw-aways function:
        def wrapper_singlerun(_):
            return single_run(self, t_kernel, X, Y)


        # print("Number of processes found: ", number_processes)
        # Do "number of reruns by spawning new processes

        all_responses = self.pool.map(wrapper_singlerun, range(self.no_of_restarts))
        self.pool._clear()

        # pool.close()
        # pool.terminate()
        # pool.join()

        # pool = None

        # Run garbage collection
        gc.collect()

        for p in multiprocessing.active_children():
            p.terminate()
            gc.collect()

        print("We have so many active children: ", multiprocessing.active_children())

        # # Make sure we don't have any zombies
        # while True:
        #     act = multiprocessing.active_children()
        #     if len(act) > 0:
        #         print("Waiting for workers to finish: ", len(act))
        #     else:
        #         break
        #     time.sleep(2)

        losses = [x[4] for x in all_responses]
        configs = [(x[0], x[1], x[2], x[3]) for x in all_responses]

        print([x[0] for x in configs])

        # print("Losses are: ", losses)

        best_index = int(np.argmax( losses )) # TODO: do we take the argmax, or the argmin? argmax seems to work well!
        best_config = configs[best_index]

        # Check if those are empty!
        if best_config[0] is None:
            best_index = int(np.argsort(losses)[-2])
            best_config = configs[best_index]

        print("Best config is: ", best_config)

        return best_config[0], best_config[1], best_config[2], best_config[3] # W, sn, l, s


    ###############################
    #        METRIC-FUNCTIONS     #
    ###############################
    def bic(self, kernel, W, sn, s, l, X, Y):
        s1 = loss(kernel, W, sn, s, l, X, Y)
        s2 = W.shape[0] * W.shape[1] + l.shape[0] + 1 # This counts how many parameters we have (is some form of regularization
        return s1 + s2