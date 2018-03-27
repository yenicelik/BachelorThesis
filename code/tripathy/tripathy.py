
from febo.utils import locate

import math
import numpy as np

from febo.models import ConfidenceBoundModel
from febo.models.model import ModelConfig
from febo.models.gpy import GPRegression
from febo.utils.config import ConfigField, assign_config

class GPConfig(ModelConfig):
    kernels = ConfigField([('GPy.kern.RBF', {'variance': 2., 'lengthscale': 0.2 , 'ARD': True})])
    noise_var = ConfigField(0.1)
    _section = 'models.gp'

import os
print(os.path.dirname(os.path.abspath(__file__)))

def optimize_gp(experiment):
    experiment.algorithm.f.gp.kern.variance.fix()
    experiment.algorithm.f.gp.optimize()
    print(experiment.algorithm.f.gp)

@assign_config(GPConfig)
class TripathyModel(ConfidenceBoundModel):
    """
    Base class for GP optimization.
    Handles common functionality.
    Parameters
    ----------
    """

    def __init__(self, d):
        super(TripathyModel, self).__init__(d)

        # the description of a kernel
        self.kernel = None
        for kernel_module, kernel_params in self.config.kernels:
            kernel_part = locate(kernel_module)(input_dim=d, **kernel_params)
            if self.kernel is None:
                self.kernel = kernel_part
            else:
                self.kernel += kernel_part

        # calling of the kernel
        self.gp = GPRegression(d, self.kernel, noise_var=self.config.noise_var)
        # number of data points
        self.t = 0

    @property
    def beta(self):
        return math.sqrt(math.log(max(self.t,2))) # TODO: we could use the theoretical bound calculated from the kernel matrix.


    # @property
    # def data(self):
    #     """Return the data within the GP models."""
    #     x = self.gp.X.copy()
    #     y = np.empty((len(x), len(self.gps)), dtype=np.float)
    #
    #     for i, gp in enumerate(self.gps):
    #         y[:, i] = gp.Y.squeeze()
    #     return x, y


    def add_data(self, x, y):
        self.add_data_point_to_gps(x,y)

    def add_data_point_to_gps(self, x, y):
        """
        Add a new function observation to the GPs.
        Parameters
        ----------
        x: 2d-array
        y: 2d-array
        """
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        self.gp.append_XY(x, y)

        self.t += y.shape[1]

    def remove_last_data_point(self):
        """Remove the data point that was last added to the GP.
        Parameters
        ----------
            gp: Instance of GPy.models.GPRegression
                The gp that the last data point should be removed from
        """

        self.gp.remove_last_data_point()


    def mean_var(self, x):
        """Recompute the confidence intervals form the GP.
        Parameters
        ----------
        context: ndarray
            Array that contains the context used to compute the sets
        """
        return self.gp.predict_noiseless(x)

    def mean_var_grad(self, x):
        return self.gp.predictive_gradients(x)

    def var(self, x):
        return self.mean_var(x)[1]

    def mean(self, x):
        return self.mean_var(x)[0]

    def set_data(self, X, Y, append=True):
        if append:
            X = np.concatenate((self.gp.X, X))
            Y = np.concatenate((self.gp.Y, Y))
        self.gp.set_XY(X, Y)
        self.t = X.shape[0]



# import febo
# from febo.models import ConfidenceBoundModel
# from febo.utils.config import Config, ConfigField, assign_config
# from febo.utils import locate
#
# class TripathyConfig(Config):
#     num_layers = ConfigField(4, comment="Number of layers used")
#     num_neurons = ConfigField([100,100,50, 50], comment="Number of units in each layer.")
#     learning_rate = ConfigField('deep.learning_rate', comment="Function providing the learning rate.")
#     _section = 'deep.model'
#
# @assign_config(TripathyConfig)
# class Tripathy(ConfidenceBoundModel):
#
#     def __init__(self):
#         print("Initializing Tripathy model")




# import sys
# import numpy as np
# from scipy.ndimage import rotate
# import scipy.optimize._minimize
# from GPy.kern.src.sde_matern import sde_Matern32
#
# import math
#
# from .gpy import GPRegression
# from models import Model, ConfidenceBoundModel
#
#
# class ActiveStiefelSubspaceGP(ConfidenceBoundModel):
#     def __init__(self, d):
#         super(ActiveStiefelSubspaceGP, self).__init__(d)
#         self.MAX_ITER = 50
#
#         # TODO: assign each of these tolerances to one value
#         self.xtol = 1e-6
#         self.ftol = 1e-6  # Assume this is the stiefel-manifold function
#         self.gtol = 1e-12  # Assume this is the tau-function
#
#         # Prior mean
#         self.prior_mean = 0
#
#         # Additional variables
#         self.kernel_is_uptodate = False
#         self.t = 0
#         self.d = 2
#         self.real_dim = d
#
#         # Sample the following: W_0, phi_0, sn_0
#         self.W = self.sample_random_orth_matrix(self.real_dim, self.d)
#         self.sn = 0.1  # TODO: how to sample this value!
#         self.s = 2.
#         self.l = np.ones((self.real_dim,)) * 0.2
#
#         # Additional hyperparameters
#         self.tau_max = 1e-3
#
#         # self.beta = 0
#
#         # TODO: how to update the kernel values?
#         self.kernel = sde_Matern32(input_dim=d, variance=self.sn, lengthscale=self.l, ARD=True)
#         #self.kernel = RBF(input_dim=d, variance=self.sn, lengthscale=self.l, ARD=True)
#
#
#         # calling of the kernel
#         # TODO: change the kernel variance, and the kernel model!
#         # TODO: we are supposed to work on this kernel variance s_n! ### J
#         self.gp = GPRegression(d, self.kernel, noise_var=0.1)
#
#
#     #######################################
#     #                                     #
#     # All data-specific functions         #
#     #                                     #
#     #######################################
#     def add_data(self, x, y):
#         self.add_data_point_to_gps(x, y)
#
#         # W update
#         #self.run_two_step_optimization_once(2)
#
#         # TODO. updatemodel
#         self.W = np.eye(self.real_dim)
#         #self.W = np.rot90(self.W)
#
#         # Create everything new, that needs to be updated, including the dimension, etc.
#         tmpX = self.gp.X
#         tmpY = self.gp.Y
#         self.kernel = sde_Matern32(input_dim=self.real_dim, variance=self.s, lengthscale=self.l, ARD=True)
#         self.gp = GPRegression(self.real_dim, self.kernel, noise_var=self.sn)
#         self.gp.set_XY(tmpX, tmpY)
#
#
#     def add_data_point_to_gps(self, x, y):
#         # TODO: does thi actually add data to the gp?
#         """
#         Add a new function observation to the GPs.
#         Parameters
#         ----------
#         x: 2d-array
#         y: 2d-array
#         """
#         x = np.atleast_2d(x)
#         y = np.atleast_2d(y)
#         self.gp.append_XY(x, y)
#
#         self.t = self.gp.Y.shape[0]
#
#     def remove_last_data_point(self):
#         """Remove the data point that was last added to the GP.
#         Parameters
#         ----------
#             gp: Instance of GPy.models.GPRegression
#                 The gp that the last data point should be removed from
#         """
#         self.kernel_is_uptodate = False
#         self.gp.remove_last_data_point()
#         self.t = self.gp.Y.shape[0]
#
#     def set_data(self, X, Y, append=True):
#         if append:
#             X = np.concatenate((self.gp.X, X))
#             Y = np.concatenate((self.gp.Y, Y))
#         self.gp.set_XY(X, Y)
#         self.t = X.shape[0]
#
#     #######################################
#     #                                     #
#     # All GP-specific functions         #
#     #                                     #
#     #######################################
#     # Implementing everything that has to be implemented form the parent class
#     def var(self, X):
#
#         return self.mean_var(X)[1]
#
#     def mean(self, X):
#         # TODO: regardless of the mean value, the same regret is being returned!!!
#         return self.mean_var(X)[0]
#
#     def mean_var(self, X):
#         # K_ss = self._kernel_ss(X, X, W=self.W)
#         # K_s = self._kernel_s(X, left=False, W=self.W)
#         # K = self._kernel(W=self.W)
#         #
#         # # calculate the mean
#         # L = np.linalg.cholesky(K + self.sn * np.eye(K.shape[0]))
#         # Lk = np.linalg.solve(L, K_s)
#         # mean = np.dot(Lk.T, np.linalg.solve(L, self.gp.Y))
#         #
#         # # calculate the variance
#         # L = np.linalg.cholesky(K + self.sn * np.eye(K.shape[0]))
#         # Lk = np.linalg.solve(L, K_s)
#         # var = np.diag(K_ss) - np.sum(Lk ** 2, axis=0)
#
#         inp = np.dot(X, self.W) # TODO: this is incorrect! You have to multiple all the contents within gp.predict_noiseless also with W! but rn this is not the case!
#
#         return self.gp.predict_noiseless(inp)
#
# #        return mean.reshape(-1, 1), var.reshape(-1, 1)
#
#     @property
#     def beta(self):
#         return np.sqrt(np.log(max(self.t, 2)))
#
#     #######################################
#     #                                     #
#     # Finding the optimal subspace        #
#     #                                     #
#     #######################################
#     def bic(self, d):
#         # TODO: make sure sn was initialized etc!
#         s1 = self._loss(self.W, self.sn, self.s, self.l, self.gp.X, self.gp.Y)
#         s2 = self.real_dim * d + self.l.shape[0] + 1
#         return s1 + s2
#
#     def identify_AS(self, d_max, tol):
#         BIC1 = -100000
#
#         for d in range(d_max):
#
#             self.run_two_step_optimization_once(d)
#
#             # Calculate BIC
#             BIC0 = BIC1
#             BIC1 = self.bic(d)
#
#             if BIC1 - BIC0 / BIC0 < tol:
#                 print("Best found dimension is: ", d, BIC1, BIC0)
#                 break
#
#     #######################################
#     #                                     #
#     # All the algorithm-specific functions#
#     #                                     #
#     #######################################
#     def run_two_step_optimization_once(self, d):
#         MAX_ITER = 10000
#
#         # Compute initial loss (which should decrease with time)
#         L0 = self._loss(self.W, self.sn, self.s, self.l, self.gp.X, self.gp.Y)
#         L1 = self.ftol * 100 + L0
#
#         for iter in range(MAX_ITER):
#             print("Running two-step-optimizer", iter)
#
#             # Optimize for W!
#             # Uncomment this again!
#             self.W = self.optimize_stiefel_manifold(
#                 self.W,
#                 self.sn,
#                 self.l,
#                 self.gp.X,
#                 self.gp.Y,
#                 m=10
#             )
#
#
#             # Optimize for sn, l!
#             self.sn, self.l = self.optimize_sn_l(
#                 self.W,
#                 self.sn,
#                 self.l,
#                 self.gp.X,
#                 self.gp.Y,
#                 n=10
#             )
#
#             # Update the kernel to be used
#             # TODO: this ARD is probably wrong! Set this back, but find a way to set the individual length-scales!
#             #self.kernel = sde_Matern32(input_dim=self.real_dim, variance=self.sn, lengthscale=self.l, ARD=True)
#
#             L0 = L1
#             L1 = self._loss(self.W, self.sn, self.s, self.l, self.gp.X, self.gp.Y)
#             if (np.abs(L1 - L0) / L0) < self.ftol:
#                 break
#
#         return self.W, self.sn, self.l
#
#     #######################################
#     #                                     #
#     #  All the matrix-specific functions  #
#     #                                     #
#     #######################################
#     def sample_random_orth_matrix(self, D, d):
#         """
#         Returns: An orthogonal matrix
#         """
#         A = np.zeros((D, d), dtype=np.float64)
#         for i in range(D):
#             for j in range(d):
#                 A[i, j] = np.random.normal(0, 1)
#         Q, R = np.linalg.qr(A)
#         assert (np.allclose(np.dot(Q.T, Q), np.eye(Q.shape[0])))
#         return Q
#
#     def optimize_stiefel_manifold(self, W, fix_sn, fix_s, fix_l, X, Y, m):
#         F_1 = self._loss(W, fix_sn, fix_s, fix_l, X, Y)
#
#         for i in range(m):
#             tau = self._find_best_tau(W, fix_sn, fix_s, fix_l, X, Y)
#             W = self._gamma(tau, W, fix_sn, fix_s, fix_l, X, Y)
#             F_0 = F_1
#             F_1 = self._loss(self.W, fix_sn, fix_s, fix_l, X, Y)
#
#             if np.abs((F_1 - F_0) / F_0) < self.gtol:
#                 break
#         return W
#
#     #######################################
#     #                                     #
#     #  W-Optimization-specific functions  #
#     #                                     #
#     #######################################
#     def _find_best_tau(self, W, fix_sn, fix_s, fix_l, X, Y):
#         # TODO: the authors say, they used the EGO scheme. For us, let's just use L-BFGS!
#
#         assert (isinstance(fix_sn, float))
#         assert (fix_l.shape == (2,))  # TODO: what do I change this to?
#
#         # Constraint function (tau may be only between 0 and tau_max)
#
#         def fnc(tau):
#             # TODO: do i have to take the negative of the output?
#             # TODO: check if the loss decreases!
#             # TODO: should there be a "-1 * "or not?
#
#             assert (not math.isnan(tau))
#             W_depending_on_tau = self._gamma(tau, W, fix_sn, fix_s, fix_l, X, Y)
#             return -1 * self._loss(W_depending_on_tau, fix_sn, fix_s, fix_l, X, Y)
#
#         # Randomly sample tau!
#         tau_0 = np.random.random_sample() * self.tau_max
#
#         assert (tau_0 >= 0)
#         assert (tau_0 <= self.tau_max)
#
#         res = scipy.optimize.minimize(
#             fnc, tau_0, method='L-BFGS-B', options={
#                 'maxiter': 50,  # TODO: because we don't use the EGO scheme, we use this one...
#                 'disp': False
#             },
#             bounds=((0, self.tau_max),)
#         )
#
#         print(res.message)
#
#         assert (not math.isnan(res.x))
#
#         return res.x
#
#     # TODO: Check this function for math errors!
#     def _gamma(self, tau, W, fix_sn, fix_l, X, Y):
#         print("Tau is: ", tau)
#         assert (tau >= 0)
#         assert (tau <= self.tau_max)
#
#         AW = self._A(W, fix_sn, fix_l, X, Y)
#         lhs = np.eye(self.real_dim) - 0.5 * tau * AW
#         rhs = np.eye(self.real_dim) + 0.5 * tau * AW
#         out = np.linalg.solve(lhs, rhs)
#         out = np.dot(out, W)
#         return out
#
#     def _A(self, W, fix_sn, fix_l, X, Y):
#         """
#         Args:
#             tau:
#             W:
#             dF_dW: Must be a function, that is the derivative of F w.r.t. W
#         Returns:
#         """
#         derivative = self.dloss_W(W, fix_sn, fix_l, X, Y)
#         return np.dot(derivative, W) - np.dot(W, derivative.T)
#
#     #######################################
#     #                                     #
#     #  Optimization-specific functions    #
#     #                                     #
#     #######################################
#     def optimize_sn_l(self, W, sn, l, X, Y, n):
#
#         assert (isinstance(sn, float))
#         assert (l.shape == (2,))
#
#         # TODO: possibly add jacobian as an argument?
#         def fnc(x):
#             if x.shape != (4,):
#                 print("Shape does not conform!!", x.shape)
#                 assert (x.shape == (4,))
#             x = x.flatten()
#             return self._loss(W, x[0], x[1], x[2:], X, Y)
#
#         x0 = np.insert(l, 0, s, axis=0).reshape((-1))
#         x0 = np.insert(x0, 0, sn, axis=0).reshape((-1))
#
#         res = scipy.optimize.minimize(
#             fnc, x0, method='BFGS', options={
#                 'maxiter': n,
#                 'disp': True
#             }
#         )
#
#         return res.x.flatten()[0], res.x.flatten()[1:]
#
#     #######################################
#     #                                     #
#     #  All the loss-specific functions    #
#     #                                     #
#     #######################################
#     def _loss(self, W, sn, s, l, X, Y):
#
#         # Modify the kernel to use sn and l
#         self.kernel = sde_Matern32(input_dim=self.real_dim, variance=s, lengthscale=l,
#                                    ARD=True)
#
#         # The matrix we are going to invert
#         res_kernel = self._kernel(W=W)
#         K_sn = res_kernel + np.power(sn, 2) * np.eye(res_kernel.shape[0])
#
#         # Calculate the cholesky-decomposition for the matrix K_sn
#         L = np.linalg.cholesky(K_sn)
#
#         # Calculate the displaced output
#         Y_hat = Y - self.prior_mean
#
#         # Solve the system of equations K_sn^{-1} s1 = Y_hat
#         # Using the cholesky decomposition of K_sn = L^T L
#         # So the new system of equations becomes
#         lhs = np.linalg.solve(L, Y_hat)
#         s1 = np.linalg.solve(L.T, lhs)
#         s1 = np.dot(Y_hat.T, s1)
#         s2 = np.log(np.matrix.trace(L)) + self.t * np.log(2. * np.pi)
#
#         out = (-0.5 * (s1 + s2)).flatten()[0]
#         assert (isinstance(out, float))
#         assert (not math.isnan(out))
#         return out
#
#     # def dF_dW(self, W, fix_ns, fix_l, X, Y):
#     #     pass
#
#     def dF_dW(self, W, fix_sn, fix_l, X, Y):
#         # Modify the kernel to use sn and l
#         # TODO: this ARD is probably wrong!
#         self.kernel = sde_Matern32(input_dim=self.real_dim, variance=fix_sn, lengthscale=fix_l, ARD=True)
#
#         # The matrix we are going to invert
#         res_kernel = self._kernel(W=W)
#         K_sn = res_kernel + np.power(fix_sn, 2) + np.eye(res_kernel.shape[0])
#
#         # Calculate the cholesky-decomposition for the matrix K_sn
#         L = np.linalg.cholesky(K_sn)
#
#         K_ss_inv = np.dot(np.linalg.inv(L.T), np.linalg.inv(L))
#         # K_ss_inv = np.linalg.inv(K_sn)
#
#         # Calculate the displaced output
#         Y_hat = Y - self.prior_mean  # TODO: implement this using cholesky
#
#         # Calculate the first term
#         tmp_cholesky_inv = np.linalg.solve(L, Y_hat)
#         lhs_rhs = np.linalg.solve(L.T, tmp_cholesky_inv)
#         #        lhs_rhs = np.linalg.solve(K_sn, Y_hat)
#
#         s1 = np.dot(lhs_rhs, lhs_rhs.T)
#         s1 -= K_ss_inv
#         # TODO: how to correctly implement this kernel operation for each vector within X?
#         kernel_derivative = self.dK_dW(X, X, fix_sn, fix_l, W)
#         s1 = np.dot(s1, kernel_derivative)
#         return -0.5 * np.matrix.trace(s1)
#
#         # Functions to help the derivative by
#
#     # def _dK_dr(self, r, sn):
#     #     return -3. * sn * r * np.exp(-np.sqrt(3.) * r)
#
#     # def dK_dW(self, a, b, sn, l, W):
#     #
#     #     z1 = np.dot(a, W)
#     #     z2 = np.dot(b, W)
#     #
#     #     d = z1 - z2
#     #
#     #     f1 = self._dK_dr(np.dot(d, d.T), sn)
#     #
#     #     f2_1 = 2 * np.divide(z1 - z2, l)
#     #     f2_1 = np.dot(f2_1, a.T)  # TODO: this transpose feels terribly wrong!!
#     #     f2_2 = 2 * np.divide(z2 - z1, l)
#     #     f2_2 = np.dot(f2_2, b.T)
#     #
#     #     f2 = f2_1 + f2_2
#     #
#     #     return np.dot(f1, f2)
#
#     #########################################
#     #                                       #
#     # All the derivative-specific functions #
#     #                                       #
#     #########################################
#     def dloss_W(self, W, sn, l, X, Y):
#
#         # TODO: create the gram matrix here! you need to take every possible combination
#
#         # TODO: this implementation seems to be complete garbage!
#         # TODO: at lesat the usual input to this function seems to be complete garbage!!
#
#         def _dK_dr(r):
#             # TODO: should be set sn as a class attribute?
#             return -3. * sn * r * np.exp(-np.sqrt(3.) * r)
#
#         # TODO: usually, a=X, and b=X. This means that the following d=0! this is very probably wrong!
#         # Calculate the term dK_dW
#
#         # TODO: ### J
#         # Instead of this, one should calculate the gram matrix of all such distances!
#         z1 = np.dot(X, W) #previously this way the first input "a"
#         z2 = np.dot(X, W) #previously, this was the seocnd input "b" (instead of X)
#         d = z1 - z2
#
#         f1 = _dK_dr(np.dot(d, d.T), sn)
#
#         f2_1 = 2 * np.divide(z1 - z2, l)
#         # TODO: this transpose feels terribly wrong!!
#         f2_1 = np.dot(f2_1, X.T) #previously, this way the first input "a.T"
#         f2_2 = 2 * np.divide(z2 - z1, l)
#         f2_2 = np.dot(f2_2, X.T) #previously, this was the second input "b.T"
#
#         f2 = f2_1 + f2_2
#         kernel_term = np.dot(f1, f2)
#
#         # Now we apply the chain rule, where the first element is the general derivative loss term
#         out = self.dloss_dparamfnc(W, sn, l, X, Y, kernel_term)
#         return out
#
#     def dloss_dsn(self, W, sn, l, X, Y):
#         pass
#
#     def dloss_l(self, W, sn, l, X, Y):
#         pass
#
#     def dloss_dparamfnc(self, W, sn, l, X, Y, param_derivative):
#
#         # Modify the kernel (just in case) before applying it
#         # TODO: this ARD is probably wrong!
#         self.kernel = sde_Matern32(input_dim=self.real_dim, variance=sn, lengthscale=l, ARD=True)
#
#         # Calculate the matrix we will later inver (K + sn I)
#         res_kernel = self._kernel(W=W)
#         K_sn = res_kernel + np.power(sn, 2) + np.eye(res_kernel.shape[0])
#
#         # Calculate the cholesky-decomposition for the matrix K_sn
#         L = np.linalg.cholesky(K_sn)
#         K_ss_inv = np.dot(np.linalg.inv(L.T), np.linalg.inv(L))
#         # K_ss_inv = np.linalg.inv(K_sn)
#
#         # Calculate the displaced output
#         Y_hat = Y - self.prior_mean
#
#         # Calculate the first term
#         tmp_cholesky_inv = np.linalg.solve(L, Y_hat)
#         lhs_rhs = np.linalg.solve(L.T, tmp_cholesky_inv)
#         #        lhs_rhs = np.linalg.solve(K_sn, Y_hat)
#
#         s1 = np.dot(lhs_rhs, lhs_rhs.T)
#         s1 -= K_ss_inv
#         # TODO: how to correctly implement this kernel operation for each vector within X?
#
#         s1 = np.dot(s1, param_derivative)
#
#         # TODO: replace these values in other functions
#         #        kernel_derivative = self.dK_dW(X, X, sn, l, W)
#         #        s1 = np.dot(s1, kernel_derivative)
#         return -0.5 * np.matrix.trace(s1)
#
#     def dloss_dl(self, fix_W, fix_sn, l, X, Y):
#         pass
#
#     def dloss_dsn(self, fix_W, sn, fix_l, X, Y):
#         pass
#
#     #######################################
#     #                                     #
#     #  All the kernel-specific functions  #
#     #                                     #
#     #######################################
#     def _kernel_ss(self, a, b, W=None):
#         # Implementation of the matern-32 kernel:
#         if W is not None:
#             return self.kernel.K(np.dot(a, W), np.dot(b, W))
#         else:
#             return self.kernel.K(a, b)
#
#     def _kernel_s(self, a, left=True, W=None):
#         if left:
#             return self.kernel.K(np.dot(a, W), np.dot(self.gp.X, W)) if W is not None else self.kernel.K(a, self.gp.X)
#         else:
#             return self.kernel.K(np.dot(self.gp.X, W), np.dot(a, W)) if W is not None else self.kernel.K(self.gp.X, a)
#
#     def _kernel(self, W=None):
#         if W is not None:
#             inp = np.dot(self.gp.X, W)
#             self.kernel_result = self.kernel.K(inp, inp)
#         else:
#             self.kernel_result = self.kernel.K(self.gp.X, self.gp.X)
#         return self.kernel_result
#
#
#         # Simple implementation for the variance function
#         # def var()::::::
#         # lhs = self._kernel(self.W) + np.eye(self.gp.X.shape[0])
#         # rhs = self._kernel_s(X, left=False, W=self.W)
#         # solved = np.linalg.solve(lhs, rhs)
#         # right_summand = self._kernel_s(X, W=self.W)
#         # right_summand = np.dot(right_summand, solved)
#         # # print("Right summand has dimensions", right_summand.shape)
#         # # print("new kernel has dimensions", self._kernel_ss(X, X).shape)
#         # res = np.diag(self._kernel_ss(X, X, W=self.W)) - np.diag(right_summand)
#         # # TODO: implement the cholesky version!! this is waaay to slow
#
#         # def mean():::::
#         # lhs = self._kernel(self.W) + self.sn * np.eye(self.gp.X.shape[0])
#         # solved = np.linalg.solve(lhs, self.gp.Y)
#         # out = np.dot(self._kernel_s(X, W=self.W), solved)
#
#         # return res.reshape(-1, 1)
#
#         # return self.mean_var(X)[1]
#
#         #
#
#         # return self.mean(X), self.var(X)
#         # self.run_two_step_optimization_once(2)
#         # Z = np.dot(X, self.W) #TODO: is this correct like so?
#         # return self.gp.predict_noiseless(Z, kern=self.kernel)
