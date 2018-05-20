"""
    Here, we offer helper functions for the predictive gradients method.
    Where we apply PCA on $ \integral \delta f \delta f p(x) dx $.

    Implementing the theory from the paper: ﻿Active Learning of Linear Embeddings for Gaussian Processes, Garnett 2013
"""

from ..t_optimization_functions import t_ParameterOptimizer
from ..t_kernel import TripathyMaternKernel

def classical_active_subspace_find_optimal(X, Y):
    """
        Create the svd from
    :return:
    """
    D = X.shape[0]

    # Iterate through all dimensions until loss is not too big

    # Values to be returned at the end of the algorithm
    W_hat, sn, l, s, d, = None, None, None, None, None

    # Spawn kernel
    kernel = TripathyMaternKernel(real_dim=D, active_dim=d)

    # First, learn the projection matrix W from the data
    parameter_optimizer = t_ParameterOptimizer(fix_W=W_hat, kernel=kernel, X=X, Y=Y)


    return W_hat, sn, l, s, d


# class ClassicActiveSubspaceGPRegression(ActiveSubspaceGPRegression):
#     """
#     The object builds on the functionality of ``ActiveSubspaceGPRegression.
#     """
#
#     # The observed gradients
#     _G = None
#
#     @property
#     def G(self):
#         """
#         :getter: the observed gradients.
#         """
#         return self._G
#
#     def __init__(self, X, Y, G, inner_kernel, W=None, **kwargs):
#         """
#         Initialize the model.
#         """
#         super(ClassicActiveSubspaceGPRegression, self).__init__(X, Y, inner_kernel, W=W,
#                                                                 **kwargs)
#         assert X.shape[0] == G.shape[0]
#         assert X.shape[1] == G.shape[1]
#         self._G = G
#         # Compute W using SVD
#         U, s, V = svd(self.G)
#         self.kern.W = V[:self.kern.active_dim, :].T
#         self.kern.W.fix()
#
#     def optimize(self, **kwargs):
#         """
#         The options are the same as those of the classic ``GPRegression.optimize()``.
#         """
#         super(ActiveSubspaceGPRegression, self).optimize(**kwargs)