"""
    Assusme we always use the matern kernel!
    Especially for the part where we describe the kernel derivatives
"""
import numpy as np
import math
import sys

from GPy.models.gp_regression import GPRegression

def loss(kernel, W, sn, s, l, X, Y):
    """
    :param W: The orthogonal projection matrix
    :param sn: The noise-variance for the regression
    :param s: The kernel scalar hyperparameter
    :param l: The kernel lengthscales hyperparameter (array)
    :param X: The data to optimize over (observations)
    :param Y: The data to optimize over (target values)
    :return: A scalar value describing the "loss" of the given model
    """
    assert kernel.real_dim == X.shape[1]
    assert Y.shape[0] == X.shape[0]

    # Laziliy implementing GPy's function!
    # # TODO: check if the kernel inherits the correct methods
    kernel.update_params(W, l, s)
    Y = Y.reshape((-1, 1))
    gp_reg = GPRegression(X, Y, kernel, noise_var=sn)

    return gp_reg.log_likelihood()

#############################
# LOSS-FUNCTION-DERIVATIVES #
#############################
def dloss_dK_naked(kernel, W, sn, s, l, X, Y):
    kernel.update_params(W, l, s)

    # Y = Y.reshape((-1, 1))
    # gp_reg = GPRegression(X, Y, kernel, noise_var=sn)
    #
    # return gp_reg._log_likelihood_gradients()
    # kernel.update_params(W=W, l=l, s=s)
    #
    # # The matrix we are going to invert
    # res_kernel = kernel.K(X, X)
    #
    # K_sn = res_kernel + np.power(sn, 2) + np.eye(res_kernel.shape[0])
    #
    # # Calculate the cholesky-decomposition for the matrix K_sn
    # L = np.linalg.cholesky(K_sn)
    #
    # K_ss_inv = np.dot(np.linalg.inv(L.T), np.linalg.inv(L))
    # # K_ss_inv = np.linalg.inv(K_sn)
    #
    # # Calculate the displaced output
    #
    # # Calculate the first term
    # tmp_cholesky_inv = np.linalg.solve(L, Y)
    # lhs_rhs = np.linalg.solve(L.T, tmp_cholesky_inv)
    # #        lhs_rhs = np.linalg.solve(K_sn, Y_hat)
    #
    # s1 = np.dot(lhs_rhs, lhs_rhs.T)
    # s1 -= K_ss_inv
    #
    # return s1
    # Ignore the cholesky for now:
    K = kernel.K(X, X)
    KsnI = K + sn ** 2 * np.eye(X.shape[0])

    # Calculate a single term
    lhs = np.linalg.solve(KsnI, Y)

    summand1 = np.dot(KsnI, KsnI.T)
    summand2 = np.linalg.inv(KsnI)

    return summand1 - summand2


def dloss_ds(kernel, fix_W, fix_sn, s, fix_l, X, Y):
    # TODO: write some tests that check if changeing X or Y affect this derivative correctly!
    kernel.update_params(W=fix_W, s=s, l=fix_l)
    Y = Y.reshape((-1, 1))
    # The following line modifies `kernel`, so don't remove it
    gp_reg = GPRegression(X, Y, kernel, noise_var=fix_sn)
    grads = kernel.inner_kernel.variance.gradient

    return grads

def dloss_dW(kernel, W, fix_sn, fix_s, fix_l, X, Y):
    """
    The derivative of the loss functions up to a parameter "param"
    :param kernel:
    :param W:
    :param fix_sn:
    :param fix_s:
    :param fix_l:
    :param X:
    :param Y:
    :return:
    """
    kernel.update_params(W=W, l=fix_l, s=fix_s)
    Y = Y.reshape((-1, 1))
    gp_reg = GPRegression(X, Y, kernel, noise_var=fix_sn)

    # return gp_reg.log_likelihood()
    # print("Gradient dictionary is: ", [ key for key, value in gp_reg.grad_dict.items() ])
    # This outputs ['dL_dK', 'dL_dthetaL', 'dL_dm']

    assert kernel.W_grad.shape == W.shape

    return kernel.W_grad

#    return grads

    # full_K_W = dK_dW(kernel, W, fix_sn, fix_s, fix_l, X)
    #
    # real_dim = W.shape[0]
    # active_dim = W.shape[1]
    #
    # # Create the matrix we're gonna output
    # grad_W = np.empty((W.shape[0], W.shape[1]))
    #
    # # for each entry within full_K_W, take the trace, and assign it to w_ij
    # for i in range(W.shape[0]):
    #     for j in range(W.shape[1]):
    #
    #         # TODO: we don't quite calculate for all pairs of X and Y!
    #
    #         tmp = np.dot(naked_dloss_dK, full_K_W[i::real_dim, j::active_dim])
    #         grad_W[i, j] = 0.5 * np.matrix.trace(tmp)
    #
    # return grad_W

################################
#   DERIVATIVE w.r.t. KERNEL   #
################################
def dK_dW(kernel, W, sn, s, l, X):
    """
    :param x: Is assumed to be a vector!
    :param y: Is assumed to be a vector!
    :param W:
    :return:
    """
    # TODO: remove this function!
    kernel.update_params(W, l, s)

    real_dim = W.shape[0]
    active_dim = W.shape[1]

    def dk_dw_ij(a, b):
        z1 = np.dot(a, W)
        z2 = np.dot(b, W)

        W_grad = np.zeros((W.shape[0], W.shape[1]))

        # How to vectorize this function!
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_grad[i, j] = 2 * (z1[j] - z2[j]) * (a[i] - b[i])

        assert W_grad.shape == W.shape, str((W_grad.shape, W.shape))

        return W_grad

    # What we're going to output
    out = np.empty((X.shape[0] * real_dim, X.shape[0] * active_dim))

    # Create the gram-matrix using the above kernel-derivative
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            out[i*real_dim:(i+1)*real_dim, j*active_dim:(j+1)*active_dim] = dk_dw_ij(X[i, :], X[j, :])

    # If you want to use this for the loss-function, you need to slice only the inputs that one is interested in!

    return out