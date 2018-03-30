"""
    Assusme we always use the matern kernel!
    Especially for the part where we describe the kernel derivatives
"""
import numpy as np
import math
import sys

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
    # TODO: check if the kernel inherits the correct methods
    # TODO: I'm sure GPy has this already implemented

    # Set the weight of the kernel to W
    # TODO: keep these auxiliary for now
    kernel.set_W(W)
    kernel.set_l(l)
    kernel.set_s(s)
    # TODO: also somehow set the variance!
    # TODO: Apply these operations before calling this function!

    # The matrix we are going to invert
    res_kernel = kernel.K(X, X)
    K_sn = res_kernel + np.power(sn, 2) * np.eye(res_kernel.shape[0])

    # Calculate the cholesky-decomposition for the matrix K_sn
    L = np.linalg.cholesky(K_sn)

    # Solve the system of equations K_sn^{-1} s1 = Y_hat
    # Using the cholesky decomposition of K_sn = L^T L
    # So the new system of equations becomes
    lhs = np.linalg.solve(L, Y)
    s1 = np.linalg.solve(L.T, lhs)
    s1 = np.dot(Y.T, s1)
    s2 = np.log(np.matrix.trace(L)) + X.shape[0] * np.log(2. * np.pi)

    out = (-0.5 * (s1 + s2)).flatten()
    out = np.asscalar(out)
    assert isinstance(out, float), str(out)
    assert not math.isnan(out), str(out)
    return out

# TODO: only calculate dloss_dW by hand, and finish it once and for all.

#############################
# LOSS-FUNCTION-DERIVATIVES #
#############################
def dloss_dK(kernel, W, sn, s, l, X, Y):
    loss_matrix = dloss_dK_naked(kernel, W, sn, s, l, X, Y)
    return 0.5 * np.matrix.trace(loss_matrix)

def dloss_dK_naked(kernel, W, sn, s, l, X, Y):
    kernel.set_W(W)
    kernel.set_l(l)
    kernel.set_s(s)
    # TODO: Apply these operations before calling this function!

    # The matrix we are going to invert
    res_kernel = kernel.K(X, X) # TODO: check or create a function that calculates the gram-matrix

    K_sn = res_kernel + np.power(sn, 2) + np.eye(res_kernel.shape[0])

    # Calculate the cholesky-decomposition for the matrix K_sn
    L = np.linalg.cholesky(K_sn)

    K_ss_inv = np.dot(np.linalg.inv(L.T), np.linalg.inv(L))
    # K_ss_inv = np.linalg.inv(K_sn)

    # Calculate the displaced output

    # Calculate the first term
    tmp_cholesky_inv = np.linalg.solve(L, Y)
    lhs_rhs = np.linalg.solve(L.T, tmp_cholesky_inv)
    #        lhs_rhs = np.linalg.solve(K_sn, Y_hat)

    s1 = np.dot(lhs_rhs, lhs_rhs.T)
    s1 -= K_ss_inv

    return s1


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
    # TODO: check this mathematically again!
    # Currently we're taking out the loss-term away from the trace term!
    naked_dloss_dK = dloss_dK_naked(kernel, W, fix_sn, fix_s, fix_l, X, Y)
    full_K_W = dK_dW(kernel, W, fix_sn, fix_s, fix_l, X)

    real_dim = W.shape[0]
    active_dim = W.shape[1]

    # Create the matrix we're gonna output
    grad_W = np.empty((W.shape[0], W.shape[1]))

    # for each entry within full_K_W, take the trace, and assign it to w_ij
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):

            tmp = np.dot(naked_dloss_dK, full_K_W[i::real_dim, j::active_dim])
            grad_W[i, j] = 0.5 * np.matrix.trace(tmp)

    return grad_W

################################
#   DERIVATIVE w.r.t. KERNEL   #
################################
# TODO: potentially just use the chain rule, and then multiple the vectors?
# TODO: Do i need to create a function that creates a gram matrix for this? (for each pair of samples?)

def dK_dW(kernel, W, sn, s, l, X):
    """
    :param x: Is assumed to be a vector!
    :param y: Is assumed to be a vector!
    :param W:
    :return:
    """
    # TODO: instead of the following, rather create a gram-matrix with the correct dimensions!
    # TODO: ask about this on stackoverflow!
    # kernel.inner_kernel.
    # TODO: check it in the code

    kernel.set_W(W)
    kernel.set_l(l)
    kernel.set_s(s)

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