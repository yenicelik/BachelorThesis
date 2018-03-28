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
    kernel.setW(W)
    kernel.setl(l)
    kernel.sets(s)
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

    out = (-0.5 * (s1 + s2)).flatten()[0]
    assert (isinstance(out, float))
    assert (not math.isnan(out))
    return out

# TODO: only calculate dloss_dW by hand, and finish it once and for all.

#############################
# LOSS-FUNCTION-DERIVATIVES #
#############################
def dloss_dq(kernel, W, sn, s, l, X, Y, q="W"):
    if q == "W":
        kernel_derivative = dK_dW(s, l, W)
    else:
        print("Wrong parameter!")
        sys.exit(-1)

    f1 = dF_dparam(kernel, W, sn, s, l, X, Y)

    # TODO: The derivative is incorrect!
    # TODO: how to correctly implement this kernel operation for each vector within X?
    out = np.dot(f1, kernel_derivative)

    return -0.5 * np.matrix.trace(out)


def dloss_dparam(kernel, W, fix_sn, fix_s, fix_l, X, Y):
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
    kernel.setW(W)
    kernel.setl(fix_l)
    kernel.sets(fix_s)
    # TODO: Apply these operations before calling this function!

    # The matrix we are going to invert
    res_kernel = kernel.K(X, X)

    K_sn = res_kernel + np.power(fix_sn, 2) + np.eye(res_kernel.shape[0])

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


############################
#    KERNEL-DERIVATIVES    #
############################
def dK_dW(kernel, W, s, l, X):
    # TODO: create a gram-matrix of X!
    kernel

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

    #     # def _dK_dr(self, r, sn):
    #     #     return -3. * sn * r * np.exp(-np.sqrt(3.) * r)
    #




