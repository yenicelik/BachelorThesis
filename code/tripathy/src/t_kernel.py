"""
    Create the kernel class here.
    The resulting kernel must inherit from the GPy kernel class
"""
import numpy as np
import scipy
from GPy.kern.src.kern import Kern
from GPy.kern.src.stationary import Matern32
from GPy.core.parameterization import Param
from paramz.transformations import Logexp

from GPy.kern.src.stationary import Matern32
from paramz.caching import Cache_this

class TripathyMaternKernel(Matern32):

    """
    A kernel of the following form:
        .. math::
            k(x, x') = k_0(Wx, Wx')
    """

    def __init__(self, real_dim, active_dim):

        assert(real_dim >= active_dim)

        self.real_dim = real_dim
        self.active_dim = active_dim

        # TODO: decide what file to put these values (this file is probably good, just check how to pass around stuff)
        # TODO: add these as priors
        self.W = self.sample_W()
        self.l = self.sample_lengthscale()
        self.s = self.sample_variance()

        # TODO: find a way to change internal variables within the following object!

        # TODO: incorporate a way to include W as the kernel-parameter (e.g. call parent function, where x = W.T x)
        # TODO: incorporate a way to easily switch out all kernel-parameters (variance and lengthscale)

        # TODO: overwrite the kernel parameters!

        super(TripathyMaternKernel, self).__init__(self.active_dim, self.s, self.l, ARD=True)

    ###############################
    #       SETTER FUNCTIONS      #
    ###############################
    def set_W(self, W):
        assert(W.shape == (self.real_dim, self.active_dim))
        self.W = W

    def set_l(self, l):
        assert(l.shape == (self.active_dim,))
        self.l = l
        self.lengthscale = Param("lengthscale", self.l, Logexp())

    def set_s(self, s):
        assert(isinstance(s, float))
        self.s = s
        self.variance = Param("variance", self.s, Logexp())

    ###############################
    #      SAMPLING FUNCTIONS     #
    ###############################
    def sample_W(self):
        """
        :return: An orthogonal matrix
        """
        A = np.zeros((self.real_dim, self.active_dim), dtype=np.float64)
        for i in range(self.real_dim):
            for j in range(self.active_dim):
                A[i, j] = np.random.normal(0, 1)
        Q, R = np.linalg.qr(A)
        assert (np.allclose(np.dot(Q.T, Q), np.eye(Q.shape[1])))
        return Q

    def sample_variance(self):
        """
        :return: A standard variance
        """
        return 0.1

    def sample_lengthscale(self):
        """
        :return: A standard length-scale
        """
        return np.ones((self.active_dim,)) * 0.2

    ###############################
    #        KERNEL-FUNCTIONS     #
    ###############################
    @Cache_this(limit=5, ignore_args=())
    def K(self, X1, X2):
        """
        :param X1: A vector (or is a matrix allowed?)
        :param X2:
        :return:
        """
        print("I am being called!")
        print("Input is: ", X1)
#        assert X1.shape[1] == self.real_dim, (X1.shape, self.real_dim)
#        assert X2.shape[1] == self.real_dim

        Z1 = np.dot(X1, self.W)
        Z2 = np.dot(X2, self.W)
        return super(TripathyMaternKernel, self).K(Z1, Z2)
        # TODO: VERY IMPORTANT! does this calculate the gram-matrix (i.e. does this work also for matrix-inputs

    def Kdiag(self, X):
        """
        :param X:
        :return:
        """
        assert(X.shape[1] == self.real_dim)

        Z = np.dot(X, self.W)
        return super(TripathyMaternKernel, self).Kdiag(Z)

    # Oriented by
    # http: // gpy.readthedocs.io / en / deploy / _modules / GPy / kern / src / stationary.html  # Matern32.dK_dr
    def K_of_r(self, r):
        """
        :param r: The squared distance that one is to input
        :return:
        """
        return self.variance * (1. + np.sqrt(3.) * r) * np.exp(-np.sqrt(3.) * r)

    ###############################
    #          DERIVATIVES        #
    ###############################
    def dK_dr(self, r):
        """
        :param r: The
        :return:
        """
        return -3. * self.variance * r * np.exp(-np.sqrt(3.) * r)

    # TODO: we could use the function _squared_scaled_distance instead, which is inherited
    def r(self, x, y):
        """
        The squared distance function
        :param x:
        :param y:
        :return:
        """
        x_prime = np.dot(x, self.W)
        y_prime = np.dot(y, self.W)

        out = np.dot((x_prime-y_prime).T, (x_prime-y_prime))
        out = np.divide(out, np.power(self.l, 2))
        return out

    def dr_da(self, x, y):
        """
        The differential of the squared scaled distance w.r.t. the first input argument
        :param x:
        :param y:
        :return:
        """
        return 2 * np.divide(x-y, np.power(self.l, 2))

    def dr_db(self, x, y):
        """
        The differential of the squared scaled distance w.r.t. the second input argument
        :param x:
        :param y:
        :return:
        """
        return - 2 * np.divide(x-y, np.power(self.l, 2))

    def check(self, X1, X2):
        print("Checking..")
        print(X1.shape)
        print(X2.shape)


if __name__ == "__main__":

    real_dim = 4
    active_dim = 3
    no_samples = 5

    sample = TripathyMaternKernel(real_dim, active_dim)
    # print(sample.to_dict())
    # print(sample.W)
    # print(sample.l)
    # print(sample.s)
    #
    # sample.set_l(np.random.rand(2,))
    # sample.set_s(5.22)
    # sample.set_W(np.random.rand(3, 2))
    #
    # print(sample.W)
    # print(sample.l)
    # print(sample.s)

    # The second dimension must correspond with the 'real dimension'
    A = np.random.rand(no_samples, real_dim)
    B = np.random.rand(no_samples, real_dim)
    print("Input is: ", A)
    Z1 = np.dot(A, sample.W)
    Z2 = np.dot(B, sample.W)
    print(Z1)
    sample.check(A, B)
    print(sample.K(A, B))
#    print(sample.K_of_r(np.random.rand(5)))
#    print(sample.Kdiag(X1))

