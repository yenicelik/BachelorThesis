"""
    Create the kernel class here.
    The resulting kernel must inherit from the GPy kernel class
"""
import numpy as np
from GPy.kern.src.kern import Kern
from GPy.kern.src.stationary import Matern32
from GPy.core.parameterization import Param

from GPy.kern.src.stationary import Matern32

class TripathyMaternKernel(Kern):

    """
    A kernel of the following form:
        .. math::
            k(x, x') = k_0(Wx, Wx')
    """


    # TODO: add W as a parameter
    def __init__(self, real_dim, active_dim, W=None, variance=None, lengthscale=None, _name="TripathyMaternKernel32"):

        # self.__dict__['_name'] = _name

        assert(real_dim >= active_dim)

        self.size = real_dim * active_dim + active_dim + 1 + 1

        self.real_dim = real_dim
        self.active_dim = active_dim

        # TODO: add these as priors
        self.W = W if W is not None else self.sample_W()

        self.inner_kernel = Matern32(
            input_dim=self.active_dim,
            variance=self.sample_variance() if variance is None else variance,
            lengthscale=self.sample_lengthscale() if lengthscale is None else lengthscale,
            ARD=True)

        self.update_params(self.W, self.inner_kernel.lengthscale, self.inner_kernel.variance)

        self.W_grad = np.zeros_like(self.W)

        super(TripathyMaternKernel, self).__init__(input_dim=self.real_dim, active_dims=None, name=_name)

        self.link_parameters(self.inner_kernel)
        # TODO: make sure these are referenced copies!
        self.variance = self.inner_kernel.variance
        self.lengtscale = self.inner_kernel.lengthscale

    ###############################
    #       SETTER FUNCTIONS      #
    ###############################
    # TODO: This takes a lot of time!
    # TODO: Update parameters always before loss?
    def update_params(self, W, l, s):
        if not (l is self.inner_kernel.lengthscale):
            self.set_l(l, True)
        if not (s == self.inner_kernel.variance):
            self.set_s(s, True)
        # We will not include W as a parameter, as we want to call the derivatives etc. separatedly
        if not (W is self.W):
            self.set_W(W, True)

    def set_W(self, W, safe=False):
        assert safe
        assert W.shape == (self.real_dim, self.active_dim)
        assert np.allclose( np.dot(W.T, W), np.eye(self.active_dim), atol=1.e-6), (W, np.dot(W.T, W), np.eye(self.active_dim))
        self.W = W

    def set_l(self, l, safe=False):
        assert safe
        assert l.shape == (self.active_dim,)
        l = np.maximum(
            1.e-3,
            l
        )
        self.inner_kernel.lengthscale = l
        # TODO: do we have to link parameters here somehow? (with this kernel, NOT the inner kernel?)


    def set_s(self, s, safe=False):
        assert safe
        assert isinstance(s, float) or isinstance(s, Param), type(s)
        if not (s == float(self.inner_kernel.variance)):
            self.inner_kernel.variance = s

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
        assert np.allclose(np.dot(Q.T, Q), np.eye(Q.shape[1]))
        assert Q.shape[0] == self.real_dim
        assert Q.shape[1] == self.active_dim
        return Q

    def sample_variance(self):
        """
        :return: A standard variance
        """
        return 1.0

    def sample_lengthscale(self):
        """
        :return: A standard length-scale
        """
        return np.ones((self.active_dim,)) * 1.5

    ###############################
    #        KERNEL-FUNCTIONS     #
    ###############################
    def K(self, X1, X2=None):
        """
        :param X1: A vector (or is a matrix allowed?)
        :param X2:
        :return:
        """
        assert X1.shape[1] == self.real_dim, (X1.shape, self.real_dim)
        if X2 is not None:
            assert X2.shape[1] == self.real_dim

        Z1 = np.dot(X1, self.W)
        Z2 = np.dot(X2, self.W) if X2 is not None else None
        return self.inner_kernel.K(Z1, Z2)
        # TODO: VERY IMPORTANT! does this calculate the gram-matrix (i.e. does this work also for matrix-inputs

    def Kdiag(self, X):
        """
        :param X:
        :return:
        """
        assert(X.shape[1] == self.real_dim)

        Z = np.dot(X, self.W)
        return self.inner_kernel.Kdiag(Z)

    def K_of_r(self, r):
        """
        :param r: The squared distance that one is to input
        :return:
        """
        return self.inner_kernel.K_of_r(r)

    ################################
    # INHERITING FROM INNER KERNEL #
    ################################
    def gradients_X(self, dL_dK, X, X2=None):
        Z = np.dot(X, self.W)
        Z2 = np.dot(X2, self.W) if X2 is not None else Z
        tmp = self.inner_kernel.gradients_X(dL_dK, Z, Z2)
        return np.einsum('ik,jk->ij', tmp, self.W)

    def update_gradients_full(self, dL_dK, X, X2):
        assert X2 is None
        Z = np.dot(X, self.W)
        # For all parameters that are also contained in the inner kernel,
        # i.e. lengthscale and variance
        self.inner_kernel.update_gradients_full(dL_dK, Z)

        # Setting the W gradients
        dL_dZ = self.inner_kernel.gradients_X(dL_dK, Z)
        self.W_grad = np.einsum('ij,ik->kj', dL_dZ, X)

    def update_non_incremental(self, X):
        self._K = self.K(X)

