import sys
sys.path.append("/Users/davidal/GoogleDrive/BachelorThesis/code/tripathy")
print(sys.path)
import numpy as np
from src.t_kernel import TripathyMaternKernel
from GPy.kern.src.stationary import Matern32

class TestKernel(object):

    def init(self):
        self.real_dim = 3
        self.active_dim = 2
        self.no_samples = 5
        self.kernel = TripathyMaternKernel(self.real_dim, self.active_dim)

    def test_parameters_are_set_successfully(self):
        """
        Check if parameters are set successfully / setters work correctly
        :return:
        """
        self.init()

        W1, l1, s1 = self.kernel.W, self.kernel.inner_kernel.lengthscale, self.kernel.inner_kernel.variance
        W1 = W1.copy()
        l1 = l1.copy()
        s1 = s1.copy()

        new_W = np.zeros((self.real_dim, self.active_dim), dtype=np.float64)
        for i in range(self.real_dim):
            for j in range(self.active_dim):
                new_W[i, j] = np.random.normal(0, 1)
        Q, R = np.linalg.qr(new_W)

        # Set new parameters
        self.kernel.update_params(
            W=Q,
            l=np.random.rand(self.active_dim,),
            s=5.22
        )

        assert not np.isclose(np.asarray(self.kernel.inner_kernel.lengthscale), np.asarray(l1)).all()
        assert not np.isclose(np.asarray(self.kernel.inner_kernel.variance), np.asarray(s1))
        assert not np.isclose(self.kernel.W, W1).all()

    def test_kernel_returns_gram_matrix_correct_shape(self):
        """
        Check
        :return:
        """
        self.init()

        A = np.random.rand(self.no_samples, self.real_dim)
        B = np.random.rand(self.no_samples, self.real_dim)

        # print("Before we go into the function: ")
        # print(A)
        # print(B)

        Cov = self.kernel.K(A, B)

        assert Cov.shape == (self.no_samples, self.no_samples)

    def test_kernel_returns_diag_correct_shape(self):
        self.init()

        A = np.random.rand(self.no_samples, self.real_dim)

        # print("Before we go into the function Kdiag: ")
        # print(A)

        Kdiag = self.kernel.Kdiag(A)

        assert Kdiag.shape == (self.no_samples,), (Kdiag.shape,)

    def test_kernel_K_of_r_words_for_vectors(self):
        self.init()

        x = np.random.rand(self.no_samples)

        # print("Before we go into the function Kdiag: ")
        # print(x)

        kr = self.kernel.K_of_r(x)

        assert kr.shape == (self.no_samples,), (kr.shape,)

class TestKernelSematics(object):

    # To calculate each element by hand, type in the following into wolfram alpha
    # f(x)=(1+sqrt(3)*x)exp(−sqrt(3)*x)
    # And calculate each r individually
    #

    def init(self):
        self.real_dim = 3
        self.active_dim = 2
        self.no_samples = 5
        self.kernel = TripathyMaternKernel(self.real_dim, self.active_dim)

    def test_kernel_identity_W_zero_inp(self):
        self.init()
        X = np.asarray([
            [0, 0, 0],
            [0, 0, 0]
        ])

        W = np.asarray([
            [1, 0],
            [0, 1],
            [0, 0]
        ])

        self.kernel.update_params(
            W=W,
            l=np.asarray([1. for i in range(self.active_dim)]),
            s=1.
        )

        self.real_kernel = Matern32(self.active_dim, ARD=True, lengthscale=self.kernel.inner_kernel.lengthscale)

        y_hat = self.kernel.K(X)

        y = np.asarray([
            [1, 1],
            [1, 1]
        ])

        assert np.isclose(y, y_hat, rtol=1e-16).all()

    def test_kernel_reverted_W(self):
        self.init()
        X = np.asarray([
            [0, 0.5, 0],
            [0, 0, 0.5]
        ])

        W = np.asarray([
            [0, 1],
            [0, 0],
            [1, 0]
        ])

        # After projection, we get
        # X_new = [
        #     [0, 0],
        #     [0.5, 0]
        # ]
        # r = 0.25

        self.kernel.update_params(
            W=W,
            l=np.asarray([1. for i in range(self.active_dim)]),
            s=1.
        )

        y_hat = self.kernel.K(X)

        y = np.asarray([
            [1, 0.784888],
            [0.784888, 1]
        ])

        assert np.isclose(y, y_hat, rtol=1e-4).all()

    def test_kernel_some_random_W(self):
        self.init()

        for i in range(100):
            X = np.random.rand(5, self.real_dim)

            # Sample and re-assign
            # TODO: just let the kernel resample all parameters
            W = self.kernel.sample_W()
            s = self.kernel.sample_variance()
            l = self.kernel.sample_lengthscale()

            self.kernel.update_params(W=W, l=l, s=s)

            y_hat = self.kernel.K(X)

            y = self.kernel.inner_kernel.K(np.dot(X, W))

            assert np.isclose(y, y_hat).all()

    def test_kernel_some_random_W_independent_inner_kernel(self):
        self.init()

        for i in range(100):
            X = np.random.rand(5, self.real_dim)

            # Sample and re-assign
            # TODO: change this by just resampling using a function within the kernel
            W = self.kernel.sample_W()
            s = self.kernel.sample_variance()
            l = self.kernel.sample_lengthscale()

            self.kernel.update_params(W=W, l=l, s=s)

            y_hat = self.kernel.K(X)

            # Define the new kernel
            real_kernel = Matern32(self.active_dim, variance=s, ARD=True, lengthscale=l)

            y = real_kernel.K(np.dot(X, W))

            assert np.isclose(y, y_hat).all()
