import sys
sys.path.append("/Users/davidal/GoogleDrive/BachelorThesis/code/tripathy")
print(sys.path)
import numpy as np
from src.t_kernel import TripathyMaternKernel

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

        W1, l1, s1 = self.kernel.W, self.kernel.l, self.kernel.s

        # Set new parameters
        self.kernel.set_l(np.random.rand(self.active_dim,))
        self.kernel.set_s(5.22)
        self.kernel.set_W(np.random.rand(self.real_dim, self.active_dim))

        assert not np.isclose(self.kernel.l, l1).all()
        assert not self.kernel.s == s1
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

    def test_set_parameters_works(self):
        # Test if set_W, set_l, set_s, (set_sn) work, and are applied on the correct kernel)
        pass

