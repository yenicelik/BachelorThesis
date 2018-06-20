import numpy as np

from bacode.tripathy.src.rembo import rembo_algorithm # import normalize, denormalize
from febo.environment.domain import ContinuousDomain
from bacode.tripathy.src.rembo.utils import sample_orthogonal_matrix

class TestNormalizeDenormalize(object):

    def init(self):
        self.dim = 2
        self.domain_lowerbounds = np.asarray([-5., -2.])
        self.domain_upperbounds = np.asarray([3., 1.])

        assert self.domain_lowerbounds.shape[0] == self.dim
        assert self.domain_upperbounds.shape[0] == self.dim

        self.center = (self.domain_upperbounds + self.domain_lowerbounds) / 2.
        self.range = self.domain_upperbounds - self.domain_lowerbounds

    # def test_simple_example1(self):
    #     X = np.asarray([
    #         [-4., 1.]
    #     ])
    #
    #     denorm_X = np.asarray([
    #         [-4./-5.]
    #     ])
    #
    #     X_new = rembo_algorithm.normalize(X, self.center, self.range)
    #     assert np.isclose(X_new, np.asarray([]) )
    #
    #
    # def test_simple_example2(self):
    #     X = np.asarray([
    #         [-1., -2.]
    #     ])
    #
    #     X_new = rembo_algorithm.normalize(X, self.center, self.range)


    def test_norm_denorm_same(self):
        self.init()

        for i in range(20):

            self.X = (np.random.random((self.dim,)) - 0.5) * 2.

            assert (self.X <= 1.0).all()
            assert (self.X >= -1.0).all()

            domain = ContinuousDomain(np.asarray([-5] * self.dim), np.asarray([2] * self.dim))

            X_norm = rembo_algorithm.normalize(self.X, domain)
            X_denorm = rembo_algorithm.denormalize(X_norm, domain)

            assert np.isclose(self.X, X_denorm).all(), ("Not quite the same values after norm+denorm! ", (self.X, X_denorm))

    def test_sampled_matrix_is_orthonormal(self):

        epochs = 5 # how often to test out that normalization works

        active_dims = [2**i for i in range(8)]
        real_dims = [2**i for i in range(8)]

        for e in range(epochs):
            for real_dim in real_dims:
                for active_dim in active_dims:
                    if active_dim > real_dim:
                        continue
                    else:
                        Q = sample_orthogonal_matrix(real_dim, active_dim)
                        assert np.isclose( np.dot(Q.T, Q), np.eye(Q.shape[1]) ).all(), ("Sampled matrices are not orthonormal!", real_dim, active_dim)








