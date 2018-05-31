from pprint import pprint

from bacode.tripathy.src.rembo.utils import sample_orthogonal_matrix

# class Rembo:
#
#     def __init__(self):
#         self.A = randommatrix
#         self.remboDomain = ContinuousDomain(u, l)  # bound according to remobo paper
#         self.optimizer = ScipyOptimizer(self.remboDomain)
#
#
#     def next(self):
#         x, _
#         self.optimizer(self.u)  # define acquisition function u somehwere
#         return x
import numpy as np
from febo.algorithms import Algorithm, AlgorithmConfig
from febo.environment import ContinuousDomain
from febo.models import GP
from febo.optimizers import ScipyOptimizer
from febo.utils.config import ConfigField, config_manager, assign_config


class RemboConfig(AlgorithmConfig):
    dim = ConfigField(2, comment='subspace dimension')


config_manager.register(RemboConfig)


def normalize(x, center, domainrange):
    """
        Normalize value of x from the range of the domain, to [-1, 1]^d
    :param x:
    :param center:
    :param range:
    :return:
    """
    assert x.shape == center.shape, ("Center and x don't have the same shape ", x.shape, center.shape)
    assert domainrange.shape == center.shape, (
        "Center and range don't have the same shape ", center.shape, domainrange.shape)
    return np.divide(x - center, domainrange)


def denormalize(x, center, domainrange):
    """
        Normalize value of x from the range of the domain, to [-1, 1]^d
    :param x:
    :param center:
    :param range:
    :return:
    """
    assert x.shape == center.shape, ("Center and x don't have the same shape ", x.shape, center.shape)
    assert domainrange.shape == center.shape, (
        "Center and range don't have the same shape ", center.shape, domainrange.shape)
    return np.multiply(x, domainrange) + center


@assign_config(RemboConfig)
class RemboAlgorithm(Algorithm):

    def ucb_acq_function(self, Z):
        return -self.gp.ucb(Z)  # TODO: Do we need denormalization here anywhere?

    def add_data(self, data):
        # Project the data to the low-dimensional subspace! # TODO: do we normalize here?
        x = data['x']
        # x = normalize(x, center=self.center, domainrange=self.domainrange)
        x = self.project_high_to_low(x)
        self.gp.add_data(x, data['y'])

    def initialize(self, **kwargs):
        """
            self.domain carries the higher-dimensional domain
            self.config.dim carries the lower-dimensional domain
        :param kwargs:
        :return:
        """
        super(RemboAlgorithm, self).initialize(**kwargs)

        self.center = (self.domain.u + self.domain.l) / 2.
        self.domainrange = (self.domain.u - self.domain.l)

        self.A = np.asarray([
            [1, 0],
            [0, 1],
            [0, 0]
        ])
        # sample_orthogonal_matrix(self.domain.d, self.config.dim)

        lowdim_lowerbound = -1 * np.ones((self.config.dim,)) * np.sqrt(self.config.dim)
        lowdim_upperbound = 1 * np.ones((self.config.dim,)) * np.sqrt(self.config.dim)

        self.optimization_domain = ContinuousDomain(l=lowdim_lowerbound, u=lowdim_upperbound)

        self.optimizer = ScipyOptimizer(self.optimization_domain)
        self.gp = GP(self.optimization_domain)

    def _next(self):
        z_ucb, _ = self.optimizer.optimize(self.ucb_acq_function)

        out = self.project_low_to_high(z_ucb)
        # out = denormalize(out, self.center, self.domainrange)

        return out

    def project_high_to_low(self, x):
        """
        self.domain.dim to self.config.dim
        :param x:
        :return:
        """
        out = np.dot(x, self.A) # TODO: normalize right after this maybe?
        out = np.maximum(out, self.optimization_domain.l)
        out = np.minimum(out, self.optimization_domain.u)
        return out

    def project_low_to_high(self, x):
        """
                self.config.dim to self.domain.dim
        :param x:
        :return:
        """
        out = np.dot(x, self.A.T) # TODO: denormalize right after this maybe?
        out = np.maximum(out, self.domain.l)
        out = np.minimum(out, self.domain.u)
        return out

#         assert self.A.shape == (self.domain.d, self.config.dim), (
#             "Something went wrong when generating the dimensions of A! ", self.A.shape,
#             (self.config.dim, self.config.dim))
#
#         # TODO: do we need to normalize these aswell?
#         self.hd_lowerbound = -1 * np.ones(self.domain.d)
#         self.hd_upperbound = 1 * np.ones(self.domain.d)
#
#         self.center = (self.domain.l + self.domain.u) / 2.
#         self.domain_range = self.domain.u - self.domain.l
#
#         assert self.center.shape == (self.domain.d,), (
#             "The shape of the cente,r and the domain does not conform! ", self.center.shape, self.domain.d)
#         assert self.domain_range.shape == (self.domain.d,), (
#             "The shape of the domain range, and the domain does not conform! ", self.domain_range.shape, self.domain.d)
#
#     def _next(self):
#         assert z_ucb.shape[1] == self.config.dim, (
#             "The output of the optimizer is not the right shape! ", z_ucb.shape, self.config.dim)
#         assert z_ucb.shape[0] > 0, ("Somehow, ucb optimizer gave us no points! ", z_ucb.shape)
#
#         # First project, then normalize! # TODO: do we normalize here? when we go from high to low?
#         assert out.shape[1] == self.domain.d, (
#             "Output of next does not conform with environment dimensions: ", out.shape, self.domain.d)
#         out = denormalize(out, self.center, self.domain_range)
#
#         return out
#
#
#     def project_low_to_high(self, z):
#         inp = np.atleast_2d(z)
#         assert inp.shape[1] == self.config.dim, (
#             "Size of the REMBO input does not conform with input point! ", z.shape, self.config.dim)
#
#         assert projection_on_hd.shape[0] == inp.shape[0], (
#             "Somehow, we lost a sample! ", (projection_on_hd.shape, inp.shape))
#         assert projection_on_hd.shape[1] == self.domain.d, (
#             "Somehow, we lost or gained a dimenion! ", (projection_on_hd.shape, self.domain.d))
#
#         # TODO: we need to take the samplewise maximum and minimum! This is not the case as of now
#         out = np.maximum(projection_on_hd, self.hd_lowerbound)
#         out = np.minimum(out, self.hd_upperbound)
#         out = np.atleast_2d(out)
#
#         # Recentralize to the real world!
#         out = (np.multiply(out, self.domain_range)) + self.center
#         assert out.shape[1] == self.A.shape[0], (
#             "Size of REMBO *projection* does not conform with output point! ", out.shape, self.A.T.shape)
#         # print("After projection (low to high)! ", out)
#         return out
#
#     def project_high_to_low(self, x):
#         """
#             This is the projection where we go from the higher dimensional subspace, to the lower dimensional subspace
#         :param x:
#         :return:
#         """
#
#         # print("Before inverse projection (high to low)! ", x)
#         # Project onto the subspace
#         inp = np.atleast_2d(x)
#         assert inp.shape[1] == self.A.shape[0], (
#             "The inverse projection does not quite match! ", (inp.shape, self.A.shape))
#         out = np.dot(x, self.A)
#
#         out = np.atleast_2d(out)
#         assert self.ld_upperbound.shape[0] == out.shape[1], (
#             "The output has a weird format and does not conform with the shape of the domain!",
#             self.ld_upperbound.shape,
#             out.shape)
#
#         # TODO: do we need this when we project to the lower dimensional embedding.. shouldn't, right??
#         # out = np.maximum(self.ld_lowerbound, out)
#         # out = np.minimum(self.ld_upperbound, out)
#
#         # print("After inverse projection (high to low)! ", out)
#
#         return out
