from pprint import pprint

from bacode.tripathy.src.rembo.utils import sample_orthogonal_matrix

import numpy as np
from febo.algorithms import Algorithm, AlgorithmConfig
from febo.environment import ContinuousDomain
from febo.models import GP
from febo.optimizers import ScipyOptimizer
from febo.utils.config import ConfigField, config_manager, assign_config


class RemboConfig(AlgorithmConfig):
    dim = ConfigField(2, comment='subspace dimension')


config_manager.register(RemboConfig)


def normalize(x, domain):
    """
        Normalize value of x from the range of the domain, to [-1, 1]^d
    :param x:
    :param center:
    :param range:
    :return:
    """
    # assert x.shape == center.shape, ("Center and x don't have the same shape ", x.shape, center.shape)
    # assert domainrange.shape == center.shape, ("Center and range don't have the same shape ", center.shape, domainrange.shape)
    return (domain.normalize(x) - 0.5) * 2.
    # return np.divide(x - center, domainrange)


def denormalize(x, domain):
    """
        Normalize value of x from the range of the domain, to [-1, 1]^d
    :param x:
    :param center:
    :param range:
    :return:
    """
    # assert x.shape == center.shape, ("Center and x don't have the same shape ", x.shape, center.shape)
    # assert domainrange.shape == center.shape, ("Center and range don't have the same shape ", center.shape, domainrange.shape)
    return domain.denormalize( (x / 2.) + 0.5)
    # return np.multiply(x, domainrange) + center


def get_subspace(effective_dimensions):
    """
        Calculates the effective search domain Y as given in the REMBO paper
        $$
         Y = [ −1/eps* max{log(de), 1}, 1/eps *  max{log(de), 1} ] ^ de
        $$
    :param effective_dimensions:
    :return:
    """
    eps = np.log(effective_dimensions) / np.sqrt(effective_dimensions) * (2.) # This modifies the chance that we get a bad entry!

    span_high = np.ones((effective_dimensions,))
    span_high = np.log(span_high)
    span_high = np.maximum(span_high, 1) / eps

    return ContinuousDomain(-1 * span_high, span_high)


@assign_config(RemboConfig)
class RemboAlgorithm(Algorithm):

    def ucb_acq_function(self, Z):
        return -self.gp.ucb(Z)

    def add_data(self, data):
        # Project the data to the low-dimensional subspace! # TODO: do we normalize here?
        x = data['x']
        x = self.project_high_to_low(x)
        self.gp.add_data(x, data['y'])
        self.gp.optimize()  # TODO: How do we optimize the kernel parameters?

    def initialize(self, **kwargs):
        """
            self.domain carries the higher-dimensional domain
            self.config.dim carries the lower-dimensional domain
        :param kwargs:
        :return:
        """
        super(RemboAlgorithm, self).initialize(**kwargs)

        # Take the domain out of the kwargs
        self.domain = kwargs.get('domain') # TODO: this was added as a result of visualizing REMBO (because usually, it is a subclass of the model)
        print("Domain is: ", self.domain)
        self.center = (self.domain.u + self.domain.l) / 2.

        if USE_REAL_MATRIX:
            self.A = np.asarray([
                [1, 0],
                [0, 1],
                [0, 0]
            ])
        else:
            self.A = sample_orthogonal_matrix(self.domain.d, self.config.dim, seed=None)
            self.A = np.asarray(
                [[-0.87029532, - 0.1387281],
                 [-0.03966056, - 0.44315976],
                 [0.10547953, - 0.8431906],
                 [-0.42296969,  0.2088443],
                [-0.22579596, - 0.17256191]]
            )
            # self.A = np.asarray(
            #     [[-0.48816741, -0.60447259],
            #      [0.2216277, -0.78353003],
            #      [0.84414083, -0.14385261]]
            # )
            print("REMBO uses the following matrix!")
            print(self.A)

            # A little too straight, but it seems acceptable!
            # [[-0.64909311  0.4687148]
            #  [-0.21805274 - 0.86921481]
            #  [0.72878744  0.1573914]]

            # Looks good enough
            # [[-0.48816741 - 0.60447259]
            #  [0.2216277 - 0.78353003]
            #  [0.84414083 - 0.14385261]]

        self.optimization_domain = get_subspace(self.config.dim)

        self.optimizer = ScipyOptimizer(self.optimization_domain)
        self.gp = GP(self.optimization_domain)

    def _next(self):
        z_ucb, _ = self.optimizer.optimize(self.ucb_acq_function)
        print("Optimized point is: ", z_ucb)
        out = self.project_low_to_high(z_ucb)
        print("Next chosen point is: ", out)
        return out

    def project_high_to_low(self, x):
        """
        self.domain.dim to self.config.dim
        :param x:
        :return:
        """
        out = x

        if DEBUG_LOW:
            print("Before projection to low dimensional space to low-dimensional space", out)

        if NORM_DENORM:
            out = normalize(out, self.domain)

        if DEBUG_LOW:
            print("Normlizing to low-dimensional space", out)

        out = np.dot(out, self.A)

        if DEBUG_LOW:
            print("After Projecting to low-dimensional space", out)

        out = np.maximum(out, self.optimization_domain.l)
        out = np.minimum(out, self.optimization_domain.u)

        if DEBUG_LOW:
            print("After bounding to low-dimensional space", out)

        return out

    def project_low_to_high(self, x):
        """
                self.config.dim to self.domain.dim
        :param x:
        :return:
        """
        if DEBUG_HIGH:
            print("\n\n\n\n\n\nProjecting to high-dimensional space", x)
        out = np.dot(x, self.A.T)

        if DEBUG_HIGH:
            print("After Projecting to high-dimensional space", out)

        if NORM_DENORM:
            out = np.maximum(out, -1 * np.ones((self.domain.d,)))  # before was self.domain.l, and self.domain.u
            out = np.minimum(out, 1 * np.ones((self.domain.d,)))
        else:
            out = np.maximum(out, self.domain.l)
            out = np.minimum(out, self.domain.u)

        if DEBUG_HIGH:
            print("Bounding to high-dimensional space", out)

        if NORM_DENORM:
            out = denormalize(out, self.domain)

        if DEBUG_HIGH:
            print("Denormalising to high-dimensional space", out)

        return out

USE_REAL_MATRIX = False
NORM_DENORM = True
DEBUG_HIGH = False
DEBUG_LOW = False

#         assert self.A.shape == (self.domain.d, self.config.dim), (
#             "Something went wrong when generating the dimensions of A! ", self.A.shape,
#             (self.config.dim, self.config.dim))
#
#         # TODO: do we need to normalize these aswell?
#         self.hd_lowerbound = -1 * np.ones(self.domain.d)
#         self.hd_upperbound = 1 * np.ones(self.domain.d)
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
#         assert out.shape[1] == self.domain.d, (
#             "Output of next does not conform with environment dimensions: ", out.shape, self.domain.d)
#         return out
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
#         # print("After inverse projection (high to low)! ", out)
#
#         return out
