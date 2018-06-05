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

        self.center = (self.domain.u + self.domain.l) / 2.

        if USE_REAL_MATRIX:
            self.A = np.asarray([
                [1, 0],
                [0, 1],
                [0, 0]
            ])
        else:
            self.A = sample_orthogonal_matrix(self.domain.d, self.config.dim)

        self.optimization_domain = get_subspace(self.config.dim)

        self.optimizer = ScipyOptimizer(self.optimization_domain)
        self.gp = GP(self.optimization_domain)

    def _next(self):
        z_ucb, _ = self.optimizer.optimize(self.ucb_acq_function)
        out = self.project_low_to_high(z_ucb)
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
