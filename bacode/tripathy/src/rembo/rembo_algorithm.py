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


@assign_config(RemboConfig)
class RemboAlgorithm(Algorithm):

    def initialize(self, **kwargs):
        """
            self.domain carries the higher-dimensional domain
            We then create a lower dimensional domain, called "self.optimization_domain"
        :param kwargs:
        :return:
        """
        super(RemboAlgorithm, self).initialize(**kwargs)

        # Sample an orthogonal matrix
        self.A = sample_orthogonal_matrix(self.domain.d, self.config.dim)
        assert self.A.shape == (self.domain.d, self.config.dim), (
            "Something went wrong when generating the dimensions of A! ", self.A.shape,
            (self.domain.d, self.config.dim))

        # Set the respective bounds for the higher and lower dimensions
        self.hd_lowerbound = self.domain.l
        self.hd_upperbound = self.domain.u
        # self.higher_domain = self.domain

        self.center = (self.hd_lowerbound + self.hd_upperbound) / 2.
        self.domain_range = self.hd_upperbound - self.hd_lowerbound
        assert self.center.shape == (self.domain.d,), (
            "The shape of the cente,r and the domain does not conform! ", self.center.shape, self.domain.d)
        assert self.domain_range.shape == (self.domain.d,), (
            "The shape of the domain range, and the domain does not conform! ", self.domain_range.shape, self.domain.d)
        print("self.center is: ", self.center)
        print("self.domain_range is: ", self.domain_range)
        # exit(0)

        # Define the higher and lower dimensions for the auxiliary, lower dimensional subspace
        # In the paper, they define this to be:

        # TODO: check this formula! (how to calculate the lower dimensions!
        # d = self.hd_upperbound.shape[0]
        d = self.config.dim
        self.ld_lowerbound = -1 * np.ones((d,)) * d  # np.sqrt(d) # * np.linalg.norm(self.A) #
        self.ld_upperbound = np.ones((d,)) * d  # np.sqrt(d) # * np.sqrt(d) * np.linalg.norm(self.A)

        self.optimization_domain = ContinuousDomain(self.ld_lowerbound,
                                                    self.ld_upperbound)  # box constrains in self.config.dim dimensions, according to Rembo paper

        self.optimizer = ScipyOptimizer(self.optimization_domain)
        self.gp = GP(self.optimization_domain)

    def _next(self):
        # return rembo's choice
        z_ucb, _ = self.optimizer.optimize(self.ucb_acq_function)
        z_ucb = np.atleast_2d(z_ucb)
        assert z_ucb.shape[1] == self.config.dim, (
            "The output of the optimizer is not the right shape! ", z_ucb.shape, self.config.dim)
        assert z_ucb.shape[0] > 0, ("Somehow, ucb optimizer gave us no points! ", z_ucb.shape)
        out = self.project(z_ucb)
        assert out.shape[1] == self.domain.d, (
            "Output of next does not conform with environment dimensions: ", out.shape, self.domain.d)
        out = out.T.squeeze()
        # Need to denormalize
        # out = self.denormalizer(out)
        return out.T.squeeze()

    def ucb_acq_function(self, Z):
        # TODO: does this need denormalization?
        return -self.gp.ucb(Z)

    def add_data(self, data):
        # x = self.normalizer(data['x'])
        x = self.inv_project(data['x'])
        self.gp.add_data(x, data['y'])

    ############################
    # REMBO SPECIFIC FUNCTIONS #
    ############################

    # Helper functions:
    def normalizer(self, x):
        # TODO: now.. how do we properly use this?
        assert x.shape == (self.domain.d,), ("X is not a single sample - normalizer!", x.shape, self.domain.d)
        print("Before norm: ", x)
        out = np.divide(x - self.center,
                        self.domain_range)  # TODO: should be elementwise division and subtraction!! (rowwise to be exact!)
        print("After norm: ", out)
        return out

    def denormalizer(self, x):
        # TODO: save all these variables internally -- they should actually be present in the domain thingy
        # print("Shape of x is: ", x.shape)
        assert x.shape == (self.domain.d,), ("X is not a single sample - denormalizer!", x.shape, self.domain.d)

        print("Before denorm ", x)
        out = (np.multiply(x, self.domain_range)) + self.center  # TODO: should be elementwise multiplication!
        print("After denorm ", out)
        return out

    def project(self, z):
        """

        Args:
            z: R^self.config.dim

        Returns: x: R^self.hidh_domain.d

        """

        # Assumptions:
        # - z is always contained within Y!
        # -

        # Normalize!
        # print("Before projection! (low to high)", z)
        inp = np.atleast_2d(z)

        assert inp.shape[1] == self.config.dim, (
            "Size of the REMBO input does not conform with input point! ", z.shape, self.config.dim)
        projection_on_hd = np.dot(inp, self.A.T)
        assert projection_on_hd.shape[0] == inp.shape[0], (
            "Somehow, we lost a sample! ", (projection_on_hd.shape, inp.shape))
        assert projection_on_hd.shape[1] == self.domain.d, (
            "Somehow, we lost or gained a dimenion! ", (projection_on_hd.shape, self.domain.d))

        out = np.maximum(projection_on_hd,
                         self.hd_lowerbound)  # TODO: we need to take the samplewise maximum and minimum! This is not the case as of now
        out = np.minimum(out, self.hd_upperbound)

        out = np.atleast_2d(out)

        # Recentralize to the real world!
        # out = (np.multiply(out, self.domain_range)) + self.center

        assert out.shape[1] == self.A.shape[0], (
            "Size of REMBO *projection* does not conform with output point! ", out.shape, self.A.T.shape)
        # print("After projection (low to high)! ", out)
        return out

    def inv_project(self, x):
        """
            This is the projection where we go from the higher dimensional subspace, to the lower dimensional subspace
        :param x:
        :return:
        """
        # x = np.divide(x - self.center, self.domain_range)

        # print("Before inverse projection (high to low)! ", x)
        # Project onto the subspace
        inp = np.atleast_2d(x)
        assert inp.shape[1] == self.A.shape[0], (
            "The inverse projection does not quite match! ", (inp.shape, self.A.shape))
        out = np.dot(x, self.A)
        out = np.atleast_2d(out)

        # self.ld_upperbound.shape[0] # Why does this not work?
        assert self.ld_upperbound.shape[0] == out.shape[1], (
            "The output has a weird format and does not conform with the shape of the domain!",
            self.ld_upperbound.shape,
            out.shape)
        # TODO: check if this conforms with the l2 projection onto the box
        out = np.maximum(self.ld_lowerbound, out)
        out = np.minimum(self.ld_upperbound, out)

        # print("After inverse projection (high to low)! ", out)

        return out
