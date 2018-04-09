import sys

from febo.environment.benchmarks import Rosenbrock

sys.path.append("/Users/davidal/GoogleDrive/BachelorThesis/code/tripathy")
print(sys.path)

import numpy as np
from src.t_kernel import TripathyMaternKernel
from src.t_loss import loss, dloss_dK, dloss_dW, dK_dW

# def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
#   """
#   a naive implementation of numerical gradient of f at x
#   - f should be a function that takes a single argument
#   - x is the point (numpy array) to evaluate the gradient at
#   """
#
#   fx = f(x) # evaluate function value at original point
#   grad = np.zeros_like(x)
#   # iterate over all indexes in x
#   it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
#   while not it.finished:
#
#     # evaluate function at x+h
#     ix = it.multi_index
#     oldval = x[ix]
#     x[ix] = oldval + h # increment by h
#     fxph = f(x) # evalute f(x + h)
#     x[ix] = oldval - h
#     fxmh = f(x) # evaluate f(x - h)
#     x[ix] = oldval # restore
#
#     # compute the partial derivative with centered formula
#     grad[ix] = (fxph - fxmh) / (2 * h) # the slope
#     if verbose:
#       print(ix, grad[ix])
#     it.iternext() # step to next dimension
#
#   return grad

class TestLoss(object):

    def init(self):
        self.real_dim = 3
        self.active_dim = 2
        self.no_samples = 5
        self.kernel = TripathyMaternKernel(self.real_dim, self.active_dim)

        self.sn = np.asscalar( np.random.rand(1) )
        self.s = np.asscalar( np.random.rand(1) )
        self.l = np.random.rand(self.active_dim,)
        self.W = self.kernel.sample_W()

        self.function = Rosenbrock()
        self.real_W = np.asarray([
            [0, 0],
            [0, 1],
            [1, 0]
        ])
        self.X = np.random.rand(self.no_samples, self.real_dim)
        Z = np.dot(self.X, self.real_W)
        self.Y = self.function._f(Z.T)

    def test_loss_returns_correct_dimensions(self):
        self.init()

        res = loss(self.kernel, self.W, self.sn, self.s, self.l, self.X, self.Y)

        # Should return a scalar, and run (not exit due to some false dimensions!
        assert isinstance(res, float), str(res)

    def test_loss_correctly_updates_parameters(self):
        self.init()

        loss(self.kernel, self.W, self.sn, self.s, self.l, self.X, self.Y)

        assert (self.kernel.W == self.W).all()
        assert self.kernel.inner_kernel.variance == self.s
        assert (self.kernel.inner_kernel.lengthscale == self.l).all()

    # def init_calculation_by_hand(self):
    #     self.W = np.asarray([
    #         [],
    #         [],
    #         []
    #     ])
    #
    # def test_loss_returns_correct_value(self):
    #     self.init()
    #
    #     loss(self.kernel, self.W, self.sn, self.s, self.l, self.X, self.Y)

class TestLossSemantics(object):
    """
        If you want to write this to wolfram alpha:
        0.5 * (t) inv(K + sn^2 I) (t)
        - 0.5 * log(K + sn^2 I)
        - 0.5 * N * log2π
    """

    def init(self):
        self.real_dim = 3
        self.active_dim = 2
        self.no_samples = 5
        self.kernel = TripathyMaternKernel(self.real_dim, self.active_dim)

    def test_return_2_samples_zero_output(self):

        self.init()

        X = np.asarray([
            [0, 0.5, 0],
            [0, 0, 0.5]
        ])
        Y = np.asarray([
            [0],
            [0]
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

        # Confirmed from another test that this works!
        K = self.kernel.K(X)
        # with approximate output of
        # np.asarray([
        #     [1, 0.784888],
        #     [0.784888, 1]
        # ])

        y_hat = loss(self.kernel, W, sn=2., s=1.,
            l=np.asarray([1. for i in range(self.active_dim)]),
            X=X,
            Y=Y
        )
        # y =




class TestDerivatives(object):

    def init(self):
        self.real_dim = 3
        self.active_dim = 2
        self.no_samples = 5
        self.kernel = TripathyMaternKernel(self.real_dim, self.active_dim)

        self.W = self.kernel.sample_W()
        self.sn = np.asscalar( np.random.rand(1) )
        self.s = np.asscalar( np.random.rand(1) )
        self.l = np.random.rand(self.active_dim,)

        self.function = Rosenbrock()
        self.real_W = np.asarray([
            [0, 0],
            [0, 1],
            [1, 0]
        ])
        self.X = np.random.rand(self.no_samples, self.real_dim)
        Z = np.dot(self.X, self.real_W)
        self.Y = self.function._f(Z.T)

    def test_dloss_dK_returns_correct_dimensions(self):
        self.init()

        res = dloss_dK(self.kernel, self.W, self.sn, self.s, self.l, self.X, self.Y)

        # Should return a scalar, and run (not exit due to some false dimensions!
        assert isinstance(res, float), str(res)

    # TODO: figure out this thing!
    def test_dK_dW_returns_correct_dimension(self):
        self.init()

        res = dK_dW(self.kernel, self.W, self.sn, self.s, self.l, self.X)

        assert res.shape == (self.no_samples * self.real_dim, self.no_samples * self.active_dim)

    def test_dloss_dW_returns_correct_dimension(self):
        self.init()

        res = dloss_dW(self.kernel, self.W, self.sn, self.s, self.l, self.X, self.Y)

        assert res.shape == (self.real_dim, self.active_dim)

