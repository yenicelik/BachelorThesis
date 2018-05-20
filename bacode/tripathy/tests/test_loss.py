import sys

from febo.environment.benchmarks import Rosenbrock, Camelback

sys.path.append("/Users/davidal/GoogleDrive/BachelorThesis/code/tripathy")
print(sys.path)

import numpy as np
from bacode.tripathy.src.t_kernel import TripathyMaternKernel
from bacode.tripathy.src.t_loss import loss, dloss_dW, dK_dW, dloss_ds

def eval_numerical_gradient(f, x, verbose=False, h=1.e-7):
  """
  a naive implementation of numerical gradient of f at x
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """

  fx = f(x) # evaluate function value at original point
  grad = np.zeros_like(x)
  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index
    oldval = x[ix]
    x[ix] = oldval + h # increment by h
    fxph = f(x) # evalute f(x + h)
    x[ix] = oldval - h
    fxmh = f(x) # evaluate f(x - h)
    x[ix] = oldval # restore

    # compute the partial derivative with centered formula
    grad[ix] = (fxph - fxmh) / (2 * h) # the slope
    if verbose:
      print(ix, grad[ix])
    it.iternext() # step to next dimension

  return grad

def eval_numerical_gradient_scalar(f, x, verbose=False, h=1.e-7):
    assert isinstance(x, float)
    fx = f(x)

    high = f(x + h)
    low = f(x - h)
    return (high - low) / (2*h)

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
        self.Y = self.function.f(Z.T)

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
        self.Y = self.function.f(Z.T)

    # TODO: figure out this thing!
    def test_dK_dW_returns_correct_dimension(self):
        self.init()

        res = dK_dW(self.kernel, self.W, self.sn, self.s, self.l, self.X)

        assert res.shape == (self.no_samples * self.real_dim, self.no_samples * self.active_dim)

    def test_dloss_dW_returns_correct_dimension(self):
        self.init()

        res = dloss_dW(self.kernel, self.W, self.sn, self.s, self.l, self.X, self.Y)

        assert res.shape == (self.real_dim, self.active_dim)

class TestDerivativesW(object):

    def init_XY(self):
        self.X = np.random.rand(self.no_samples, self.real_dim)
        Z = np.dot(self.X, self.real_W)
        self.Y = self.function.f(Z.T)

    def init(self):
        self.real_dim = 3
        self.active_dim = 2
        self.no_samples = 5
        self.kernel = TripathyMaternKernel(self.real_dim, self.active_dim)

        self.W = self.kernel.sample_W()
        self.sn = np.asscalar( np.random.rand(1) )
        self.s = np.asscalar( np.random.rand(1) )
        self.l = np.random.rand(self.active_dim,)

        self.function = Camelback()
        self.real_W = np.asarray([
            [0, 0],
            [0, 1],
            [1, 0]
        ])

        self.init_XY()

    def loss_function_W(self, W):
        return loss(self.kernel, W, self.sn, self.s, self.l, self.X, self.Y)

    def loss_function_sn(self, sn):
        return loss(self.kernel, self.W, sn, self.s, self.l, self.X, self.Y)

    def loss_function_s(self, s):
        return loss(self.kernel, self.W, self.sn, s, self.l, self.X, self.Y)

    def test_numerical_evaluator(self):

        for i in range(100):
            X = np.random.rand(1)

            def real_sin(x):
                return np.sin(x)

            def real_sin_deriv(x):
                return np.cos(x)

            # Test on cosine
            X_grad_hat = eval_numerical_gradient(real_sin, X)
            X_grad = real_sin_deriv(X)

            assert np.allclose(X_grad, X_grad_hat)

            # Test on polynomial
            def poly(x):
                return x ** 2

            def poly_deriv(x):
                return 2 * x

            X_grad_hat = eval_numerical_gradient(poly, X)
            X_grad = poly_deriv(X)

            assert np.allclose(X_grad, X_grad_hat), (X_grad, X_grad_hat)


    def test_dloss_dW(self):
        self.init()

        def grad_W(W):
            return dloss_dW(self.kernel, W, self.sn, self.s, self.l, self.X, self.Y)

        # Test the gradient at a few points of W (sample W a few times)
        for i in range(10):
            W = self.kernel.sample_W()
            self.kernel.update_params(W=W, l=self.l, s=self.s)
            grad_numeric = eval_numerical_gradient(self.loss_function_W, W)
            grad_analytical = grad_W(W)

            assert np.allclose(grad_numeric, grad_analytical, atol=1.e-6)

    def test_dloss_dW_changingXY(self):
        """
            Test if the loss predictions still works
            if we change the data X or Y
        :return:
        """
        self.init()

        def grad_W(W):
            return dloss_dW(self.kernel, W, self.sn, self.s, self.l, self.X, self.Y)

        # Test the gradient at a few points of W (sample W a few times)
        for i in range(10):
            W = self.kernel.sample_W()
            self.kernel.update_params(W=W, l=self.l, s=self.s)
            grad_numeric = eval_numerical_gradient(self.loss_function_W, W)
            grad_analytical = grad_W(W)

            assert np.allclose(grad_numeric, grad_analytical, atol=1.e-6)

    def test_dloss_dvariance(self):
        self.init()

        def grad_variance(s):
            return dloss_ds(self.kernel, self.W, self.sn, s, self.l, self.X, self.Y)

        for i in range(10):
            s = np.asscalar( np.random.rand(1) )
            self.kernel.update_params(W=self.W, l=self.l, s=s)
            grad_numeric = eval_numerical_gradient_scalar(self.loss_function_s, s)
            grad_analytical = grad_variance(s)

            assert np.allclose(grad_numeric, grad_analytical, atol=1.e-6)

