import sys
sys.path.append("/Users/davidal/GoogleDrive/BachelorThesis/code/tripathy")
print(sys.path)
import numpy as np
from src.t_kernel import TripathyMaternKernel
from src.t_loss import loss, dloss_dK, dloss_dW, dK_dW

def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
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

class TestLoss(object):

    def init(self):
        self.real_dim = 3
        self.active_dim = 2
        self.no_samples = 5
        self.kernel = TripathyMaternKernel(self.real_dim, self.active_dim)

        self.W = np.random.rand(self.real_dim, self.active_dim)
        self.sn = np.random.rand(1)
        self.s = np.random.rand(1)
        self.l = np.random.rand(self.active_dim,)

        self.X = np.random.rand(self.no_samples, self.real_dim)
        self.Y = np.random.rand(self.np_samples,)

    def test_loss_returns_correct_dimensions(self):
        self.init()

        loss(self.kernel, self.W, self.sn, self.s, self.l, self.X, self.Y)





    def test_dK_dW(self):

        self.init()

        x = np.random.rand(self.real_dim)
        y = np.random.rand(self.real_dim)

        print("Input in dK_dW is: ")
        print(x)
        print(y)

