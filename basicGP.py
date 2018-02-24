
class BasicGP:

	def __init__(self, kernel_param):
	    self.kernel_param = kernel_param

	    # These need to be set!
	    self.X = np.empty((0, 0))
	    self.K_star_star = np.empty((0, 0))
	    self.K_star = np.empty((0, 0))
	    self.K = np.empty((0, 0))

	def kernel(self, x1, x2):
		"""
			Squared exponential kernel
		"""
		sqdist = np.sum(x1**2,1).reshape(-1,1) + np.sum(x2**2,1) - 2*np.dot(x1, x2.T)
		return np.exp(-.5 * (1/param) * sqdist)

	def set_datapoints(self, X):
		self.X = X
		self.K = self.kernel(X)

	def predict(self, x_star):
		self.K_star_star = self.kernel(x_star, x_star)
		self.K_star = self.kernel(x_star, X)


