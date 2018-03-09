import numpy as np
from scipy.optimize import minimize

class BasicGP:

	def __init__(self, kernel_param):
	    self.kernel_param = kernel_param

	    # These need to be set!
	    self.X = np.empty((0, 0))
	    self.y = np.empty((0))
	    self.K_star_star = np.empty((0, 0))
	    self.K_star = np.empty((0, 0))
	    self.K = np.empty((0, 0))

	    self.L = np.empty((0, 0))
	    self.Lk = np.empty((0, 0))

	def kernel(self, x1, x2):
		"""
			Squared exponential kernel
		"""
		sqdist = np.sum(x1**2, 1).reshape(-1,1) + np.sum(x2**2, 1) - 2*np.dot(x1, x2.T)
		return np.exp(-.5 * (1/self.kernel_param) * sqdist)

	def set_datapoints(self, X, y):
		self.X = X
		self.y = y

	def set_predictant(self, x_star):
		self.x_star = x_star

	def calculate_kernels(self):
		self.K = self.kernel(self.X, self.X)
		self.K_star = self.kernel(self.x_star, self.X)
		self.K_star_star = self.kernel(self.x_star, self.x_star)

	def predict_mean(self):
		variance = 5e-5
		lhs = self.K + variance * np.eye(self.X.shape[0])
		solved = np.linalg.solve(lhs, self.y)
		return np.dot(self.K_star, solved).reshape((-1,))

	def predict_stddev(self):
		variance = 5e-5
		lhs = self.K + variance * np.eye(self.X.shape[0])
		rhs = self.kernel(self.X, self.x_star)
		solved = np.linalg.solve(lhs, rhs)
		right_summand = self.kernel(self.x_star, self.X)
		right_summand = np.dot(right_summand, solved)
		res = self.kernel(self.x_star, self.x_star) - right_summand 
		return np.diagonal(np.sqrt(res)).reshape((-1,))

	def predict_single(self, x):
		#TODO: This is a duplicate of the above lines!
		self.set_predictant(x) # Where x can be a scalar!
		return self.predict_mean(), self.predict_stddev() 



