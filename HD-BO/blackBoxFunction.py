import math
import numpy as np
np.random.seed(seed=42)

class BlackBoxFunction:

	def __init__(self, effective_dim=None, real_dim=None):
		print("Importing BlackBoxFunction Module")

		self.d = effective_dim
		self.D = real_dim

		if effective_dim is not None and real_dim is not None: 

			# Simple sine function
			def g(x): return np.sum( np.sin( (x - 1) * 5.5) + 0.5 )
			self.g = g

			# # Create mean and cov (p.s.d matrix) sigma
			sigma_sqr = np.random.rand(self.d, self.d) * 5 # TODO: What should the dimensions be?
			cov = sigma_sqr.T * sigma_sqr
			mu = np.random.rand(self.d) * 5

			def h(x):
				x_hat = x - mu
				out = np.linalg.solve(cov, x_hat)
				out = np.dot(x_hat, out)
				out = 3 * np.exp( -0.5 * out)
				return out			
			self.h = h

			def f(x): return self.g(x) * self.h(x) - 1
			self.f = f



	def eval(self, x):
		return 3 * np.sin(x) + 2

	def eval_multi(self, x):
		# Includes a 'scalarizer' such that enough dimensions are captures
		out = []
		for i in range(x.shape[0]):
			out.append(np.linalg.norm(x[0,:]))

		return np.asarray(out)

	def eval_hd(self, x):
		return self.f(x)

	





