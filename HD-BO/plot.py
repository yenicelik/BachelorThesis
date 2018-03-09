import matplotlib.pyplot as plt 
import ggplot
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pandas.tools.plotting import scatter_matrix

class Plotter:

	def __init__(self):
		pass

	def visualize_acq_1D(self, Xpred, acq_fnc, gp, beta):
		plt.plot(Xpred, acq_fnc(Xpred, gp, beta))

		plt.title('Acquisition Function.')
		plt.show()

	def visualize_1D(self, Xtrain, ytrain, Xpred, BFF_fnc, mean_vec, stddev_vec):
		plt.plot(Xtrain, ytrain, 'bs', ms=8)
		plt.plot(Xpred, mean_vec)
		plt.plot(Xpred, BFF_fnc(Xpred), 'r:', label=u'$f(x) = x\,\sin(x)$')

		plt.gca().fill_between(Xpred.reshape(-1,), mean_vec-2*stddev_vec, mean_vec+2*stddev_vec, color="#dddddd")

		plt.title('Samples from the GP posterior.')
		plt.show()

	def visualize_2D(self, Xtrain, ytrain, Xpred, BFF_func, mean_vec, stddev_vec):
		"""
			This function plots f for each individual dimension
			f :: (high dimensional) function
			X :: R^nxd (n points, each with d dimensions/features)
		"""

		pass



		# assert( X.shape[0] > 0 )
		# assert( X.shape[1] > 0 )

		# n, d = X.shape
		# y = np.empty(n)

		# # Calculate function for each datapoint
		# for i in range(n):
		# 	y[i] = fnc(X[i,:])

		# # Visualize function for each individual dimension
		# hgrids = ceil(sqrt(d))

		# plt.figure(1)
		# for j in range(d):
		# 	plt.subplot(hgrids * 100 + hgrids * 10 + i)
		# 	plt.plot(X[:,j], y)
		# 	plt.grid(True)

		# plt.show()










