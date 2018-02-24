import matplotlib.pyplot as plt 
import ggplot
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from pandas.tools.plotting import scatter_matrix

class Plotter:

	def __init__():
		pass

	def visualize_function(fnc, X):
		"""
			This function plots f for each individual dimension
			f :: (high dimensional) function
			X :: R^nxd (n points, each with d dimensions/features)
		"""
		assert( X.shape[0] > 0 )
		assert( X.shape[1] > 0 )

		n, d = X.shape
		y = np.empty(n)

		# Calculate function for each datapoint
		for i in range(n):
			y[i] = fnc(X[i,:])

		# Visualize function for each individual dimension
		hgrids = ceil(sqrt(d))

		plt.figure(1)
		for j in range(d):
			plt.subplot(hgrids * 100 + hgrids * 10 + i)
			plt.plot(X[:,j], y)
			plt.grid(True)

		plt.show()










