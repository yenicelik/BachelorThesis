import math
import numpy as np

class BlackBoxFunction:

	def __init__():
		print("Importing BlackBoxFunction Module")

	def eval(x):
		return 3 * np.sin(x) + 2

	def eval_multi(x):
		# Includes a 'scalarizer' such that enough dimensions are captures
		out = []
		for i in range(x.shape[0]):
			out.append(np.linalg.norm(x[0,:]))

		return np.asarray(out)

	





