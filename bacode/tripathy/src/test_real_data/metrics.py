"""
    This file includes common metrics that tell us how well our model can predict the real embedding.
"""
import numpy as np

def l2difference(y_pred, y_real):
    diff = np.abs(y_pred - y_real)
    diff = np.square(diff)
    return np.mean(diff)