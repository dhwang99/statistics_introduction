#encoding: utf8

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

'''
已知样本来自于以下混合高斯分布

F = 0.2 * N(1, 2**2) + 0.8 * N(6, 1)

估计方差、偏差. 这样就估计出mse了

X ~ F
X_bar ~ 4.997
'''

def mix_norm_model():

