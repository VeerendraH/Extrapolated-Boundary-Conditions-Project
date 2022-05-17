import numpy as np
eps = 0.05
def W1(x):
	return 1 - (np.exp(-x/eps)-1)/(np.exp(-4/eps)-1)
def W2(x):
	return (np.exp(x/eps)-1)/(np.exp(4/eps)-1)