#!/bin/python3

import math

import pandas as pd, numpy as np

from scipy import stats
from scipy.special import factorial
from matplotlib import pyplot as plt

def fb(v, deterministic=True):
	"""
	Function to plot the signal
	"""
	if deterministic:
		plt.fill_between(range(v.shape[1]), np.mean(v,0)-np.std(v,0), np.mean(v,0)+np.std(v,0), color='g')
		plt.yscale('log')
		plt.show()
	else:
		for i in range(v.shape[0]):
			plt.scatter(range(v.shape[1]), v[i,:], alpha=0.5,c='g')
			plt.ylim(1e-2,1000)				
		plt.yscale('log')
		plt.show()




def main():
	# Open the unemployment rate data
	df = pd.read_csv('data/UNRATE.csv')
	x = df.iloc[:,0].tolist()
	y = df.iloc[:,1].tolist()

	# Extract lists of yearly values
	x = np.asarray([x[i*12:(i+1)*12] for i in range(len(x)//12-1)])
	y = np.asarray([y[i*12:(i+1)*12] for i in range(len(y)//12-1)])

	# Compute the FFT along with its radius and theta
	z = np.asarray([np.fft.fft(y[i,:]) for i in range(len(y))])
	zradius = np.asarray([np.abs(z_) for z_ in z])
	ztheta = np.asarray([[np.arctan(z__.imag / z__.real) if z__.real !=0 else np.pi/2 for z__ in z_] for z_ in z])


	# Show the FFT coefficients of the yearly unemployment signal's absolute values
	fb(zradius, deterministic=False)

	# Model zradius as a multidimensional gaussian with gaussian prior

	# Compute posterior

	# Antitransform

	# Show the expected value, which should be the expected value of the bayesian model's fourier conjugate right?		
	



if __name__=='__main__':
	main()
