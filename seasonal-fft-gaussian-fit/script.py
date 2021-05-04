#!/bin/python3

import pandas as pd, numpy as np

from matplotlib import pyplot as plt
from auxiliary_functions import plotter_fourier, plotter_inverse_fourier,\
				 plotter_statistics, get_mu_and_sigma, plotter_raw_data

DISPLAY_FOURIER = True
DISPLAY_RESULT = True
DISPLAY_STATISTICS = True
DISPLAY_RAW = True
f,ax = plt.subplots(1,4, figsize=(40,15))


def main():
	# Open the unemployment rate data
	df = pd.read_csv('data/UNRATE.csv')
	x = df.iloc[:,0].tolist()
	y = df.iloc[:,1].tolist()

	# Extract lists of yearly values
	x = np.asarray([x[i*12:(i+1)*12] for i in range(len(x)//12-1)])
	y = np.asarray([y[i*12:(i+1)*12]/np.mean(y[i*12:(i+1)*12]) for i in range(len(y)//12-1)])

	# Compute the FFT along with its radius and theta
	z = np.asarray([np.fft.fft(y[i,:]) for i in range(len(y))])
	ztot = z[:,1:] #Ignore the constant component


	# Show the FFT coefficients of the yearly unemployment signal's 
	# absolute values and fit them with a gaussian function
	if DISPLAY_FOURIER:
		plotter_fourier(np.abs(ztot),0,'module', ax)
		ax[0].set_ylim(1e-2, 3e+0)
		ax[0].set_yscale('log')

	# Show the yearly unemployment signal as reconstructed 
	# from the gaussian model fitted in Fourier space
	if DISPLAY_RESULT:
		padded_ztot = np.concatenate([np.zeros((ztot.shape[0],1)), ztot],1)
		plotter_inverse_fourier(padded_ztot,1,'complex', lambda x: np.real(x), ax)
		eps=0.1
		ax[1].set_ylim(1-eps,1+eps)
		ax[1].set_yticks([1-eps,1,1+eps])
		ax[1].set_yticklabels([1-eps,1,1+eps])
		ax[1].set_xticks([0,3,6,9,11])
		ax[1].set_xticklabels(['jan', 'apr', 'jul', 'oct', 'dec'])	


	# Show the yearly unemployment signal as reconstructed 
	# from the gaussian model fitted in Fourier space
	if DISPLAY_STATISTICS:
		plotter_statistics(y, 2, ax)
		eps=0.1
		ax[2].set_ylim(1-eps,1+eps)
		ax[2].set_yticks([1-eps,1,1+eps])
		ax[2].set_yticklabels([1-eps,1,1+eps])
		ax[2].set_xticks([0,3,6,9,11])
		ax[2].set_xticklabels(['jan', 'apr', 'jul', 'oct', 'dec'])	

	if DISPLAY_RAW:
		plotter_raw_data(y,3, ax)

	# Display the figure	
	plt.show()



if __name__=='__main__':
	main()
