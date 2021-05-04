#!/bin/python3

import pandas as pd, numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['xtick.minor.size'] = 0
matplotlib.rcParams['xtick.minor.width'] = 0

# Auxiliary functions: for plotting
def plotter_fourier(v,r, nm, ax):
	ax[r].set_title(f'first {v.shape[1]}\n'+r'$\mathbf{Fourier}$ $\mathbf{'+nm+ '}$ $\mathbf{components}$\n (without the constant term)')
	for i in range(v.shape[0]):
		ax[r].scatter(range(v.shape[1]), v[i,:], alpha=0.5,c='g')
	if False: ax[r].fill_between(range(v.shape[1]),
			 [np.mean(v[:,i])-2*np.std(v[:,i]) for i in range(v.shape[1])],
			 y2=[np.mean(v[:,i])+2*np.std(v[:,i]) for i in range(v.shape[1])],
 			label=r'2 $\sigma$', color='y', alpha=0.2)
	ax[r].fill_between(range(v.shape[1]),
			 [np.mean(v[:,i])-1*np.std(v[:,i]) for i in range(v.shape[1])],
			 y2=[np.mean(v[:,i])+1*np.std(v[:,i]) for i in range(v.shape[1])],
 			label=r'1 $\sigma$', color='y', alpha=0.5)	
	ax[r].plot(range(v.shape[1]), [np.mean(v[:,i]) for i in range(v.shape[1])], label=r'$\mu$', c='r', lw=2)					
	ax[r].legend()


def plotter_raw_data(v,r, ax):
	ax[r].set_title(r'$\mathbf{Raw}$'+r' $\mathbf{Data}$'+'\n(monthly unemployment\nrate 1948-2020)')
	for i in range(v.shape[0]):
		ax[r].plot(range(v.shape[1]), v[i,:], alpha=0.5,c='k')
	ax[r].legend()

def plotter_inverse_fourier(v,r, nm, op, ax):
	ax[r].set_title(r'$\mathbf{Reconstructed}$'+r' $\mathbf{signal}$'+'\nfrom the '+f'frequency\'s \n{nm}')
	mu, sigma = [op(x) for x in get_mu_and_sigma(v)]
	if False: ax[r].fill_between(range(v.shape[1]),
			 mu-2*sigma,
			 y2=mu+2*sigma,
 			label=r'2 $\sigma$', color='y', alpha=0.2)
	ax[r].fill_between(range(v.shape[1]),
			 np.abs(1+np.fft.ifft(mu-1*sigma)),
			 y2=np.abs(1+np.fft.ifft(mu+1*sigma)),
 			label=r'1 $\sigma$', color='y', alpha=0.5)	
	ax[r].scatter(range(v.shape[1]), (1+np.fft.ifft(mu)).real, c='r')
	ax[r].plot(range(v.shape[1]), (1+np.fft.ifft(mu)).real, label=r'$\mu$', c='r', lw=2)					
	ax[r].legend()


def plotter_statistics(v,r, ax):
	ax[r].set_title(f'Mean and std deviation \n '+r'$\mathbf{computed}$ $\mathbf{from}$ $\mathbf{data}$')
	mu, sigma = get_mu_and_sigma(v)
	ax[r].fill_between(range(v.shape[1]),
			 mu-sigma,
			 y2=mu+sigma,
 			label=r'1 $\sigma$', color='y', alpha=0.5)	
	ax[r].scatter(range(v.shape[1]), mu, c='r')
	ax[r].plot(range(v.shape[1]), mu, label=r'$\mu$', c='r', lw=2)					
	ax[r].legend()

# Extract mean and standard deviation from an (N,M) shaped vector, 
# with N samples and M dimensions
def get_mu_and_sigma(v):
	mu = np.asarray([np.mean(v[:,i]) for i in range(v.shape[1])])
	sigma = np.asarray([np.std(v[:,i]) for i in range(v.shape[1])])
	print(f'mu and sigma are {mu} and {sigma} respectively')
	return mu, sigma

