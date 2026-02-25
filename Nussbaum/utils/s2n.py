# -*- coding: utf-8 -*-
"""
Nussbaum - a python package for automated high-resolution Mössbauer spectroscopy temperature profile measurements

@author: Andrew R. C. Grigg*, James M Byrne, Ruben Kretzschmar
*ETH Zurich, Department of Environmental System Science, Institute for Biogeochemistry and Pollutant Dynamics

License: Apache 2.0 (http://www.apache.org/licenses/)

Utility functions for calculating signal to noise ratio

"""

import numpy as np

def bkg(data):
    dset=list(data[:30])+list(data[-30:])
    bline=np.mean(dset)
    return(bline)

def s2n(data):
    dset=list(data[:30])+list(data[-30:])
    for s in np.arange(0,len(data),1):        
        noise,bline,signal=np.std(dset),np.mean(dset),(min(data)+(3*np.std(dset)))
        s2n=np.divide(bline-signal,noise)
    return(s2n)

def s2n_sext(data):
    dset=list(data[:30])+list(data[-30:])
    for s in np.arange(0,len(data),1):        
        noise,bline,signal=np.std(dset),np.mean(dset),(min(data[:200])+(3*np.std(dset)))
        s2n=np.divide(bline-signal,noise)
    return(s2n)

def as2n(data):
    dset=list(data[:30])+list(data[-30:])
    noise=np.std(dset)
    signals=[dset-s for s in data]
    tot_sig=np.sum(signals)
    return(np.divide(tot_sig,noise))

def time_curve(cumsum, k, a):
      s2n_c=(k*np.sqrt(cumsum))+a
      return(s2n_c)
  
def time_curve_params(cumsum,s2ns):    
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(time_curve, cumsum, s2ns)
    k_reduction_factor, a_reduction_start=popt[0],popt[1]
    return(k_reduction_factor, a_reduction_start)

def curve_area(spectrum, velocity):
    return np.trapz(spectrum, velocity)