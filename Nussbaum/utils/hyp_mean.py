# -*- coding: utf-8 -*-
"""
Nussbaum - a python package for automated high-resolution Mössbauer spectroscopy temperature profile measurements

@author: Andrew R. C. Grigg*, James M Byrne, Ruben Kretzschmar
*ETH Zurich, Department of Environmental System Science, Institute for Biogeochemistry and Pollutant Dynamics

License: Apache 2.0 (http://www.apache.org/licenses/)

This code provides supporting functions to calculate the mean hyperfine field.

To calculate the mean of the components, it is important to use the folded Gaussian. 
This is effectively the modulus, or at least all positive values. 
The Mean can then be calculated using an analytical function which includs the erf() function. 
"""

import numpy as np
import math



def mean_folded_gaussian(mu,sigma):

    '''
    Calculate the mean of the folded gaussian function. Quite a straight forward function which makes use of 
    the erf() function available in Math library.
    '''

    mean = sigma*np.sqrt(2/np.pi)*np.exp((-mu**2)/(2*sigma**2)) + mu*math.erf(mu/(np.sqrt(2*sigma**2)))

    return mean
    


def compute_adjusted_H(mu,sigma):
    '''
    Function to compute adjusted total hyperfine field per site
    
    ''' 

    mu = mu
    sigma = sigma
    mean_components = [mean_folded_gaussian(mu, sigma)]
    mean_H = np.sum([mean_components])
    
    std = np.sqrt(np.abs(mu**2 + sigma**2 - mean_H**2))
    
    return mean_H, std