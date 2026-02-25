# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 15:29:19 2024

Functions to be used for the fitting of Mössbuer spectral models
"""

Nussbaum - a python package for automated high-resolution Mössbauer spectroscopy temperature profile measurements

@author: Andrew R. C. Grigg*, James M Byrne, Ruben Kretzschmar
*ETH Zurich, Department of Environmental System Science, Institute for Biogeochemistry and Pollutant Dynamics

License: Apache 2.0 (http://www.apache.org/licenses/)

import numpy as np
from nussbaum.utils import s2n
import astropy.modeling.functional_models as astro
from joblib import Parallel, delayed # Import Parallel and delayed from joblib
import os
import sys
from tqdm import tqdm
import warnings

from scipy.optimize import minimize
from scipy.optimize import fixed_point
from scipy.optimize import curve_fit
import scipy.special as scipysp
from scipy import stats
from scipy import integrate
from scipy import constants
from scipy.integrate import quad
from scipy.signal import convolve
import math
import matplotlib.pyplot as plt

def voigt_byrne(phi, w0, sigma, gamma, h):
    
    '''
    Numercal approximation of a Voigt based peak.

    Input arguments:
        - phi - predictor variable
        - sigma - st dev of distribution of H
        - gamma - broadness of the peak (linewidth)
        - w0 - midpoint of the peak
        - area - peak area       
    '''

    # Numerical coefficients
    ak=[-1.2150, -1.3509, -1.2150, -1.3509]
    bk=[1.2359, 0.3786, -1.2359, -0.3786]
    gk=[-0.3085, 0.5906, -0.3085, 0.5906]
    dk=[0.0210, -1.1858, -0.0210, 1.1858]

    w = gamma

    # define x
    x=((phi-w0)/(sigma*np.sqrt(2)))
    
    # define y
    y=(gamma/(sigma*2*np.sqrt(2)))

    ksum = (
        ((gk[0]*(y - ak[0]) + dk[0]*(x-bk[0]))/((y-ak[0])**2+(x-bk[0])*(x-bk[0])))+
        ((gk[1]*(y - ak[1]) + dk[1]*(x-bk[1]))/((y-ak[1])**2+(x-bk[1])*(x-bk[1])))+
        ((gk[2]*(y - ak[2]) + dk[2]*(x-bk[2]))/((y-ak[2])**2+(x-bk[2])*(x-bk[2])))+
        ((gk[3]*(y - ak[3]) + dk[3]*(x-bk[3]))/((y-ak[3])**2+(x-bk[3])*(x-bk[3]))))

    Voigt_const=((np.pi*h*w)/(2*np.sqrt(2*np.pi)*sigma))

    V = Voigt_const * ksum

    return -V
 
#%%

def doublet_VBF(x,CS,QS,sigma=0,intensity=30000,counts=1):
    
    '''
    VBF doublet.

    Input arguments:
        - x - velocity scale
        - CS - centre shift in mm/s
        - epsilon - quadrupole shift in mm/s
        - sigma - standard deviation of the gaussian component of the voigt peak (QS)
        - intensity - scaling factor for the size of the peak
        - counts - background counts
    '''

    gamma=0.139627911023937
    
    # p1
    x1 = 0 + CS -0.5*QS
    p1 = voigt_byrne(x, x1, sigma, gamma, intensity*0.5/(np.pi*2*gamma)) 

    # p2
    x2 = 0 + CS +0.5*QS
    p2 = voigt_byrne(x, x2, sigma, gamma, intensity*0.5/(np.pi*2*gamma))

    y=counts-(-1*(p1+p2))
    return (y)

def doublet(x,CS,QS,sigma=0,intensity=30000,counts=1):

    warnings.warn(
    "The function `doublet()` is deprecated. Use `doublet_VBF()` instead.",
    DeprecationWarning,
    stacklevel=2
)

    return doublet_VBF(x,CS,QS,sigma=0,intensity=30000,counts=1)

def doublet_xVBF(x,CS,QS,sigma_CS=0,sigma_QS=0,intensity=30000,counts=1):

    '''
    xVBF doublet.

    Input arguments:
        - x - velocity scale
        - CS - centre shift in mm/s
        - epsilon - quadrupole shift in mm/s
        - sigma_CS - standard deviation of the distribution of the CS
        - sigma_QS - standard deviation of the distribution of the QS
        - intensity - scaling factor for the size of the peak
        - counts - background counts
    '''    

    gamma=0.139627911023937
   
    # p1
    x1 = 0 + CS -0.5*QS
    p1 = voigt_byrne(x, x1, np.sqrt((sigma_QS/2)**2 + sigma_CS**2), gamma, intensity*0.5/(np.pi*2*gamma)) 

    # p2
    x2 = 0 + CS +0.5*QS
    p2 = voigt_byrne(x, x2, np.sqrt((sigma_QS/2)**2 + sigma_CS**2), gamma, intensity*0.5/(np.pi*2*gamma))

    y=counts-(-1*(p1+p2))
    return (y)

def doublet_xVBF_relax(x,CS,QS,sigma_CS=0,sigma_QS=0,intensity=30000,counts=1, W_relax=10e10, C=0.1):

    '''
    xVBF doublet with broadening of the inner lines to simulate dynamic relaxation

    Input arguments:
        - x - velocity scale
        - CS - centre shift in mm/s
        - epsilon - quadrupole shift in mm/s
        - sigma_CS - standard deviation of the distribution of the CS
        - sigma_QS - standard deviation of the distribution of the QS
        - intensity - scaling factor for the size of the peak
        - counts - background counts
        - W_relax - relaxation time of the spectrum
        - C - scalaing factor relating W-relax to the width of the Lorentzian part of the voigt profile
    '''        

    gamma=0.139627911023937
    
    #set limits on the broadening of the peaks due to dynamic relaxation
    L4=gamma + C * W_relax
    Max_LL=0.3
    if L4 > Max_LL: L4 = Max_LL
    
    # p1
    x1 = 0 + CS -0.5*QS
    p1 = voigt_byrne(x, x1, np.sqrt((sigma_QS/2)**2 + sigma_CS**2), L4, intensity*0.5/(np.pi*2*L4)) 

    # p2
    x2 = 0 + CS +0.5*QS
    p2 = voigt_byrne(x, x2, np.sqrt((sigma_QS/2)**2 + sigma_CS**2), L4, intensity*0.5/(np.pi*2*L4))

    y=counts-(-1*(p1+p2))
    return (y)


#%%

def sextet_VBF(x,CS,epsilon,H,sigma=0,intensity=30000,counts=0):
    
    '''
    VBF sextet.

    Input arguments:
        - x - velocity scale
        - CS - centre shift in mm/s
        - epsilon - quadrupole shift in mm/s
        - H - hyperfine field in T
        - sigma - standard deviation of the gaussian component of the voigt peak (H)
        - intensity - scaling factor for the size of the peak
        - counts - background counts
    '''
    
    gamma=0.139627911023937
    # sigma=0.5*sigma # I am not sure why this is necessary
    sigma=1.0*sigma # I am not sure why this is necessary


    mun=3.1524512605e-8 #ev/T
    # mun=5.050783699e-27
    mu0=0.09024*mun
    mu1=0.1535*mun
    
    c=299792458000  #mm/s
    Egam=14412.497 #eV
    
    # 1T = 10000 Oe
    
    B=H

    # gg= (c/Egam)*(B*mu0)/(1/2)
    ge= (c/Egam)*(B*mu1)/(3/2)
    sig=(c/Egam)*(sigma*mu0)/(1/2)
       
    Z=1.7509
    z=ge   
       
    gamma2=gamma*2 # convert to FWHM
    
    # p1
    x1 = 0 + CS + epsilon - ((Z + 3) * (z / 2))
    LorAmp1 = (3/12)*intensity/(np.pi*gamma2)
    p1 = voigt_byrne(x, x1, np.abs((Z+3)*(1/2)*sig), gamma, LorAmp1)

    # p2
    x2 = 0 + CS - epsilon - ((Z + 1) * (z / 2))
    LorAmp2 = (2/12)*intensity/(np.pi*gamma2)
    p2 = voigt_byrne(x, x2, np.abs((Z+1)*(1/2)*sig), gamma, LorAmp2)

    # p3
    x3 = 0 + CS - epsilon - ((Z - 1) * (z / 2))
    LorAmp3 = (1/12)*intensity/(np.pi*gamma2)
    p3 = voigt_byrne(x, x3, np.abs((Z-1)*(1/2)*sig), gamma, LorAmp3)

    # p4
    x4 = 0 + CS - epsilon + ((Z - 1) * (z / 2))
    LorAmp4 = (1/12)*intensity/(np.pi*gamma2)
    p4 = voigt_byrne(x, x4, np.abs((Z-1)*(1/2)*sig), gamma, LorAmp4)

    # p5
    x5 = 0 + CS - epsilon + ((Z + 1) * (z / 2))
    LorAmp5 = (2/12)*intensity/(np.pi*gamma2)
    p5 = voigt_byrne(x, x5, np.abs((Z+1)*(1/2)*sig), gamma, LorAmp5)

    # p6
    x6 = 0 + CS + epsilon + ((Z + 3) * (z / 2))
    LorAmp6 = (3/12)*intensity/(np.pi*gamma2)
    p6 = voigt_byrne(x, x6, np.abs((Z+3)*(1/2)*sig), gamma, LorAmp6)
    
    y=counts+((p1+p2+p3+p4+p5+p6
                         ))
    return (y)

def sextet(x,CS,epsilon,H,sigma=0,intensity=30000,counts=0):

    warnings.warn(
    "The function `sextet()` is deprecated. Use `sextet_VBF()` instead.",
    DeprecationWarning,
    stacklevel=2
)
    return sextet_VBF(x,CS,epsilon,H,sigma=0,intensity=30000,counts=0)

def sextet_xVBF(x,CS,epsilon,H,sigma_CS=0,sigma_ep=0,sigma_H=0,intensity=30000,counts=0):

    '''
    VBF sextet.

    Input arguments:
        - x - velocity scale
        - CS - centre shift in mm/s
        - epsilon - quadrupole shift in mm/s
        - H - hyperfine field in T
        - sigma_CS - standard deviation of the distribution of the CS
        - sigma_ep - standard deviation of the distribution of the QS
        - sigma_H - standard deviation of the gaussian component of the voigt peak (H)
        - intensity - scaling factor for the size of the peak
        - counts - background counts
    '''    

    gamma=0.139627911023937

    mun=3.1524512605e-8 #ev/T
    mu0=0.09024*mun
    mu1=0.1535*mun    
    c=299792458000  #mm/s
    Egam=14412.497 #eV    
    # 1T = 10000 Oe   
    B=H

    # gg= (c/Egam)*(B*mu0)/(1/2)
    ge= (c/Egam)*(B*mu1)/(3/2)
    sigH=(c/Egam)*(sigma_H*mu0)/(1/2)
       
    Z=1.7509
    z=ge   
       
    gamma2=gamma*2 # convert to FWHM
    
    # p1
    x1 = 0 + CS + epsilon - ((Z + 3) * (z / 2))
    LorAmp1 = (3/12)*intensity/(np.pi*gamma2)
    p1 = voigt_byrne(x, x1, np.sqrt( ((Z+3)*(1/2)*sigH)**2 + sigma_ep**2 + sigma_CS**2 ), gamma2, LorAmp1)

    # p2
    x2 = 0 + CS - epsilon - ((Z + 1) * (z / 2))
    LorAmp2 = (2/12)*intensity/(np.pi*gamma2)
    p2 = voigt_byrne(x, x2, np.sqrt( ((Z+1)*(1/2)*sigH)**2 + sigma_ep**2 + sigma_CS**2 ), gamma2, LorAmp2)

    # p3
    x3 = 0 + CS - epsilon - ((Z - 1) * (z / 2))
    LorAmp3 = (1/12)*intensity/(np.pi*gamma2)
    p3 = voigt_byrne(x, x3, np.sqrt( ((Z-1)*(1/2)*sigH)**2 + sigma_ep**2 + sigma_CS**2 ), gamma2, LorAmp3)

    # p4
    x4 = 0 + CS - epsilon + ((Z - 1) * (z / 2))
    LorAmp4 = (1/12)*intensity/(np.pi*gamma2)
    p4 = voigt_byrne(x, x4, np.sqrt( ((Z-1)*(1/2)*sigH)**2 + sigma_ep**2 + sigma_CS**2 ), gamma2, LorAmp4)

    # p5
    x5 = 0 + CS - epsilon + ((Z + 1) * (z / 2))
    LorAmp5 = (2/12)*intensity/(np.pi*gamma2)
    p5 = voigt_byrne(x, x5, np.sqrt( ((Z+1)*(1/2)*sigH)**2 + sigma_ep**2 + sigma_CS**2 ), gamma2, LorAmp5)

    # p6
    x6 = 0 + CS + epsilon + ((Z + 3) * (z / 2))
    LorAmp6 = (3/12)*intensity/(np.pi*gamma2)
    p6 = voigt_byrne(x, x6, np.sqrt( ((Z+3)*(1/2)*sigH)**2 + sigma_ep**2 + sigma_CS**2 ), gamma2, LorAmp6)
    
    y=counts+(p1+p2+p3+p4+p5+p6
                         )
    return (y)    

def sextet_xVBF_relax(x,CS,epsilon,H,sigma_CS=0,sigma_ep=0,sigma_H=0,intensity=30000,counts=0, W_relax=10e10, A=0.1, B=0.3):

    '''
    VBF sextet.

    Input arguments:
        - x - velocity scale
        - CS - centre shift in mm/s
        - epsilon - quadrupole shift in mm/s
        - H - hyperfine field in T
        - sigma_CS - standard deviation of the distribution of the CS
        - sigma_ep - standard deviation of the distribution of the QS
        - sigma_H - standard deviation of the gaussian component of the voigt peak (H)
        - intensity - scaling factor for the size of the peak
        - counts - background counts
        - W_relax - relaxation time of the spectrum
        - A - scalaing factor relating W-relax to the width of the Lorentzian part of the voigt profile for peaks 2 and 5
        - B - scalaing factor relating W-relax to the width of the Lorentzian part of the voigt profile for peaks 3 and 4
    '''        

    gamma=0.139627911023937
    sigmaH=1.0*sigma_H # I am not sure why this is necessary

    mun=3.1524512605e-8 #ev/T
    # mun=5.050783699e-27
    mu0=0.09024*mun
    mu1=0.1535*mun
    
    c=299792458000  #mm/s
    Egam=14412.497 #eV
    
    # 1T = 10000 Oe
    B=H

    ge= (c/Egam)*(B*mu1)/(3/2)
    sigH=(c/Egam)*(sigmaH*mu0)/(1/2)
       
    Z=1.7509
    z=ge   
       
    gamma2=gamma*2 # convert to FWHM
    
    Max_L2=0.8
    Max_L3=1.0
    L1=gamma2
    L2=2*gamma + A * W_relax
    if L2 > Max_L2: L2 = Max_L2
    L3=2*gamma + B * W_relax
    if L3 > Max_L3: L3 = Max_L3
    
    # p1
    x1 = 0 + CS + epsilon - ((Z + 3) * (z / 2))
    LorAmp1 = (3/12)*intensity/(np.pi*gamma2)
    p1 = voigt_byrne(x, x1, np.sqrt( ((Z+3)*(1/2)*sigH)**2 + sigma_ep**2 + sigma_CS**2 ), L1, LorAmp1)

    # p2
    x2 = 0 + CS - epsilon - ((Z + 1) * (z / 2))
    LorAmp2 = (2/12)*intensity/(np.pi*gamma2)
    p2 = voigt_byrne(x, x2, np.sqrt( ((Z+1)*(1/2)*sigH)**2 + sigma_ep**2 + sigma_CS**2 ), L2, LorAmp2)

    # p3
    x3 = 0 + CS - epsilon - ((Z - 1) * (z / 2))
    LorAmp3 = (1/12)*intensity/(np.pi*gamma2)
    p3 = voigt_byrne(x, x3, np.sqrt( ((Z-1)*(1/2)*sigH)**2 + sigma_ep**2 + sigma_CS**2 ), L3, LorAmp3)

    # p4
    x4 = 0 + CS - epsilon + ((Z - 1) * (z / 2))
    LorAmp4 = (1/12)*intensity/(np.pi*gamma2)
    p4 = voigt_byrne(x, x4, np.sqrt( ((Z-1)*(1/2)*sigH)**2 + sigma_ep**2 + sigma_CS**2 ), L3, LorAmp4)

    # p5
    x5 = 0 + CS - epsilon + ((Z + 1) * (z / 2))
    LorAmp5 = (2/12)*intensity/(np.pi*gamma2)
    p5 = voigt_byrne(x, x5, np.sqrt( ((Z+1)*(1/2)*sigH)**2 + sigma_ep**2 + sigma_CS**2 ), L2, LorAmp5)

    # p6
    x6 = 0 + CS + epsilon + ((Z + 3) * (z / 2))
    LorAmp6 = (3/12)*intensity/(np.pi*gamma2)
    p6 = voigt_byrne(x, x6, np.sqrt( ((Z+3)*(1/2)*sigH)**2 + sigma_ep**2 + sigma_CS**2 ), L1, LorAmp6)
    
    y=counts+(p1+p2+p3+p4+p5+p6)
    return (y)    

#%%

def Blume(Sig1, Sig2, Q1, Q2, H1, H2, L, W, m0, m1, V, R):
    """
    the Blume function is the same implementation as found in the Syncmoss source code
    """
    
    ggr = 0.18121*0.656
    gex = -0.10353*0.656 # 0.656 is a conversion from a dimensionless g factor to a velocity g factor
    
    alf = (ggr * m0 - gex * m1) * (H2-H1)/2 + (Q2-Q1)/2 * (3 * m1 ** 2 - 15 / 4) + (Sig2-Sig1)/2
    a = V - (Sig1+Sig2)/2 - (Q2+Q1)/2 * (3 * m1 ** 2 - 15 / 4) - (ggr * m0 - gex * m1) * (H2+H1)/2
    Line = np.array([float(0)]*len(V))
    for i in range(0, len(V)):
        a1 = 1j*(a[i]+alf) + L/2 + W
        a3 = -R*W
        a2 = -W
        a4 = 1j*(a[i]-alf) + L/2 + R*W
        delimeter = 1/(a1*a4-a2*a3)
        b1 = np.real(a4 *  delimeter)
        b2 = np.real(-a2 * delimeter)
        b3 = np.real(-a3 * delimeter)
        b4 = np.real(a1 *  delimeter)
        Line[i] = (b1+b2)*R/(R+1) + (b3+b4)/(R+1)

    return(Line/np.pi)

def create_gaussian_samples(mean, sigma, N_STEPS):
    """
    This wrapper uses the Blume-Tjon function as impelemented in SyncMoss,
    but is extended to convolve a gaussian distribution of the key hyperfine parameters
    using a numerical approximation for a convolution, this function adds a gaussian element for the H and QS
    Creates discrete sample points and weights for a Gaussian distribution.
    Used for "slow" numerical integration loops.
    THIS FUNCTION WAS NOT IMPLEMENTED IN THE FINAL RESULT IN THE PAPER
    
    Args:
        mean (float): The center of the distribution (e.g., H_mean).
        sigma (float): The width of the distribution (e.g., sigma_H).
        N_STEPS (int): The number of points to sample.
        
    Returns:
        (np.ndarray, np.ndarray): A tuple of (points, weights)
    """
    
    # If sigma is tiny, just return the mean value
    if sigma < 1e-6:
        return np.array([mean]), np.array([1.0])
        
    # Define the integration range - this captures 99.7% of the distribution
    cutoff = 3.0 * sigma
    
    # Create linearly spaced points across this range to loop over
    points = np.linspace(mean - cutoff, mean + cutoff, N_STEPS)
    
    # Calculate the Gaussian "weight" for each point
    weights = np.exp(-0.5 * ((points - mean) / sigma)**2)
    
    # Normalize the weights to sum to 1.0
    weights = weights / np.sum(weights)
    
    return points, weights

# 
def create_gaussian_kernel(x_values, sigma):
    """
    Creates a 1D Gaussian kernel for fast convolution.
    
    Using a FFT, this function convolves the spectra with a gausian for the CS,
    but is also used to mimic convolution of other prameters with cheaper computational cost
    
    Args:
        x_values (np.ndarray): The velocity axis (used for scale).
        sigma (float): The Gaussian width in physical units (e.g., mm/s).
        
    Returns:
        np.ndarray: A 1D kernel, normalized to sum to 1.0.
    """
    
    # If sigma is tiny, just return a delta function.
    if sigma < 1e-6:
        n_points = len(x_values)
        kernel = np.zeros(n_points)
        kernel[n_points // 2] = 1.0
        return kernel
        
    # 1. Get properties of the x-axis and cnvert sigma from physical units (mm/s) to "pixels"
    n_points = len(x_values)
    delta_x = x_values[1] - x_values[0]
    sigma_pixels = sigma / delta_x
    
    # Create a centered x-axis for the kernel (in pixels)
    kernel_x = np.arange(n_points) - n_points // 2
    
    # Calculate the Gaussian
    kernel = np.exp(-0.5 * (kernel_x / sigma_pixels)**2)
    
    # Normalize the kernel to sum to 1.0
    return kernel / np.sum(kernel)

    
def extended_blume_tjon(x_values, CS_mean, sigma_CS, 
                        QS, sigma_QS, H, sigma_H, Gamma_relax, linewidth_L=0.135, 
                        H_STEPS=1, EPS_STEPS=1): 
    
    """
    Calculates a Mössbauer transmission spectrum using the Blume-Tjon relaxation model, 
    extended to account for Gaussian distributions in Hyperfine Field (H), 
    Quadrupole Splitting (QS), and Isomer Shift (CS).

    This function simulates the relaxation between two magnetic states (+H and -H) 
    and allows for parameter broadening via two distinct methods: fast convolution 
    approximation or explicit numerical integration.

    Parameters
    ----------
    x_values : velocity axis values (mm/s).
    CS_mean : Isomer Shift (Centre Shift) in mm/s.
    sigma_CS : Standard deviation of the Isomer Shift distribution (Gaussian broadening) in mm/s.
    QS : Mean Quadrupole Splitting in mm/s.
    sigma_QS : Standard deviation of the Quadrupole Splitting distribution in mm/s.
    H : Mean Magnetic Hyperfine Field in Tesla.
    sigma_H : Standard deviation of the Magnetic Field distribution in Tesla.
    Gamma_relax : Relaxation rate (hopping frequency W) between magnetic states.
    linewidth_L : Natural linewidth (Lorentzian FWHM) of the source/absorber. Default is 0.135 mm/s.
    H_STEPS : Number of integration steps for the magnetic field distribution.
        - If 1: Uses an approximation (convolves lines with a Gaussian width derived from sigma_H).
        - If >1: Performs explicit numerical integration (summation) over `H_STEPS` samples.
        Default is 1.
    EPS_STEPS : Number of integration steps for the Quadrupole Splitting distribution.
        - If 1: Uses a "Fast Hack" approximation (convolves lines with a Gaussian width derived from sigma_QS).
        - If >1: Performs explicit numerical integration (summation) over `EPS_STEPS` samples.
        Default is 1.

    Returns: The calculated spectral intensity array corresponding to `x_values`.
    
    """
    
    # HANDLE MAGNETIC SMEAR
    if int(H_STEPS) == 1:
        H_dist = np.array([H])
        H_weights = np.array([1.0])
        base_magnetic_smear_mms = sigma_H * 0.161 #convert T to mm/s
    else:
        H_dist, H_weights = create_gaussian_samples(H, sigma_H, H_STEPS)
        base_magnetic_smear_mms = 0.0 

    # HANDLE QUADRUPOLE SMEAR
    if int(EPS_STEPS) == 1:
        Eps_dist = np.array([QS])
        Eps_weights = np.array([1.0])
        qs_smear_mms = sigma_QS * 3.0
    else:
        Eps_dist, Eps_weights = create_gaussian_samples(QS, sigma_QS, EPS_STEPS)
        qs_smear_mms = 0.0 

    sensitivity_factors = [1.0, 0.58, 0.16, 1.0, 0.58, 0.16]

    # Initialize accumulator
    unsmeared_spectrum = np.zeros_like(x_values)
    
    # THE BLUME LOOP
    for H_val, H_weight in zip(H_dist, H_weights):
        for Eps_val, Eps_weight in zip(Eps_dist, Eps_weights):          
            sextet_accumulator = np.zeros_like(x_values)          
            transitions = [(-0.5, -1.5, 3.0/12.0), (-0.5, -0.5, 2.0/12.0), (-0.5, 0.5, 1.0/12.0),
                           ( 0.5,  1.5, 3.0/12.0), ( 0.5,  0.5, 2.0/12.0), ( 0.5, -0.5, 1.0/12.0)]
            
            for i, (m0, m1, intensity) in enumerate(transitions):
                line_shape = Blume(
                    Sig1=CS_mean, Sig2=CS_mean, 
                    Q1=Eps_val, Q2=Eps_val,     
                    H1=H_val, H2=-1*H_val,      
                    L=linewidth_L,           
                    W=Gamma_relax,
                    m0=m0, m1=m1,
                    V=x_values, R=1.0
                )
                
                specific_magnetic_smear_mms = base_magnetic_smear_mms * sensitivity_factors[i] #apply sensitivity factor for each line
                
                # COMBINE WIDTHS
                total_line_width = np.sqrt(sigma_CS**2 + specific_magnetic_smear_mms**2 + qs_smear_mms**2)
                
                # Convolve each line with the gaussian smear ("Stretching")
                if total_line_width > 1e-4:
                    kernel = create_gaussian_kernel(x_values, total_line_width)
                    line_shape = np.convolve(line_shape, kernel, mode='same')
                
                sextet_accumulator += line_shape * intensity
            
            unsmeared_spectrum += sextet_accumulator * H_weight * Eps_weight

    return unsmeared_spectrum

#%%

# A series of functions required for the fitting of temperature-velocity matrix

def calculate_QS(thD, T, QS_nought):
    return QS_nought   

# def calculate_QS(thD, T, QS_nought):
#     '''   

#     Parameters
#     ----------
#     thD : the Debye temperature (K)
#     T : measurement temperature (K)
#     QS_nought : the QS at 0 K

#     Returns
#     -------
#     QS at this temperature

#     '''
        
#     def intfun(x):
#         if abs(x) < 1e-6:  # Threshold for 'close to zero' - adjust if needed
#             return x**(1/2) / (x + 1e-9) # Approximation for small x, adding small term to denominator to avoid division by zero at x=0 if needed during integration.
#         else:
#             return (x**(1/2)) / (-1 + math.exp(x))
#     ul=thD/T
#     integral=integrate.quad(lambda x: intfun(x),0,ul)
#     DELeQ=QS_nought*(1-(3/2)*(T/thD)**(3/2)*integral[0])
#     return DELeQ    

def calculate_CS(thD, T, del1):
    '''
    

    Parameters
    ----------
     thD : the Debye temperature (K)
     T : measurement temperature (K)
     CS_nought : the QS at 0 K

    Returns
    -------
    CS at this temperature

    '''
    
    def delSOD(thD, T):
        M=9.2732796e-23
        def intfun(x):
            if abs(x) < 1e-6:  # Threshold for 'close to zero' - adjust if needed
                return x**2 # Approximation for small x
            else:
                return (x**3) / (-1 + math.exp(x))
        
        ul=thD/T
        integral=integrate.quad(lambda x: intfun(x),0,ul)
        delS=1000000*((-9*constants.Boltzmann*thD)/(16*M*constants.c))*(1+(8*(T/thD)**4)*integral[0])
        # factor of 1000000 because the unit of (J.s/m.g) cancels out to 1000 m/s = 1000000 mm/s
        return delS
  
    return del1+delSOD(thD, T)   

def calculate_f_factor(T_measured, thD):
    """
    Calculates the Mössbauer f-factor (recoilless fraction)
    using the full Debye model integral.

    Args:
        T_measured (float): Measurement temperature (Kelvin).
        thD (float): Debye temperature (Kelvin).

    Returns:
        float: f-factor value at T_measured.
    """
    
    if T_measured < 1e-3: T_measured = 1e-3  # Avoid division by zero at T=0
    if thD < 1e-3: thD = 1e-3            # Avoid division by zero
    
    # E_R = E_gamma^2 / (2 * m_Fe57 * c^2)
    # The whole constant term 6*E_R / k_B for 57Fe is ~136.5 K
    PREFACTOR = 136.5  # K

    def integral_func(x):
        """Integrand for the Debye-Waller factor."""
        if abs(x) < 1e-6:
            return 1.0  # Limit of x/(e^x - 1) as x -> 0 is 1
        return x / (math.exp(x) - 1)

    # Upper limit for integration
    ul = thD / T_measured
    
    # Calculate the integral
    integral_val, _ = quad(integral_func, 0, ul)

    # Debye-Waller factor W(T)
    W_T = (PREFACTOR / thD) * (0.25 + (T_measured / thD)**2 * integral_val)
    
    # f-factor = exp(-W(T))
    f_factor = math.exp(-W_T)
    
    return f_factor

def coth(i):
    """
    Numerically stable hyperbolic cotangent function.

    Args:
        i (float or numpy.ndarray): Input value(s).

    Returns:
        float or numpy.ndarray: Hyperbolic cotangent of i.
        
    Approximation for Large i:
    If i > 20: We approximate coth(i) as 1.0. For large positive i, coth(i) approaches 1.
    If i < -20: We approximate coth(i) as -1.0. For large negative i, coth(i) approaches -1.
    the threshold of 20 can be altered
    """

    if isinstance(i, np.ndarray):
        output = np.zeros_like(i, dtype=float)
        for index, val in np.ndenumerate(i):
            if val > 20:  # Threshold for large positive values
                output[index] = 1.0
            elif val < -20: # Threshold for large negative values
                output[index] = -1.0
            else:
                output[index] = np.cosh(val) / np.sinh(val)
        return output
    else: # For scalar input
        if i > 20:
            return 1.0
        elif i < -20:
            return -1.0
        else:
            return np.cosh(i) / np.sinh(i)


def brillouin (Temp,T_Block,B_sat,B_Temp,nu):
    """.
    Calculates the magnetic hyperfine field using a Brillouin function. 
    The implementation is tailored for J=2.5 (Fe3+), 
    using hard-coded constants for J=2.5.

    Used in a self-consistent loop to determine the magnetization (or Hyperfine Field) vs Temperature curve.

    Parameters
    ----------
    Temp : Current temperature in Kelvin.
    T_Block : The magnetic ordering temperature (Blocking temperature) in Kelvin.
    B_sat : Saturation magnetic field (at T=0) in Tesla.
    B_Temp : The magnetic field at the current temperature `Temp`. 
        (Used to calculate the reduced magnetization for the Bean-Rodbell correction).
    nu : The Bean-Rodbell deformation parameter ($\eta$), which characterizes the 
        volume dependence of the exchange integral.


    Returns
    -------
    float
        The calculated magnetic field value (B_val) in Tesla.
    """


    def BeanRodbell(Temp,T_Block,B_sat,B_Temp,nu):
        """Bean-Rodbell approximation for x.
        Calculates the argument 'x' for the Brillouin function 
        using the Bean-Rodbell approximation.
        
        This computes x = (g * mu_B * H_eff) / (k_B * T), modified by the Bean-Rodbell 
        term which scales the ordering temperature based on the square of the 
        reduced magnetization ((B_Temp/B_sat)^2).
        
        Returns
        -------
        The argument 'x' to be passed into the coth (Brillouin) function.
        """
            
        if abs(Temp) < 1e-9:  # Handle near-zero Temp
            return np.inf
        if abs(B_sat) < 1e-9: # Handle near-zero B_sat
            return 0.0 # Return 0 for x if B_sat is very close to zero
        x=(-2*(-15/14)-4*(2331/9604)*((B_Temp**2)/(B_sat**2))*nu)*((T_Block)/Temp)*(B_Temp/B_sat)
        return x

    x=BeanRodbell(Temp,T_Block,B_sat,B_Temp,nu)

    if x == 0:
        B_val=0
    else:
        B_val=B_sat*( (6)/(5)*coth((6)/(5)*x) -  1/(5)*coth(x/(5)) )
    return B_val
   
def Temp_H(Temp, T_Block, B_sat, B_Temp_init_input, nu):
    """  
    Calculates the temperature-dependent magnetic hyperfine field by solving the 
    Brillouin equation self-consistently.

    Since the internal magnetic field in the Brillouin function depends on the 
    magnetization itself (Mean Field Theory), this function uses a fixed-point 
    iteration method to find the stable solution for B at a given Temperature.

    Parameters
    ----------
    Temp : float
        The temperature of interest in Kelvin.
    T_Block : float
        The magnetic ordering temperature (e.g., Neel or Blocking temperature) in Kelvin.
    B_sat : float
        The saturation magnetic field (at T=0) in Tesla.
    B_Temp_init_input : float
        Initial guess for the magnetic field in Tesla. 
        A good guess (e.g., slightly less than B_sat) is crucial for the 
        fixed-point solver to converge to the non-zero solution below Tc.
    nu : float
        The Bean-Rodbell deformation parameter.

    Returns
    -------
    float
        The self-consistent magnetic hyperfine field in Tesla. 
        Returns 0.0 if `Temp` > `T_Block`.

    Raises
    ------
    RuntimeError
        If the fixed-point iteration fails to converge within the maximum number of iterations (3000).
    """
    
    if Temp > (T_Block + 0.1):
        return 0.0
    
    def recursive_B_Temp(B_Temp, iteration_counter=[0]): # Initialize iteration counter
        iteration_counter[0] += 1 # Increment counter
        brillouin_result = brillouin(Temp, T_Block, B_sat, B_Temp, nu) # Call brillouin here
        return brillouin_result

    B_Temp_init = B_Temp_init_input # Use input init value

    try:
        B_Temp_solution = fixed_point(recursive_B_Temp, B_Temp_init, maxiter=3000,  xtol=1e-3) # Increased maxiter
    except RuntimeError as e:
        raise # Re-raise the exception to stop optimization

    return B_Temp_solution

   
def Temp_distribution(T_Block,T_B_sigma,res=100):
    """
    Generates a discretized Gaussian distribution of Blocking Temperatures (T_Block).

    This function creates a range of T_Block values sampled from a normal distribution
    defined by a mean (`T_Block`) and standard deviation (`T_B_sigma`). It uses 
    inverse transform sampling (via the Percent Point Function, ppf) to generate 
    samples corresponding to evenly spaced probabilities.

    The sampling range is implicitly clamped to reasonable physical bounds 
    (roughly -50K to 295K) via the CDF calculation to avoid extreme outliers 
    or unphysical infinite values.

    Parameters
    ----------
    T_Block : The mean Blocking Temperature (center of the distribution) in Kelvin.
    T_B_sigma : The standard deviation of the Blocking Temperature distribution in Kelvin.
    res : The resolution (number of sample points) to generate. Default is 100.

    Returns
    -------
    An array of `res` Blocking Temperature values sorted from low to high.
    """
    
    # set the distribution for loc=mean blocking temperature and scale=stdev of blocking tmeprature distribution
    distribution = stats.norm(loc=T_Block, scale=T_B_sigma)
    
    # percentile point, the range for the inverse cumulative distribution function:
    bounds_for_range = distribution.cdf([-50, 295])
    
    # Linspace for the inverse cdf:
    pp = np.linspace(*bounds_for_range, num=res)
    
    T_Block_dist = distribution.ppf(pp)
    if T_Block_dist[-1]==np.inf:
        T_Block_dist[-1]=bounds_for_range[-1]
    if T_Block_dist[0]==-np.inf:
        T_Block_dist[0]=bounds_for_range[0]
    
    return(T_Block_dist)


def prefit_debye_temp(spectra_dict, velocity_axis):
    """ 
    A function to fit the debye temperature based on measured area of the spectra
    """ 

    temps = sorted(spectra_dict.keys())
    areas = []
    valid_temps = []
    
    # Integrate the Experimental Data
    for T in temps:
        spectrum = spectra_dict[T]
        
        # Simple Trapezoidal Integration of the absorption
        absorption = 1.0 - spectrum
        area = np.trapz(absorption, x=velocity_axis)
            
    areas = np.array(areas)
    valid_temps = np.array(valid_temps)
    
    # Normalize areas to the lowest temperature
    normalized_areas = areas / areas[0]

    # 2. Define the Model to fit ONLY the areas
    def area_model(T, fitted_thD, scale_factor):
        # Calculate f-factor for all T
        
        f_vals = np.array([calculate_f_factor(t, fitted_thD) for t in T])
        f_ref = calculate_f_factor(valid_temps[0], fitted_thD)
        
        # The curve should follow the ratio of f-factors
        return scale_factor * (f_vals / f_ref)

    # Fit
    popt, pcov = curve_fit(area_model, valid_temps, normalized_areas, p0=[350, 1.0], bounds=([100, 0.5], [1000, 1.5]))
    
    fitted_thD = popt[0]
    fitted_scale = popt[1]
    
    print(f"Pre-fitted thD: {fitted_thD:.2f} K")  
    return fitted_thD, fitted_scale

#%%

# functions that create lineshapes that are appropriately collapsed for the temperature

def collapsed_static(x,
                     thD=400,
                     del1=0.6,
                     QS_nought=0.4,
                     sigma_QS=0,
                     sigma_ep=0,
                     B_sat=0,
                     sigma_H=0,
                     intensity=1,
                     counts=1 ,
                     T_Block=4,
                     sig_T_Block=70,
                     T_measured=65,
                     nu=2,
                     theta=0.78,
                     sigTB_res=100,
                    ):
    """
    Simulates a "collapsed" Mössbauer spectrum using a weighted sum of static (xVBF) models.

    This function models a system of nanoparticles with a distribution of blocking 
    temperatures. It calculates the spectrum by averaging distinct static sextets 
    and doublets with an array of paramters.
    
    For each particle in the distribution:
    1. The magnetic hyperfine field H is calculated self-consistently (Mean Field Theory).
    2. If H > 0, a static sextet is generated (using `sextet_xVBF`).
    3. If H ~ 0, a static doublet is generated (using `doublet_xVBF`).

    This approach does not model relaxation line broadening (wings/asymmetry); 
    it only models the loss of magnetic splitting due to thermal fluctuations 
    (superparamagnetism) as a binary "on/off" or reduced-field effect.

    Parameters
    ----------
    x : Velocity axis in mm/s.
    thD : Debye temperature in Kelvin (affects Isomer Shift and f-factor). Default 400.
    del1 : Isomer Shift offset parameter. Default 0.6.
    QS_nought : Intrinsic Quadrupole Splitting at T=0. Default 0.4.
    sigma_QS : Distribution width of Quadrupole Splitting (Voigt broadening).
    sigma_ep : Distribution width of the epsilon parameter (Sextet quadrupole interaction).
    B_sat : Saturation Hyperfine Field at T=0 (Tesla).
    sigma_H : Distribution width of the Hyperfine Field (Voigt broadening).
    intensity : Global intensity scaling factor.
    counts : Baseline counts (usually 1.0 for normalized transmission).
    T_Block : Mean Blocking Temperature of the particle distribution (Kelvin).
    sig_T_Block : Standard deviation of the Blocking Temperature distribution (Kelvin).
    T_measured : The experimental measurement temperature (Kelvin).
    nu : Bean-Rodbell deformation parameter.
    theta : Angle between the magnetic field and the principal axis of the EFG (in radians).
    sigTB_res : Resolution (number of steps) for the Blocking Temperature integration.

    Returns
    -------
    tuple
        (collapsed_spectrum, sext_mean, doub_mean)
        - collapsed_spectrum: The final total calculated spectrum (product of transmission).
        - sext_mean: The isolated contribution from magnetic sextets.
        - doub_mean: The isolated contribution from paramagnetic doublets.
    """

    f_factor_temp = calculate_f_factor(T_measured, thD) 

    T_Block_dist = Temp_distribution(T_Block, sig_T_Block, res=sigTB_res)
    B_Temp_dist = []

    # This loop calculates the H-field distribution at T_measured based on the T_Block distribution
    for T_Block_val in T_Block_dist[1:-1]:  
        fail = False
        try:
            B = abs(float(Temp_H(T_measured, T_Block_val, B_sat, B_sat - 5, nu, J=2.5)))
            B_Temp_dist.append(B)
        except:
            fail = True
        if fail:
            try:
                B = abs(float(Temp_H(T_measured, T_Block_val, B_sat, 5, nu, J=2.5)))
                B_Temp_dist.append(B)
            except:
                B_Temp_dist.append(np.nan)

    sext, doub = [], []
    for H in B_Temp_dist:
        if H > 1e-5:
            # Particle is a sextet
            cs = calculate_CS(thD, T_measured, del1)
            epsilon = (calculate_QS(thD, T_measured, QS_nought) * (3 * np.cos(theta) ** 2 -1)) / (4)
            
            sext.append(np.array(sextet_xVBF(x,
                                              CS=cs,
                                              epsilon=epsilon,
                                              H=H,
                                              sigma_ep=sigma_ep, 
                                              sigma_H=sigma_H, 
                                              intensity=intensity * f_factor_temp,
                                              counts=counts
                                             )))       
            doub.append(np.ones(len(x)))  # Doublet is zero when sextet is present
        else:
            # Particle is a doublet
            cs = calculate_CS(thD, T_measured, del1)
            qs = calculate_QS(thD, T_measured, QS_nought)
            
            doub.append(np.array(doublet_xVBF(x,
                                              CS=cs,
                                              QS=qs,
                                              sigma_QS=sigma_QS,   
                                              intensity=intensity * f_factor_temp, 
                                              counts=counts
                                             )))       
            sext.append(np.ones(len(x)))  # Sextet is zero when doublet is present

    sext_mean = np.nanmean(sext, axis=0)
    doub_mean = np.nanmean(doub, axis=0)
    
    # Combine components by multiplication (correct for transmission spectra)
    collapsed_spectrum = sext_mean * doub_mean 

    sys.stdout.flush()
    return collapsed_spectrum, sext_mean, doub_mean


def collapsed_wickman(x,
                      thD=400,
                      del1=0.6,
                      QS_nought=0.4,
                      sigma_QS=0,
                      sigma_ep=0,
                      B_sat=0,
                      sigma_H=0,
                      intensity=1,
                      counts=1 ,
                      T_Block=4,
                      sig_T_Block=70,
                      T_measured=65,
                      nu=2,
                      theta=0.78,
                      log10_f0=10,
                      A=0.1,  # Fittable param for lines 2,5
                      B=0.5,   # Fittable param for lines 3,4
                      C=0.2,
                      sigTB_res=100,
                     ):
    """
    Simulates a "collapsed" Mössbauer spectrum with dynamic relaxation broadening 
    (Wickman-like approximation).

    Like `collapsed_static`, this sums spectra over a distribution of Blocking 
    Temperatures. However, it adds a physical relaxation layer:
    
    1. For each particle, it calculates the relaxation rate (W_relax) using the 
       Arrhenius law: f = f0 * exp(-KV / kT).
    2. It passes this rate to `sextet_xVBF_relax` or `doublet_xVBF_relax`.
    3. These sub-functions apply specific broadening to the Lorentzian linewidths 
       based on the relaxation speed.

    This allows the model to fit "wings" and intermediate relaxation shapes 
    that static models cannot reproduce.

    Parameters
    ----------
    x : Velocity axis in mm/s.
    thD : Debye temperature in Kelvin (affects Isomer Shift and f-factor). Default 400.
    del1 : Isomer Shift offset parameter. Default 0.6.
    QS_nought : Intrinsic Quadrupole Splitting at T=0. Default 0.4.
    sigma_QS : Distribution width of Quadrupole Splitting (Voigt broadening).
    sigma_ep : Distribution width of the epsilon parameter (Sextet quadrupole interaction).
    B_sat : Saturation Hyperfine Field at T=0 (Tesla).
    sigma_H : Distribution width of the Hyperfine Field (Voigt broadening).
    intensity : Global intensity scaling factor.
    counts : Baseline counts (usually 1.0 for normalized transmission).
    T_Block : Mean Blocking Temperature of the particle distribution (Kelvin).
    sig_T_Block : Standard deviation of the Blocking Temperature distribution (Kelvin).
    T_measured : The experimental measurement temperature (Kelvin).
    nu : Bean-Rodbell deformation parameter.
    theta : Angle between the magnetic field and the principal axis of the EFG (in radians).
    sigTB_res : Resolution (number of steps) for the Blocking Temperature integration.
    log10_f0 : Base-10 logarithm of the attempt frequency f0 (Hz). Default 10 (i.e., 1e10 Hz).
    A : Relaxation sensitivity parameter for sextet lines 1 and 6. 
        (Controls how much the outer lines broaden vs shift).
    B : Relaxation sensitivity parameter for sextet lines 2 and 5.
    C : Relaxation sensitivity parameter for the Doublet or inner lines.
    sigTB_res : Resolution for the T_Block distribution integration.

    Returns
    -------
    tuple
        (collapsed_spectrum, sext_mean, doub_mean)
        - collapsed_spectrum: The final total calculated spectrum.
        - sext_mean: The dynamically broadened magnetic component.
        - doub_mean: The dynamically broadened paramagnetic component.
    """

    f0 = 10**log10_f0
    k_B = constants.Boltzmann  # J/K (1.3806e-23)
    CONV_FACTOR_HZ_TO_MMS = 4.605e-7 
    f_factor_temp = calculate_f_factor(T_measured, thD) 
    MOSStime = 1e-8
    E_barrier_ratio = np.log(f0 * MOSStime)
    
    # Get the distribution of particle blocking temperatures
    T_Block_dist = Temp_distribution(T_Block, sig_T_Block, res=sigTB_res)
    sext, doub = [], []

    # We loop one over the T_Block distribution. For each particle, we calculate H and W_relax. 
    for T_B_particle in T_Block_dist[1:-1]: # Using your [1:-1] slice
        
        # Calculate H-field for this particle
        try:
            H = abs(float(Temp_H(T_measured, T_B_particle, B_sat, B_sat - 5, nu, J=2.5)))
        except RuntimeError:
            try:
                H = abs(float(Temp_H(T_measured, T_B_particle, B_sat, 5, nu, J=2.5)))
            except RuntimeError:
                H = np.nan # Particle calculation failed
        
        if np.isnan(H):
            continue # Skip this failed particle

        # Calculate W_relax (dynamic broadening)
        E_B = E_barrier_ratio * k_B * T_B_particle
        Gamma_relax_Hz = f0 * np.exp(-E_B / (k_B * T_measured))
        W_relax_mms = Gamma_relax_Hz * CONV_FACTOR_HZ_TO_MMS

        # H=0 is a doublet
        if H > 1e-5:
            # SEXTET PATH
            
            # Calculate common parameters
            cs = calculate_CS(thD, T_measured, del1)
            epsilon = (calculate_QS(thD, T_measured, QS_nought) * (3 * np.cos(theta) ** 2 -1)) / (4)
                    
            # Call relax-aware sextet function
            sext.append(np.array(sextet_xVBF_relax(x,
                                 CS=cs,
                                 epsilon=epsilon,
                                 H=H,
                                 sigma_ep=sigma_ep, 
                                 sigma_H=sigma_H, 
                                 intensity=intensity * f_factor_temp,
                                 counts=counts,
                                 W_relax=W_relax_mms,
                                 A=A,
                                 B=B
                                )))
            
            doub.append(np.ones(len(x))) # Add placeholder for averaging

        else:
            # DOUBLET PATH
            cs = calculate_CS(thD, T_measured, del1)
            qs = calculate_QS(thD, T_measured, QS_nought)
            
            doub.append(np.array(doublet_xVBF_relax(x,
                                 CS=cs,
                                 QS=qs,
                                 sigma_QS=sigma_QS, 
                                 intensity=intensity * f_factor_temp, 
                                 counts=counts,
                                 W_relax=W_relax_mms, C=C
                                )))
            
            sext.append(np.ones(len(x))) # Add placeholder for averaging

    # Final Averaging
    sext_mean = np.nanmean(sext, axis=0)
    doub_mean = np.nanmean(doub, axis=0)
    
    # Combine components by multiplication (correct for transmission spectra)
    collapsed_spectrum = sext_mean * doub_mean 

    return collapsed_spectrum, sext_mean, doub_mean

def collapsed_blume(x,
                      thD=400,
                      del1=0.6,
                      QS_nought=0.4,
                      sigma_CS=0,
                      sigma_QS=0,
                      sigma_ep=0,
                      B_sat=0,
                      sigma_H=0,
                      intensity=1,
                      counts=1 ,
                      T_Block=4,
                      sig_T_Block=70,
                      T_measured=65,
                      nu=2,
                      theta=0.78,
                      log10_f0=10,
                      linewidth_L=0.135,
                      H_STEPS=15,    
                      EPS_STEPS=15,
                      sigTB_res=100,
                     ):
    """
    Generates a collapsed Mössbauer spectrum at the interface between doublet and sextet.

    Args:
        x (array): The x-axis values for the spectrum.
        thD (float, optional): Debye temperature. Defaults to 400.
        del1 (float, optional):  Parameter related to centre shift. Defaults to 0.6.
        QS_nought (float, optional):  Quadrupole splitting parameter at 0K. Defaults to 0.4.
        sigmaQS (float, optional): Sigma for quadrupole splitting distribution. Defaults to 0.
        sima_ep: sigma of the squadrupole shift distribution. Defaults to 0.
        B_sat (float, optional): Saturation magnetic field. Defaults to 0.
        sigmaH (float, optional): Sigma for hyperfine field distribution. Defaults to 0.
        intensity (int, optional): Overall intensity scaling factor. Defaults to 30000.
        counts (int, optional): Counts parameter (not directly used in the provided code). Defaults to 1.
        T_Block (float, optional): Blocking temperature. Defaults to 4.
        sig_T_Block (float, optional): Sigma for blocking temperature distribution. Defaults to 70.
        T_measured (float, optional): Measurement temperature. Defaults to 65.
        nu (int, optional):  Exponent in the temperature dependence of hyperfine field. Defaults to 2.
        theta: angle between the magnetic field and EFG
        log10_f0: the attempt frequency (frequency of attempt to flip mahnetic moment)
        T_crit: curie or neel temperature
        H_STEPS: resolution of the convolution of gaussian to H
        EPS_Steps: resolution of the convolution of the gaussian to epsilon
        sigTB_res: Number of steps of the TB sigma distribution integrated into each collapsed spectrum

    Returns:
        tuple: A tuple containing:
            - collapsed_spectrum (numpy.ndarray): The combined collapsed spectrum.
            - sext_mean (numpy.ndarray): The sextet component of the spectrum.
            - doub_mean (numpy.ndarray): The doublet component of the spectrum.
    """
    
    f0 = 10**log10_f0
    MossTimeTau = 1e-8
    ln_f0_tau = np.log(f0 * MossTimeTau)
    if ln_f0_tau < 1.0:
        ENERGY_BARRIER_RATIO = 1.0 
    else:
        ENERGY_BARRIER_RATIO = ln_f0_tau
    
    k_B = constants.Boltzmann
    f_factor_temp = calculate_f_factor(T_measured, thD)
    
    effective_area = intensity * f_factor_temp
    
    T_Block_dist = Temp_distribution(T_Block, sig_T_Block, res=sigTB_res)
    # cs_smear_kernel = create_gaussian_kernel(x, sigma_CS)
    
    dynamic_spectra = []
    sextet_shapes = []
    doublet_shapes = []
    # B_Temp_dist = []

    # We loop over each particle's T_Block ONCE
    do,se=[],[]
    for T_B_particle in T_Block_dist: 
        
        if T_B_particle < 0.01: T_B_particle = 0.0
            
        # Calculate H-Field  
        fail = False
        try:
            # Attempt 1: Guess near saturation (B_sat - 5)
            H_particle = abs(float(Temp_H(T_measured, T_B_particle, B_sat, B_sat - 5, nu, J=2.5)))
        except:
            fail = True
        
        if fail:
            try:
                # Attempt 2: Guess low (5)
                H_particle = abs(float(Temp_H(T_measured, T_B_particle, B_sat, 5, nu, J=2.5)))
            except:
                # Attempt 3: Failed to converge, assume collapsed (0.0)
                H_particle = 0.0      
            
        # Calculate parameters
        if H_particle < 0.05:
            cs = calculate_CS(thD, T_measured, del1)
            QS_val = (calculate_QS(thD, T_measured, QS_nought))
            QS_for_blume=QS_val/6 #6 is a correction for the matrix operator in the blume function
            sigQS=(sigma_QS/6)
            current_sigma_H = 0.0
        else:
            cs = calculate_CS(thD, T_measured, del1)
            QS_val = (calculate_QS(thD, T_measured, QS_nought))
            QS_for_blume= (QS_val * (3 * np.cos(theta)**2 - 1)/2)/6 #6 is a correction for the matrix operator in the blume function
            sigQS=0
            current_sigma_H = sigma_H
        
        # Calculate Gamma_relax based on this particle's T_B
        E_B = ENERGY_BARRIER_RATIO * k_B * T_B_particle 
        Gamma_relax_Hz = f0 * np.exp(-E_B / (k_B * T_measured))
        
        HZ_TO_MMS = 8.605e-8
        Gamma_relax_mms = Gamma_relax_Hz * HZ_TO_MMS

        # Cap the mm/s value, not the Hz value.
        # A cap of 10,000 mm/s is more than enough for full collapse.
        if Gamma_relax_mms > 1e4: Gamma_relax_mms = 1e4
        
        # 1. Call the Blume engine to get a raw ABSORPTION shape
        absorption_shape = extended_blume_tjon(
            x_values=x, 
            CS_mean=cs, sigma_CS=sigma_CS, # sigma_CS is handled INSIDE
            QS=QS_for_blume, sigma_QS=sigQS, 
            H=H_particle,
            sigma_H=current_sigma_H,
            Gamma_relax=Gamma_relax_mms,
            linewidth_L=linewidth_L,
            H_STEPS=H_STEPS,
            EPS_STEPS=EPS_STEPS
        )
        
        # 2. Manually apply the Beer-Lambert Law
        total_absorption = absorption_shape * effective_area
        spectrum_slice = np.exp(-total_absorption)

        # 'dynamic_spectra' is now a clean list of *only* transmission spectra
        dynamic_spectra.append(spectrum_slice)
        
        if T_B_particle<0.01:
            is_doublet = True
        else:
            is_doublet = False
            
        if is_doublet:
            doublet_shapes.append(spectrum_slice)
            do.append(1)
            sextet_shapes.append(np.ones_like(x)) # Identity for multiplication
        else:
            sextet_shapes.append(spectrum_slice)
            se.append(1)
            doublet_shapes.append(np.ones_like(x)) # Identity for multiplication
        
    # Combine components by multiplication
    collapsed_spectrum = np.nanmean(dynamic_spectra, axis=0)
    sys.stdout.flush()
    return collapsed_spectrum, np.nanmean(sextet_shapes, axis=0), np.nanmean(doublet_shapes, axis=0)

#%%

def _generate_spectrum_for_temp(temp, x, collapsed_func, **kwargs): 
    """Helper function to generate a single spectrum for a given temperature (for joblib)."""
    combined_spectrum, _, _ = collapsed_func(x, T_measured=temp, **kwargs)
    return combined_spectrum


def generate_model_spectra_matrix(temp_range, x, collapsed_func, **kwargs):
    """
    Generates matrix of MODEL Mössbauer spectra at different temperatures in parallel using joblib.
    
    WARNING: This function is programmed to use all available cores   
    """
    num_cores = os.cpu_count() # Determine number of cores

    # Use joblib's Parallel and delayed to parallelize the loop
    parallel_spectra = Parallel(n_jobs=num_cores)( # n_jobs=num_cores uses all available cores
        delayed(_generate_spectrum_for_temp)(temp, x, collapsed_func, **kwargs) # delayed wraps the function for parallel execution
        for temp in temp_range # Iterate over temperatures
    )

    return np.array(parallel_spectra) # Convert list of spectra to a matrix

def process_temperature_spectrum(temp, measured_spectrum, params, x_values, collapsed_func, param_names, fixed_kwargs):
    """
    Helper function to process a single temperature's spectrum in parallel.
    """
    try:
        param_dict = dict(zip(param_names, params))
        model_spectrum_tuple = collapsed_func(
            x_values, T_measured=temp, **param_dict, **fixed_kwargs
        )
        model_spectrum = model_spectrum_tuple[0] # Extract spectrum from tuple
        mse = np.mean((model_spectrum - measured_spectrum) ** 2)
        return mse # Return MSE for this temperature

    except Exception as e:
        print(e)
        return 0.0 # Return 0 MSE in case of error to avoid breaking optimization


def objective_function_2d_dict_input(params, measured_spectra_dict, x_values, collapsed_func, param_names, fixed_kwargs):
    mse_sum = 0.0
    results = Parallel(n_jobs=-1)(  # Use all available CPU cores (-1)
        delayed(process_temperature_spectrum)(  # Define a helper function below
            temp, measured_spectrum, params, x_values, collapsed_func, param_names, fixed_kwargs
        )
        for temp, measured_spectrum in measured_spectra_dict.items()
    )
    mse_sum = sum(results) # Sum up MSE values from parallel runs
    return mse_sum

def fit_spectra_dict_2d(measured_spectra_dict, x_values, collapsed_func, param_names, initial_params, fixed_parameters, bounds=None, method='L-BFGS-B', options={'maxiter': 1000}):
    """
    Performs 2D fitting with progress bar displayed
    """
    
    objective_function_partial = lambda params: objective_function_2d_dict_input(
        params, measured_spectra_dict, x_values, collapsed_func, param_names, fixed_parameters 
    )

    n_iterations = options.get('maxiter', 5) #this was 1000
    progress_bar = tqdm(total=n_iterations, desc="Optimisation", unit="iteration")

    def callback_function(xk):
        current_mse = objective_function_partial(xk)
        progress_bar.update(1)
        progress_bar.set_postfix({"MSE": f"{current_mse:.4f}"})

    result = minimize(objective_function_partial, initial_params, method=method, bounds=bounds, options=options, callback=callback_function)
    
    progress_bar.close()
    return result

#%%

def plot_initial_model_vs_measured(measured_spectra_dict, x_values, collapsed_func, initial_params, param_names, fixed_kwargs, temperatures_to_plot=None, plot_dir='fit_plots'):
    """
    Generates plots comparing initial model spectra to measured spectra for representative temperatures.

    Args:
        measured_spectra_dict (dict): Dictionary of measured spectra, keys are temperatures.
        x_values (np.ndarray): Velocity x-values.
        collapsed_func (function): Function to generate a single model spectrum.
        initial_params (list/np.ndarray): List of initial parameter values.
        param_names (list): List of parameter names.
        fixed_kwargs (dict): Dictionary of fixed keyword arguments for collapsed_func.
        temperatures_to_plot (list, optional): List of temperatures to plot. If None, will select min, middle, and max temperatures.
        plot_dir (str, optional): Directory to save plots. Defaults to 'fit_plots'.
    """

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)

    if temperatures_to_plot is None:
        temps = sorted(list(measured_spectra_dict.keys()))
        if len(temps) >= 3:
            temperatures_to_plot = [temps[0], temps[len(temps) // 2], temps[-1]]
        else:
            temperatures_to_plot = temps  # If less than 3 temperatures, plot all

    for temp_to_plot in temperatures_to_plot:
        if temp_to_plot not in measured_spectra_dict:
            print(f"Warning: Temperature {temp_to_plot}K not found in measured_spectra_dict. Skipping plot for this temperature.")
            continue

        measured_spectrum = measured_spectra_dict[temp_to_plot]

        # Generate initial model spectrum
        param_dict = dict(zip(param_names, initial_params))
        initial_model_spectrum_tuple = collapsed_func(
            x_values, T_measured=temp_to_plot, **param_dict, **fixed_kwargs
        )
        initial_model_spectrum = initial_model_spectrum_tuple[0] # Take the spectrum from tuple

        plt.figure(figsize=(8, 6)) 
        plt.plot(x_values, measured_spectrum, label=f'Measured ({temp_to_plot}K)', linewidth=2) # Thicker line for measured
        plt.plot(x_values, initial_model_spectrum, label=f'Initial Model ({temp_to_plot}K)', linestyle='--') # Dashed line for model

        plt.xlabel('velocity') # x-axis label
        plt.ylabel('Transmission (Background Normalized)') # y-axis label
        plt.title(f'Measured vs Initial Model Spectrum at {temp_to_plot}K\n(Background Normalized)') # Informative title
        plt.legend()
        plt.grid(False) # Keep grid off for cleaner look if you prefer
        plt.xlim(x_values.min(), x_values.max()) # Set x-axis limits to data range

        plot_filename = os.path.join(plot_dir, f'initial_model_vs_measured_{temp_to_plot}K.png')
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight') # Save plot, high DPI, tight bbox
        plt.close() # Close figure to free memory

        print(f"Saved initial model vs. measured spectrum plot for {temp_to_plot}K to: {plot_filename}")

    print(f"Initial Model vs. Measured Spectrum plots saved to directory: {plot_dir}")

 
