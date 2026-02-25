# -*- coding: utf-8 -*-
"""
Nussbaum - a python package for automated high resolution Mössbauer spectroscopy temperature profile measurements

@author: Andrew R. C. Grigg*, James M Byrne, Ruben Kretzschmar
*ETH Zurich, Department of Environmental System Science, Institute for Biogeochemistry and Pollutant Dynamics

License: Apache 2.0 (http://www.apache.org/licenses/)

The purpose of this script is to take a spectrum measured of a calibration sample, and then 
use the information from that file to help fold and fit raw data collected for a real sample.
it is an essential aspect of any Moessbauer fitting protocol and relates to "unfolded" data

The script contains two main functions which can be called in other scripts:
    'calibrate' is used to find the peaks in a calibration spectrum and relate their location to the peaks in a raw data file.
    'fold' is used to convert a raw spectrum with positive and negative velocity measurements into a single spectrum.
"""

from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d

def calibrate(calpath, plots_on=False):

    # --- Import data ---
    
    # The first dataset corresponds to the calibration itself. The raw data is a single column of data which will be either 
    # 1024, 512, or even 256 lines long. The length depends on the number of channels corresponds to an instrumental parameter 
    # which is sometimes adjusted before a measurement. This script is designed for 1023 channels and so some minor tweaks will
    # be needed for other channel numbers. 
    
    

    calibration = [] #array to be filled
    calibration = np.genfromtxt(calpath, max_rows=1024)
         
    # Create an array of the same size to plot against. This corresponds to the number of channels used for the measurement
    channels = np.arange(0, len(calibration), 1)
    
    # Combine channels and the calibration intensity into a single data frame
    d = {'channels': channels, 'intensity': calibration}
    df = pd.DataFrame(data=d)
    
    # Plot the raw data from the calibration file. This is not necessary for the final product but is a nice check.
    if plots_on==True:
        plt.plot(df.iloc[:,0],df.iloc[:,1])
        plt.title("Plot of the calibration data before fitting")
        plt.show()
    
    # The source moves back and forth and so yields a spectrum in the form of a sine curve. In most cases this sine curve
    # is too small to notice, however occasionally it is there and can have an influence so we should elimate it as best as possible.
    
    # Pick out parts from the left half of the sine wave and average
    sincorr_A = df.iloc[[210,214,218,222,284,288,292,296],1]
    mean_sincorr_A = np.mean(sincorr_A)
    
    # Pick out parts from the right half of the sine wave and average
    sincorr_B = df.iloc[[724,728,732,736,800,804,808,812],1]
    mean_sincorr_B = np.mean(sincorr_B)
    
    # Calculate mean of the sine wave. This is an approximation of the sine wave amplitude
    amplitude=(mean_sincorr_A-mean_sincorr_B)/2
      
    # There is also some background to the data because it is not at zero. 
    # We can use the start and end of the data to find an approximation of the background
    bk1=pd.DataFrame.head(df,40)
    bk2=pd.DataFrame.tail(df,40)
    
    # Combine upper and lower parts of the data frame
    bk3=pd.concat([bk1,bk2],ignore_index=True)
    
    # Calculate the background as the average of the extracted data.
    bkgd=np.mean(bk3.iloc[0:, 1])
        
    # Update the calibration intensity to remove the background
    calibration = calibration-bkgd
    
    # For the fitting we need to create set of guesses for peak positions.
    position = [144, 191, 240, 276, 324, 373, 651, 699, 746, 784, 831, 880]
    
    # The intensty of each peak can be approximated by the maximum peak intensity
    Imax = -1*np.min(calibration)
    
    # Now create a set of parameters which will be used as the intial guesses for fitting the data
    calguess = [amplitude] #the first values is the amplitude of the sine curve
    
    # run a loop to add linewidth (w), centre of each peak (CS) and intensity (I)
    for i in range(0, len(position), 1):
                   
            w = 0.1
            CS = position[i]
            I = Imax
            
            calguess.append(w)
            calguess.append(CS)
            calguess.append(I)
    
    # Here we have a function to describe each seperate peak. 
    def cal(x,*calparams):
        
        amp = calparams[0] #sine wave amplitude
        
        y = np.zeros_like(x)
        for i in range(1, len(calparams), 3):
                   
            w = calparams[i]    # linewidth
            CS = calparams[i+1] # centre of the peak
            I = calparams[i+2]  # intensity
            
            #Lorentzian profile:
            y = y - I*(1/np.pi)*(w*0.5/((x-(CS))**2+(w)**2))
        
        # in the final output, add the Sine wave background
        return y + amp*np.sin(x*np.pi/512)
    
    
    # Creata a line showing the guesses. 
    # This is not necessary other than to illustrate how close the initial guesses are to the data
    calguess_line = cal(channels, *calguess)
    
    if plots_on==True:
        plt.plot(channels,calibration)
        plt.plot(channels,calguess_line)
        plt.title("Approximate peak position guesses")
        plt.show()
    
    # Define a new function for the sine wave. This is not necessay other than to illustrate the sine wave separately from the peaks
    def sin(x,amp):
        y = amp*np.sin(x*np.pi/512)
        return y
     
    upperbounds = [-np.inf] #the existing entry is for amplitude (sine curve)
    lowerbounds = [np.inf]  #the existing entry is for amplitude (sine curve) 
    
    for i in range(0, len(position), 1):
                   
            upper_w = -np.inf   # enables the width parameter to float
            upper_CS = -np.inf  # enables the center shift (i.e. position) parameter to float
            upper_I = -np.inf  # enables intensity to float
            
            upperbounds.append(upper_w)
            upperbounds.append(upper_CS)
            upperbounds.append(upper_I)
        
            lower_w = np.inf
            lower_CS = np.inf
            lower_I = np.inf
            
            lowerbounds.append(lower_w)
            lowerbounds.append(lower_CS)
            lowerbounds.append(lower_I)
            
    # Fit the data
    popt, pcov = curve_fit(cal, channels, calibration, p0=calguess, 
                    bounds=[upperbounds,lowerbounds])
    
    # plot the results. Again this is not necessary other than for illustrative purposes.
    if plots_on==True:
        
        # Create data showing the fitted line
        fit = cal(channels, *popt)
        x=channels
        y=fit
        
        # Create a sine curve based on the fitted amplitude
        ysin=sin(channels,popt[0])
        
        #plot
        plt.plot(channels,calibration)
        plt.plot(x,ysin)
        plt.plot(x,y)
        plt.title("Result of the fit")
        plt.show()
    
    return(popt)
 
    
def fold(popt,datapath) :
    # Once we have fitted the calibrtion curve and accurately obtained the position of each individual peak, we can calibrate raw data. 
    try:
        datatofit = np.genfromtxt(datapath)
    except:
        datatofit = datapath
      
    # First extract the centre points of each peak
    pos=[]
    for i in range(2, len(popt), 3):
        pos.append(popt[i])
        
    # The calibration should have peaks at specific velocities depending on the calibration material.  
    # For a Fe(0) thin film foil, measured between -12 and 12 mm/s (this is a typical standard calibration material):
    position_Fe0_12mms = [5.3123,3.076,0.8397,-0.8397,-3.076,-5.3123,
                          -5.3123,-3.076,-0.8397,0.8397,3.076,5.3123]
    
    # since the data is roughly symmetrical and will be folded over on top of each other like closing a book, we will refer to the
    # left hand side (LHS) and right hand side (RHS) and two separate features.
     
    # Extract the LHS (i.e. take the top 6 positions) from both the fitted peak positions (as a channel number), and the actual positions (as a velocity)
    LHSx = np.array(pos[0:6]).reshape((-1, 1))
    LHSy = np.array(position_Fe0_12mms[0:6])
    
    # Fit a straight line to find an equation to convert from channel number to velocity 
    modelLHS = LinearRegression().fit(LHSx, LHSy)
    LHSc=modelLHS.intercept_
    LHSm=modelLHS.coef_
      
    # Extract the RHS (i.e. take the top 6 positions) from both the fitted peak positions (as a channel number), and the actual positions (as a velocity) 
    RHSx = np.array(pos[7:12]).reshape((-1, 1))
    RHSy = np.array(position_Fe0_12mms[7:12])
    
    # Fit a straight line to find an equation to convert from channel number to velocity 
    modelRHS = LinearRegression().fit(RHSx, RHSy)
    RHSc=modelRHS.intercept_
    RHSm=modelRHS.coef_
      
    #Once we know the coefficients of the linear equation we can convert channel to velocity for LHS
    # NOTE, this is set up for spectra with 1024 channels.
    vLHS = np.arange(0.5, 512.5, 1)
    velocityLHS = vLHS*LHSm+LHSc
    countLHS=datatofit[0:512]
    
    #Once we know the coefficients of the linear equation we can convert channel to velocity for RHS
    vRHS = np.arange(512.5, 1024.5, 1) 
    velocityRHS = vRHS*RHSm+RHSc
    countRHS=datatofit[512:len(datatofit)] 
       
    # At this point, the velocities of the LHS and RHS will be slightly different. 
    # However, we need them to be the same so that when we add them together because there is no offset. 
    # We do this by interpolating both sides of the data so that they have equivalent x-axes
        
    # create a new vector x which will be used as the new x-axis. The step size should be small but not smaller than the measured step size.
    stepsize = (24)/512
    
    # Start of the interpolation range should not exceed lowest value. 
    # End of the interpolation range should not exceed highest value. 
    # (tweak if necessary) 
    x = np.arange(-11.5, 11.5, stepsize)

    # interpolate both LHS and RHS using the new x-axis
    yLHS_interp = interp1d(velocityLHS, countLHS)
    yRHS_interp = interp1d(velocityRHS, countRHS)
    
    # create a series of intensities based on the new x-axis
    yLHS,yRHS=[],[]
    
    for i in range(0, len(x),1):
        yLHS.append(yLHS_interp(x[i]))    
        yRHS.append(yRHS_interp(x[i]))
    
    folded=[x+y for (x,y) in zip(yLHS,yRHS)]
    
    return(x, folded)