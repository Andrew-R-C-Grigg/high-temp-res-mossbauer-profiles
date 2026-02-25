# -*- coding: utf-8 -*-
"""
@author: Andrew R. C. Grigg*, James M Byrne, Ruben Kretzschmar
*ETH Zurich, Department of Environmental System Science, Institute for Biogeochemistry and Pollutant Dynamics

License: Apache 2.0 (http://www.apache.org/licenses/)
"""

import matplotlib.pyplot as plt
import re
import os
import glob
import numpy as np
from nussbaum.utils import fold, s2n


#LOAD FILES
def load_files(directory, pattern, calfile):
    
    #perform calibration
    cal=fold.calibrate(calfile, plots_on=False)   
    
    # Dictionary to store file contents with the unique part of file names as the key
    file_dict = {}
    number_pattern = r"(\d+\.\d+)"
    
    # Loop through all files in the specified directory
    for filename in os.listdir(directory):
        # Check if the file is a .dat file and matches the pattern
        if filename.endswith(".dat") and re.search(pattern, filename):
            # Extract the numeric part using regex
            match = re.search(number_pattern, filename)
            if match:
            
                # Use the matched number (the first group in the regex match)
                key = match.group(1)
                
                # Open and read the contents of the file line-by-line
                with open(os.path.join(directory, filename), 'r') as file:
                    # Read each line from the file
                    lines = file.readlines()
                    
                    # Process lines (assuming the numbers are newline-delimited)
                    raw=[float(line.strip()) for line in lines]
                    x,folded=fold.fold(cal,raw)
                    
                    #find background value for velocity domain spectrum to normalise each spectrum to its background count intensity
                    bkg=s2n.bkg(folded)
                    
                    file_dict[key] = folded/bkg                
    
    print('calibration completed')                
    return file_dict, x


temp_to_plot='55'

fig, ax = plt.subplots(figsize=(6,10))
for e,(folder,title) in enumerate(zip(
        [
    "2Lc_Fh",
    "2L_Fh",
    "6L_Fh"
            ],
    [
    "2Lc Ferrihydrite",
    "2L Ferrihydrite",
    "6L Ferrihydrite"
         ])):
    
    # Define paths
    directory_path=f"..\Data\{folder}\high_SNR_spectra"
    # pattern = r"{}\.0K_+\d+\.\d+h+\d+\.\d+m".format(temp_to_plot)
    pattern = r"{}.0K".format(temp_to_plot)   
    print(pattern)
    calfile = glob.glob(f"..\Data\{folder}\*_v12.dat")
    
    spectra, velocity= load_files(directory_path, pattern, calfile[0])
    
    bkg=s2n.bkg(spectra[temp_to_plot+'.0'])
    area=np.trapz(y=bkg-spectra[temp_to_plot+'.0'], x=velocity)  
    print(area)
    data_norm=[(y/area) for y in spectra[temp_to_plot+'.0']]
    bkg_norm=s2n.bkg(data_norm)
    plot_data=[(e*0.2)+y+(bkg-bkg_norm) for y in data_norm]

    
    ax.plot(velocity,plot_data)
    ax.text(-11,1+0.2*e+0.02,title, fontsize='large')
    
    ax.set_yticks([])

ax.set_xlabel('Velocity (mm/s)', fontsize=20)

ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=18)
    
fig.suptitle("Comparison of spectra measured at 55K", fontsize=16, x=0.5)
fig.savefig("high_SNR_spectra_at_{}K.png".format(temp_to_plot), dpi=300, bbox_inches="tight")




