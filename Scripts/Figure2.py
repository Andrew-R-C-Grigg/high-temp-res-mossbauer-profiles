# -*- coding: utf-8 -*-
"""

@author: Andrew R. C. Grigg*, James M Byrne, Ruben Kretzschmar
*ETH Zurich, Department of Environmental System Science, Institute for Biogeochemistry and Pollutant Dynamics

License: Apache 2.0 (http://www.apache.org/licenses/)
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
import numpy as np
import csv
import os
import re
from nussbaum.utils import fold, s2n
import glob


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


#PLOT FILES
def plot_data_with_velocity(file_dict, velocity, title, ax=None, global_vmin=0.5, global_vmax=1):
    # Create a figure and axis for plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    # Sort the keys numerically (as float values)
    sorted_keys = sorted(file_dict.keys(), key=lambda x: float(x))
    sorted_keys_float = [float(key) for key in sorted_keys]  # Convert keys to floats for plotting

    # Flatten all data into a single list to calculate universal vmin and vmax
    all_data_values = [float(value) for key in sorted_keys for value in file_dict[key]]
    vmin = min(all_data_values)
    global_vmin = np.floor(vmin * 10) / 10
    # global_vmax = max(all_data_values)
    # global_vmin = 0.75
    # global_vmax = 1.03

    # Loop over the sorted dictionary keys and their corresponding data
    grid_data = []
    for key in sorted_keys:

        # Convert the list of strings to floats (assuming they are numeric strings)
        data_values = [float(value) if float(value) >= 0.0 else np.nan for value in file_dict[key]]
        grid_data.append(data_values)
    
    
    # Convert grid_data to a NumPy array for compatibility with pcolormesh
    grid_data = np.array(grid_data)

    # Create a meshgrid for the x (velocity) and y (keys) axes
    x, y = np.meshgrid(velocity, sorted_keys_float)

    # Use pcolormesh to create the grid plot
    mesh = ax.pcolormesh(
        x, y, grid_data, 
        cmap='viridis', 
        shading='auto', 
        vmin=global_vmin, 
        vmax=global_vmax
    )

    # Add labels and title
    ax.set_xlabel('Velocity (mm/s)', fontsize=20)
    ax.set_ylabel('Temperature (K)', fontsize=20)
    ax.set_title('{} Ferrihydrite'.format(title), fontsize=25)
    
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)

    # Return the figure for saving or further manipulation
    return mesh,global_vmin,global_vmax




#execute
spectra_dicts={}
figcomb, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(20, 6))

# Define the universal vmin and vmax for consistent color scaling
vmin = 0.5
vmax = 1.02

for folder,ax,title in zip(
        [
            "2L_Fh",
         "2Lc_Fh",
          "6L_Fh"
         ],
        [
            ax1,
            ax2,
            ax3
            ],
        [
            "2L",
         "2Lc",
          "6L"
         ]
        ):
    
    # Define paths
    directory_path=f"..\Data\{folder}\profile"
    pattern = r"\d+\.\d+K"  
    calfile = glob.glob(f"..\{folder}\*_v12.dat")
    
    # Load the data
    spectra, velocity= load_files(directory_path, pattern, calfile[0])
    
    # Plot data directly on the given axis
    mesh,v_min,v_max=plot_data_with_velocity(spectra, velocity, title, ax=ax, 
                                 global_vmin=vmin, 
                                 global_vmax=vmax)
    
    spectra_dicts[title]=spectra
    
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.8)
    cbar_ticks = np.arange(v_min, v_max + 0.01, 0.1)
    cbar = figcomb.colorbar(mesh, cax=cax, orientation="horizontal", ticks=cbar_ticks)
    cbar.set_label("Counts Normalised to Background", fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    # cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

# Add a shared colorbar
# Use the last mesh returned from `plot_data_with_velocity
# cbar = figcomb.colorbar(mesh, ax=[ax1, ax2, ax3], orientation="vertical", shrink=0.8)
# cbar.set_label("Counts Normalised to Background")
    
# Add a shared title for the entire figure
figcomb.suptitle("High-resolution temeprature plots for $^{57}$Fe-ferrihydrite", fontsize=25, x=0.5, y=1.1)

# Save the combined figure
figcomb.savefig("temperature-velocity_matrix.png", dpi=300, bbox_inches="tight")
plt.show()


    