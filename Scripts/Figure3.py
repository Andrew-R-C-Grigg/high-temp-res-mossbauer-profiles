# -*- coding: utf-8 -*-
"""
@author: Andrew R. C. Grigg*, James M Byrne, Ruben Kretzschmar
*ETH Zurich, Department of Environmental System Science, Institute for Biogeochemistry and Pollutant Dynamics

License: Apache 2.0 (http://www.apache.org/licenses/)
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import re
import glob
from nussbaum.utils import fold, s2n

# =============================================================================
# SECTION 1: FILE LOADING
# =============================================================================

def load_files(file_paths, calfile_path):
    """
    Loads selected .dat files and a calibration file, then processes them.
    """
    file_dict = {}
    number_pattern = r"(\d+\.\d+)"  # Extracts temperature (e.g., "105.0")
    velocity = None 
    
    if not file_paths:
        print(f"Error: No .dat files found in the specified path.")
        return None, None
      
    if not os.path.exists(calfile_path):
        print(f"CRITICAL Error: Calibration file NOT FOUND at: {calfile_path}")
        return None, None
    
    try:
        cal = fold.calibrate(calfile_path)
    except Exception as e:
        print(f"Error: Failed to load calibration file: {e}")
        return None, None
        
    for filename in file_paths:
        try:
            match = re.search(number_pattern, os.path.basename(filename))
            if match:
                key = match.group(1)  # Temperature as string
                
                with open(filename, 'r') as file:
                    lines = file.readlines()
                    raw = [float(line.strip()) for line in lines]
                    x, folded = fold.fold(cal, raw)
                    
                    if velocity is None:
                        velocity = x
                    
                    bkg = s2n.bkg(folded)
                    file_dict[key] = folded / bkg
                    
        except Exception as e:
            print(f"Error: Failed to process file {os.path.basename(filename)}: {e}")
            continue

    if not file_dict:
        print("Error: No files were successfully processed.")
        return None, None
        
    return file_dict, velocity

# =============================================================================
# SECTION 2: PATH SETUP & EXECUTION
# =============================================================================

base_path = r"..\Data\Fh-Gt_mix\"
MATRIX_DATA_FOLDER = os.path.join(base_path, "profile")
CALIBRATION_FILE_PATH = os.path.join(base_path, "251104_v12.dat") 

# --- Load Data ---
print(f"Loading matrix data from: {MATRIX_DATA_FOLDER}")
matrix_files = glob.glob(os.path.join(MATRIX_DATA_FOLDER, "*.dat"))
file_dict_matrix, velocity_matrix = load_files(matrix_files, CALIBRATION_FILE_PATH)

# =============================================================================
# SECTION 3: PLOTTING
# =============================================================================

# Setup Figure: 
# Changed to layout='constrained' to automatically fix overlapping titles and colorbars.
# Increased width slightly (14, 8) to give the labels more room.
fig = plt.figure(figsize=(14, 8), layout='constrained')

# Removed manual wspace/hspace; constrained_layout handles this now.
gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[1.5, 1])

ax_matrix = fig.add_subplot(gs[:, 0])  # Left column, full height
ax_75k = fig.add_subplot(gs[0, 1])     # Right column, top
ax_5k = fig.add_subplot(gs[1, 1])      # Right column, bottom

# --- Helper Function for Matrix Plot ---
def plot_matrix_on_ax(ax, file_dict, velocity, title, cmap='viridis', vmin=None, vmax=None):
    try:
        # Sort keys by temperature (as float)
        sorted_keys = sorted(file_dict.keys(), key=lambda x: float(x))
        sorted_keys_float = [float(key) for key in sorted_keys]
        
        grid_data = []
        for key in sorted_keys:
            data_values = [float(value) for value in file_dict[key]]
            grid_data.append(data_values)
            
        grid_data = np.array(grid_data)
        
        if grid_data.size == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            return
            
        x, y = np.meshgrid(velocity, sorted_keys_float)
        
        # Plotting
        mesh = ax.pcolormesh(
            x, y, grid_data,
            cmap=cmap,
            shading='auto',
            vmin=vmin,
            vmax=vmax
        )
        
        ax.set_xlabel('Velocity (mm/s)')
        ax.set_ylabel('Temperature (K)')
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add colorbar
        # constrained_layout will automatically create space for this
        plt.colorbar(mesh, ax=ax, label='Normalised Counts', fraction=0.046, pad=0.04)
    
    except Exception as e:
        ax.text(0.5, 0.5, f"Error:\n{e}", ha='center', va='center', color='red')
        print(f"Error in plot_matrix_on_ax: {e}")

# --- 1. Plot Matrix (Left Panel) ---
print("Plotting Matrix...")
if file_dict_matrix and velocity_matrix is not None:
    plot_matrix_on_ax(
        ax_matrix, 
        file_dict_matrix, 
        velocity_matrix, 
        "Temperature Profile (Matrix)", 
        cmap='viridis', 
        vmin=0.75, 
        vmax=1.01
    )
else:
    ax_matrix.text(0.5, 0.5, "Data not loaded", ha='center')

# --- 2. Plot 75 K Spectrum (Top Right) ---
print("Plotting 75 K Spectrum...")
if file_dict_matrix and velocity_matrix is not None:
    try:
        # Find key closest to 75.0
        key_75k = min(file_dict_matrix.keys(), key=lambda k: abs(float(k) - 75.0))
        
        if abs(float(key_75k) - 75.0) > 2.0: # Tolerance check
            ax_75k.text(0.5, 0.5, "75 K spectrum missing", ha='center', color='red')
        else:
            ax_75k.plot(velocity_matrix, file_dict_matrix[key_75k], color='black', linestyle=' ', marker='.', markersize=3)
            ax_75k.set_xlabel("Velocity (mm/s)")
            ax_75k.set_ylabel("Normalised Counts")
            ax_75k.set_title(f"Spectrum at {key_75k} K")
            ax_75k.grid(True, linestyle='--', alpha=0.5)
    except Exception as e:
        print(f"Error plotting 75K: {e}")

# --- 3. Plot 5 K Spectrum (Bottom Right) ---
print("Plotting 5 K Spectrum...")
if file_dict_matrix and velocity_matrix is not None:
    try:
        # Find key closest to 5.0
        key_5k = min(file_dict_matrix.keys(), key=lambda k: abs(float(k) - 5.0))
        
        if abs(float(key_5k) - 5.0) > 2.0: # Tolerance check
            ax_5k.text(0.5, 0.5, "5 K spectrum missing", ha='center', color='red')
        else:
            ax_5k.plot(velocity_matrix, file_dict_matrix[key_5k], color='black', linestyle=' ', marker='.', markersize=3)
            ax_5k.set_xlabel("Velocity (mm/s)")
            ax_5k.set_ylabel("Normalised Counts")
            ax_5k.set_title(f"Spectrum at {key_5k} K")
            ax_5k.grid(True, linestyle='--', alpha=0.5)
    except Exception as e:
        print(f"Error plotting 5K: {e}")

# --- Save ---
try:
    save_name = "mossbauer_matrix_and_cuts.png"
    print(f"Saving figure to {save_name}...")
    fig.savefig(save_name, dpi=300, bbox_inches='tight')
    print("Done.")
except Exception as e:
    print(f"Error saving figure: {e}")

plt.show()