# -*- coding: utf-8 -*-
"""

@author: Andrew R. C. Grigg*, James M Byrne, Ruben Kretzschmar
*ETH Zurich, Department of Environmental System Science, Institute for Biogeochemistry and Pollutant Dynamics

License: Apache 2.0 (http://www.apache.org/licenses/)

Script for Mossbauer fitting and plotting.

"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D  # Imported for custom legend handles
import numpy as np
import os
import glob
import pandas as pd
from scipy.optimize import curve_fit
from nussbaum.utils import fold, s2n
from nussbaum.utils import curve as moss
import re
import traceback
from datetime import datetime
import csv


# =============================================================================
# Spectrum functions
# =============================================================================

def spec_1s2d(x, Aread, CSd, QSd, sigmaQSd, Aread2, CSd2, QSd2, sigmaQSd2, 
              Areas, CSs, epsilons, Hs, sigmaepsilons, sigmaHs, intensity=30000, counts=0):
    
    sigmaCSd, sigmaCSd2, sigmaCSs = 0, 0, 0    

    yd = moss.doublet_xVBF(x, CSd, QSd, sigmaCSd, sigmaQSd, Aread*intensity, counts)
    yd2 = moss.doublet_xVBF(x, CSd2, QSd2, sigmaCSd2, sigmaQSd2, Aread2*intensity, counts)
    ys = moss.sextet_xVBF(x, CSs, epsilons, Hs, sigmaCSs, sigmaepsilons, sigmaHs, Areas*intensity, counts)
    
    y = ((yd - counts) + (yd2 - counts) + (ys - counts) + counts)
    return y

# 2. THE FITTING FUNCTION
def fit_1s2d(data, x, Aread, CSd, QSd, sigmaQSd, Aread2, CSd2, QSd2, sigmaQSd2, 
             Areas, CSs, epsilons, Hs, sigmaepsilons, sigmaHs, intensity=30000, counts=0):
    
    intensity = 100
    
    # Initial Guess
    p0 = [Aread, CSd, QSd, sigmaQSd, 
          Aread2, CSd2, QSd2, sigmaQSd2,
          Areas, CSs, epsilons, Hs, sigmaepsilons, sigmaHs, intensity, counts]

    # Bounds
    b_lower = [0,     -0.0,    0.5, 0,  
               0,     -0.0,    0.5, 0,         
               0,     -0.0, -0.3, 0, -1,     0,       intensity-(0.2*intensity),    counts-1]
    
    b_upper = [10000,    1,    1.1, 0.8,  
               10000,    1,    2.0, 2.0,   
               10000,   1,     0,   55, 1,    Hs,       intensity+(0.2*intensity),    counts+1]
    
    b0 = (b_lower, b_upper)

    tolerance_options = {
        'xtol': 1e-10,  
        'ftol': 1e-10,  
        'gtol': 1e-8
    }
    
    # First Fit
    popt, pcov = curve_fit(spec_1s2d, x, data, p0, 
                           bounds=b0, 
                           method='trf', 
                           maxfev=10000, 
                           **tolerance_options)
    
    if popt[11] < 0.55:
        # New Bounds for second pass
        b_lower_2 = [0,     -0.0,    0.5, 0,  
                     0,     -0.0,    0.5, 0,         
                     0,     -0.0, -0.3, 0, -1,     0,       intensity-(0.2*intensity),    counts-1]
        
        b_lower_2[8] = 1e-11 

        b_upper_2 = [10000,    1,    1.1, 0.8,  
                     10000,    1,    2.0, 2.0,   
                     1e-9,   1,     0,   55, 1,    Hs,       intensity+(0.2*intensity),    counts+1]
        
        b0_2 = (b_lower_2, b_upper_2)
        
        p0_new = list(popt)
        p0_new[8] = 1e-11
        
        popt, pcov = curve_fit(spec_1s2d, x, data, p0_new, 
                               bounds=b0_2, 
                               method='trf', 
                               maxfev=10000, 
                               **tolerance_options)
        
    return popt, pcov

# =============================================================================
# GLOBAL PLOT SETTINGS
# =============================================================================
plt.rcParams.update({
    'font.size': 15,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
    'figure.titlesize': 20
})

# =============================================================================
# FILE LOADING FUNCTIONS
# =============================================================================

def load_files(file_paths, calfile_path):
    file_dict = {}
    number_pattern = r"(\d+\.\d+)" 
    velocity = None 
    
    if not file_paths:
        print("Error: No .dat files found.")
        return None, None
      
    if not os.path.exists(calfile_path):
        print(f"CRITICAL Error: Calibration file NOT FOUND at: {calfile_path}")
        return None, None
    
    try:
        cal = fold.calibrate(calfile_path)
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return None, None
        
    for filename in file_paths:
        try:
            match = re.search(number_pattern, os.path.basename(filename))
            if match:
                key = match.group(1)
                
                with open(filename, 'r') as file:
                    lines = file.readlines()
                    raw = [float(line.strip()) for line in lines]
                    x, folded = fold.fold(cal, raw)
                    
                    if velocity is None:
                        velocity = x 
                    
                    bkg = s2n.bkg(folded)
                    file_dict[key] = folded / bkg
                    
        except Exception as e:
            print(f"Error processing {os.path.basename(filename)}: {e}")
            continue

    if not file_dict:
        print("Error: No files processed.")
        return None, None
        
    return file_dict, velocity

# =============================================================================
# MAIN SCRIPT
# =============================================================================

base_path = r"T:\3_Experimental Data\2403_Hönggerberg_microsites\Mössbauer\Initial minerals\57Fe-2L-Fh"
MATRIX_DATA_FOLDER = os.path.join(base_path, "profile")
STACK_DATA_FOLDER = os.path.join(base_path, "high_SNR_spectra")
CALIBRATION_FILE_PATH = os.path.join(base_path, "250624_v12.dat") 

# --- Load Data ---
print(f"Loading matrix data...")
matrix_files = glob.glob(os.path.join(MATRIX_DATA_FOLDER, "*.dat"))
file_dict_matrix, velocity_matrix = load_files(matrix_files, CALIBRATION_FILE_PATH)

print(f"Loading stack data...")
stack_files = glob.glob(os.path.join(STACK_DATA_FOLDER, "*.dat"))
file_dict_stack, velocity_stack = load_files(stack_files, CALIBRATION_FILE_PATH)



# --- Setup Figure ---
fig = plt.figure(figsize=(14, 18)) 
gs = fig.add_gridspec(nrows=5, ncols=2, 
                      width_ratios=[1, 1], 
                      height_ratios=[0.3, 0.3, 0.8, 0.5, 0.6])
ax1 = fig.add_subplot(gs[0:4, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[2, 1])
ax6 = fig.add_subplot(gs[3, 1])
ax5 = fig.add_subplot(gs[4, :])

axes = [ax1, ax2, ax3, ax4, ax5, ax6]
labels = ['A', 'B', 'C', 'D', 'F', 'E']
for ax, label in zip(axes, labels):
    ax.text(-0.09, 1.05, label, transform=ax.transAxes,
            fontsize=20, fontweight='bold', va='top', ha='right')

# Data storage
sextet_areas, doublet_areas, T = [], [], []
excel_data_rows = []

print("Fitting and Plotting Stack (ax1)...")

if file_dict_stack and velocity_stack is not None:
    # Initial Guess for 1s2d
    p0 = [0.1, 0.49, 0.8, 0.1,   # Doublet 1
          0.1, 0.49, 0.8, 0.1,   # Doublet 2
          0.9, 0.49, -0.02, 49.0, 0.1, 0.0001, 100.0, 0.0] # Sextet + I + C
    
    try:
        sorted_keys = sorted(file_dict_stack.keys(), key=float, reverse=False)
        plot_index = 0 
        
        for key in sorted_keys:
            if abs(float(key) - 77.0) < 1.0:
                print(f"Skipping {key}K (Exclusion)")
                continue

            spectrum = file_dict_stack[key]
            
            try:
                popt, perr = fit_1s2d(
                    data=spectrum, x=velocity_stack, 
                    Aread=p0[0], CSd=p0[1], QSd=p0[2], sigmaQSd=p0[3], 
                    Aread2=p0[4], CSd2=p0[5], QSd2=p0[6], sigmaQSd2=p0[7], 
                    Areas=p0[8], CSs=p0[9], epsilons=p0[10], Hs=p0[11], 
                    sigmaepsilons=p0[12], sigmaHs=p0[13], intensity=1, counts=0)
                
                spectrum_fit = spec_1s2d(velocity_stack, *popt)
                
                # --- CALCULATE AREAS ---
                total_area = popt[0] + popt[4] + popt[8]
                frac_sextet = popt[8] / total_area
                frac_doublet_1 = popt[0] / total_area
                frac_doublet_2 = popt[4] / total_area
                frac_doublet_total = frac_doublet_1 + frac_doublet_2
                
                sextet_areas.append(frac_sextet)
                doublet_areas.append(frac_doublet_total)
                current_temp = float(key)
                T.append(current_temp)

                # --- PLOTTING AX1 ---
                offset_scale = -0.2
                current_offset = plot_index * offset_scale
                
                # Plot data and fit (Note: Labels removed from individual plots to avoid clutter)
                ax1.plot(velocity_stack, (spectrum*-1) + current_offset, color='black', marker='.', linestyle='None')
                ax1.plot(velocity_stack, (spectrum_fit*-1) + current_offset, color='r') 
                ax1.text(velocity_stack[0] + 2.8, (spectrum[0]*-1) + (current_offset-0.05), f"{key} K", 
                         ha='right', va='center', fontsize=14)
                
                plot_index += 1

                # --- EXCEL DATA STORAGE ---
                excel_data_rows.append({
                    'Temp': f"{key}K", 'temp_val': current_temp, 
                    'Phase': 'Doublet 1', 'Interp': 'Doublet 1',
                    'Rel. spec. area': frac_doublet_1,
                    'CS': popt[1], 'QS or ε': popt[2], 'σQS or σε': popt[3], 'H': 'N/A', 'σH': 'N/A'
                })
                excel_data_rows.append({
                    'Temp': f"{key}K", 'temp_val': current_temp, 
                    'Phase': 'Doublet 2', 'Interp': 'Doublet 2',
                    'Rel. spec. area': frac_doublet_2,
                    'CS': popt[5], 'QS or ε': popt[6], 'σQS or σε': popt[7], 'H': 'N/A', 'σH': 'N/A'
                })
                excel_data_rows.append({
                    'Temp': f"{key}K", 'temp_val': current_temp,
                    'Phase': 'Sextet', 'Interp': 'Sextet',
                    'Rel. spec. area': frac_sextet,
                    'CS': popt[9], 'QS or ε': popt[10], 'σQS or σε': popt[12], 'H': popt[11], 'σH': popt[13]
                })

            except RuntimeError:
                print(f"Fit failed for {key}K")
                continue

        # >>>>>> ADD LEGEND FOR PANEL A (AX1) <<<<<<
        # Create custom legend handles
        legend_elements_ax1 = [
            Line2D([0], [0], color='black', marker='.', linestyle='None', markersize=10, label='Data'),
            Line2D([0], [0], color='red', lw=2, label='Fit')
        ]
        ax1.legend(handles=legend_elements_ax1, loc='lower center', frameon=False, ncol=2)
        # >>>>>> END LEGEND ADDITION <<<<<<

        ax1.set_xlabel("Velocity (mm/s)")
        ax1.set_yticks([])
        ax1.set_ylabel("Normalised Counts (Offset)")
        ax1.set_title("Profile of high SNR spectra")
        ax1.invert_yaxis()
        
    except Exception as e:
        print("Error in fitting loop:")
        traceback.print_exc()

# --- Plotting Matrix (ax3) ---
print("Plotting Matrix (ax3)...")
def plot_matrix_on_ax(ax, file_dict, velocity, title, cmap='viridis', vmin=None, vmax=None):
    try:
        sorted_keys = sorted(file_dict.keys(), key=lambda x: float(x))
        grid_data = [file_dict[key] for key in sorted_keys]
        if not grid_data: return
        grid_data = np.array(grid_data)
        x, y = np.meshgrid(velocity, [float(k) for k in sorted_keys])
        mesh = ax.pcolormesh(x, y, grid_data, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
        ax.set_xlabel('Velocity (mm/s)')
        ax.set_ylabel('Temperature (K)')
        ax.set_title(title)
        cbar = plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Normalised Counts', size=16)
        cbar.ax.tick_params(labelsize=14)
    except Exception as e:
        print(f"Error matrix: {e}")

plot_matrix_on_ax(ax4, file_dict_matrix, velocity_matrix, "Matrix plot", vmin=0.5, vmax=1.01)

# --- Plotting Single Spectra (ax2, ax4) ---
if file_dict_matrix:
    try:
        key_75k = min(file_dict_matrix.keys(), key=lambda k: abs(float(k) - 75.0))
        ax2.plot(velocity_matrix, file_dict_matrix[key_75k], 'k.')
        ax2.set_title(f"Low SNR ({key_75k} K)")
        ax2.set_yticks([])
        ax2.set_xlabel('Velocity (mm/s)')
    except: pass
    try:
        key_5k = min(file_dict_matrix.keys(), key=lambda k: abs(float(k) - 5.0))
        ax3.plot(velocity_matrix, file_dict_matrix[key_5k], 'k.')
        ax3.set_title(f"Low SNR ({key_5k} K)")
        ax3.set_yticks([])
        ax3.set_xlabel('Velocity (mm/s)')
    except: pass

# --- Plotting the SNR curve ---

def plot_asnr_with_fit(ax, data_path, cal, title, c, T):

    # 2. File Discovery and Sorting
    files = [f for f in os.listdir(data_path) if f.endswith('.dat')]
    
    def get_timestamp(filename):
        try:
            time_str = filename.split('__')[0]
            return datetime.strptime(time_str, "%Y%m%d_%Hh%Mm%Ss")
        except:
            return None

    files.sort(key=get_timestamp)
    
    times_sec = []
    asnr_values = []
    first_ts = None
    
    print(f"Processing {len(files)} spectra...")

    for filename in files:
        ts = get_timestamp(filename)
        if ts is None: continue
        
        if first_ts is None:
            first_ts = ts
            
        # We calculate seconds for the fitting function
        elapsed_sec = (ts - first_ts).total_seconds()
        file_path = os.path.join(data_path, filename)
        
        try:
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                raw_data = [float(row[0]) for row in reader if row]
            
            if len(raw_data) >= 1023:
                data_array = np.array(raw_data[:1023]).astype(np.float64)
                _, folded_data = fold.fold(cal, data_array)
                
                # Area Normalised SNR calculation
                as2n_val = s2n.as2n(folded_data) / 1023
                
                times_sec.append(elapsed_sec)
                asnr_values.append(as2n_val)
                
        except Exception as e:
            print(f"Error in {filename}: {e}")

    if len(times_sec) < 2:
        print("Not enough data points for fitting.")
        return

    # 3. Curve Fitting
    # Replicating the logic: k, a = s2n.time_curve_params(times, values)
    try:
        ak, aa = s2n.time_curve_params(times_sec, asnr_values)
        fitted_asnr = s2n.time_curve(times_sec, ak, aa)
    except Exception as e:
        print(f"Fitting failed: {e}")
        fitted_asnr = None

    # 4. Plotting
    # plt.figure(figsize=(10, 6))

    times_min = [t / 60.0 for t in times_sec]
    
    # Raw Data Points
    ax.plot(times_min, asnr_values, 'ko', markersize=6, alpha=0.6)
    
    # Fitted Curve
    if fitted_asnr is not None:
        ax.plot(times_min, fitted_asnr, color=c, linewidth=1.5, label='{} - fitted'.format(T))

    ax.set_title(title)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("area-normlaised\n SNR")
    ax.legend()


# --- Configuration ---
DATA_DIR_5 = r"T:\3_Experimental Data\2403_Hönggerberg_microsites\Mössbauer\Initial minerals\57Fe-2L-Fh\5K_spectra-every-second"
DATA_DIR_75 = r"T:\3_Experimental Data\2403_Hönggerberg_microsites\Mössbauer\Initial minerals\57Fe-2L-Fh\75K_spectra-every-second"
CAL_FILE = r"T:\3_Experimental Data\2403_Hönggerberg_microsites\Mössbauer\Initial minerals\57Fe-2L-Fh\250624_v12.dat" 
cal = fold.calibrate(calpath=CAL_FILE, plots_on=False)

plot_asnr_with_fit(ax6, DATA_DIR_5, cal, "SNR evolution (5K)", 'red', '5 K')
plot_asnr_with_fit(ax6, DATA_DIR_75, cal, "SNR evolution (75K)", 'blue', '75 K')

# --- Plotting Logistic Fits (ax5) ---
print("Plotting Logistic Fits (ax5)...")

if len(T) > 3:
    
    def global_transition_model(x_concat, L_s, L_d, k, x0):
        # We split the concatenated x back into two parts
        half_point = len(x_concat) // 2
        t1 = x_concat[:half_point] # Time points for Sextet
        t2 = x_concat[half_point:] # Time points for Doublet
        
        # We model Sextet as a standard Logistic, but we expect 'k' to be NEGATIVE for decay.
        y_sextet = L_s / (1 + np.exp(-k * (t1 - x0)))
        
        # We model Doublet as the Inverse.
        y_doublet = L_d * (1 - (1 / (1 + np.exp(-k * (t2 - x0)))))
        
        return np.concatenate([y_sextet, y_doublet])

    # 1. Prepare Data
    T_array = np.array(T)
    sextet_array = np.array(sextet_areas)
    doublet_array = np.array(doublet_areas)
    
    # Concatenate Input (X) and Output (Y)
    X_combined = np.concatenate([T_array, T_array])
    Y_combined = np.concatenate([sextet_array, doublet_array])
    
    # 2. Initial Guess
    # L_s ~ 1, L_d ~ 1, k ~ -0.1 (decay), x0 ~ mean(T)
    mean_T = np.mean(T_array)
    p0_global = [1.0, 1.0, -0.1, mean_T] 
    
    try:
        popt_global, pcov_global = curve_fit(global_transition_model, X_combined, Y_combined, 
                                             p0=p0_global, maxfev=10000)
        
        L_s_fit, L_d_fit, k_fit, x0_fit = popt_global
        print(f"Global Fit Results: k={k_fit:.4f}, x0={x0_fit:.1f} K")

        T_fit_curve = np.linspace(min(T), max(T), 200)
        
        # Calculate manually using the fitted shared parameters
        curve_sextet = L_s_fit / (1 + np.exp(-k_fit * (T_fit_curve - x0_fit)))
        curve_doublet = L_d_fit * (1 - (1 / (1 + np.exp(-k_fit * (T_fit_curve - x0_fit)))))
        
        color_sextet = 'blue'
        color_doublet = 'red'
        color_sextet_fit = 'navy'
        color_doublet_fit = 'darkred'
        
        # Plot Curves
        ax5.plot(T_fit_curve, curve_sextet, color=color_sextet_fit, linewidth=2.0)
        ax5.plot(T_fit_curve, curve_doublet, color=color_doublet_fit, linewidth=2.0)
        
        # Plot Data Points
        ax5.plot(T, sextet_areas, marker='o', color=color_sextet, linestyle='None', 
                 markersize=12, label='Sextet Area')
        ax5.plot(T, doublet_areas, marker='o', color=color_doublet, linestyle='None', 
                 markersize=12, label='Doublet Area')

        ax5.set_xlabel("Temperature (K)")
        ax5.set_ylabel("Fractional area")
        ax5.set_title(f"Coupled Logistic Fit")
        ax5.legend()
        
    except Exception as e:
        print(f"Global fit failed: {e}")
        traceback.print_exc()

else:
    ax5.text(0.5, 0.5, "Insufficient data for logistic fit", ha='center')

fig.tight_layout(pad=1.0)

# --- SAVE FIGURE ---
try:
    save_fig_path = "mossbauer_multi_panel_plot.png"
    fig.savefig(save_fig_path, dpi=300)
    print(f"Figure saved to: {os.path.abspath(save_fig_path)}")
except Exception as e: 
    print(f"Error saving figure: {e}")

# --- SAVE EXCEL ---
if excel_data_rows:
    print("Generating Excel report...")
    df_output = pd.DataFrame(excel_data_rows)
    df_output = df_output.sort_values(by=['temp_val', 'Phase'])
    cols = ['Temp', 'Phase', 'Interp', 'Rel. spec. area', 'CS', 'QS or ε', 'σQS or σε', 'H', 'σH']
    df_final = df_output[cols]
    
    save_excel_path = "Table_S1_Fitting_Parameters.xlsx"
    try:
        df_final.to_excel(save_excel_path, index=False)
        print(f"Excel table saved successfully to: {os.path.abspath(save_excel_path)}")
    except Exception as e: 
        print(f"Error saving Excel: {e}")
else:
    print("Warning: excel_data_rows list is empty. No data to save.")

plt.show()

# =============================================================================
# SEXTET PARAMETER PLOTS (3 Cols x 2 Rows | 0 - 85 K)
# =============================================================================
if excel_data_rows:
    df_plot = pd.DataFrame(excel_data_rows)
    df_plot['temp_val'] = pd.to_numeric(df_plot['temp_val'], errors='coerce')
    
    df_sx = df_plot[(df_plot['Phase'] == 'Sextet') & 
                    (df_plot['temp_val'] >= 0) & 
                    (df_plot['temp_val'] <= 85)].sort_values('temp_val')

    if not df_sx.empty:
        fig_sx, axes_sx = plt.subplots(2, 3, figsize=(18, 10))
        fig_sx.suptitle("Sextet Parameters vs Temperature (0 - 85 K)", fontsize=22)

        # Row 1
        axes_sx[0, 0].plot(df_sx['temp_val'], df_sx['CS'], 'bo-', label='Sextet')
        axes_sx[0, 0].set_title("Isomer Shift ($\delta$)")
        axes_sx[0, 1].plot(df_sx['temp_val'], df_sx['QS or ε'], 'ro-')
        axes_sx[0, 1].set_title("Quadrupole Shift ($\epsilon$)")
        axes_sx[0, 2].plot(df_sx['temp_val'], df_sx['σQS or σε'], 'go-')
        axes_sx[0, 2].set_title("Distribution of $\epsilon$")

        # Row 2
        axes_sx[1, 0].plot(df_sx['temp_val'], df_sx['H'], 'mo-')
        axes_sx[1, 0].set_title("Hyperfine Field ($B_{hf}$)")
        axes_sx[1, 1].plot(df_sx['temp_val'], df_sx['σH'], 'ko-')
        axes_sx[1, 1].set_title("Distribution of H")
        axes_sx[1, 2].plot(df_sx['temp_val'], df_sx['Rel. spec. area'], 'co-')
        axes_sx[1, 2].set_title("Relative Spectral Area")

        for ax in axes_sx.flat:
            ax.set_xlabel("Temperature (K)")
            ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_sx.savefig("Sextet_Analysis_0-85K.png", dpi=300)

# =============================================================================
# DOUBLET PARAMETER PLOTS (3 Cols x 2 Rows | 35 - 135 K)
# =============================================================================
    df_d1 = df_plot[(df_plot['Phase'] == 'Doublet 1') & (df_plot['temp_val'] >= 35) & (df_plot['temp_val'] <= 135)].sort_values('temp_val')
    df_d2 = df_plot[(df_plot['Phase'] == 'Doublet 2') & (df_plot['temp_val'] >= 35) & (df_plot['temp_val'] <= 135)].sort_values('temp_val')

    fig_db, axes_db = plt.subplots(2, 3, figsize=(18, 10))
    fig_db.suptitle("Doublet Parameters vs Temperature (35 - 135 K)", fontsize=22)

    # Row 1: Parameters
    axes_db[0, 0].plot(df_d1['temp_val'], df_d1['CS'], 'bo-', label='D1')
    axes_db[0, 0].plot(df_d2['temp_val'], df_d2['CS'], 'ro-', label='D2')
    axes_db[0, 0].set_title("Isomer Shift ($\delta$)")
    axes_db[0, 0].legend()

    axes_db[0, 1].plot(df_d1['temp_val'], df_d1['QS or ε'], 'bo-')
    axes_db[0, 1].plot(df_d2['temp_val'], df_d2['QS or ε'], 'ro-')
    axes_db[0, 1].set_title("Quadrupole Splitting ($\Delta E_Q$)")

    axes_db[0, 2].plot(df_d1['temp_val'], df_d1['σQS or σε'], 'bo-')
    axes_db[0, 2].plot(df_d2['temp_val'], df_d2['σQS or σε'], 'ro-')
    axes_db[0, 2].set_title("Distribution of QS")

    # Row 2: Parameters & Placeholders
    # Placeholder for Hyperfine Field (White)
    axes_db[1, 0].text(0.5, 0.5, 'N/A', ha='center', va='center', color='gray')
    axes_db[1, 0].set_title("H (N/A)")
    
    # Placeholder for sigma H (White)
    axes_db[1, 1].text(0.5, 0.5, 'N/A', ha='center', va='center', color='gray')
    axes_db[1, 1].set_title("$\sigma H$ (N/A)")

    # Total Doublet Area
    df_comb = df_plot[df_plot['Phase'].str.contains('Doublet')].groupby('temp_val')['Rel. spec. area'].sum().reset_index()
    df_comb = df_comb[(df_comb['temp_val'] >= 35) & (df_comb['temp_val'] <= 135)]
    axes_db[1, 2].plot(df_comb['temp_val'], df_comb['Rel. spec. area'], 'ko-', linewidth=2)
    axes_db[1, 2].set_title("Total Relative Doublet Area")

    for ax in axes_db.flat:
        ax.set_xlabel("Temperature (K)")
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_db.savefig("Doublet_Analysis_35-135K.png", dpi=300)
    print("Doublet analysis plots saved with white placeholders.")