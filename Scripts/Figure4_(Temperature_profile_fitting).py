# -*- coding: utf-8 -*-
"""

Nussbaum - a python package for automated high-resolution Mössbauer spectroscopy temperature profile measurements

@author: Andrew R. C. Grigg*, James M Byrne, Ruben Kretzschmar
*ETH Zurich, Department of Environmental System Science, Institute for Biogeochemistry and Pollutant Dynamics

License: Apache 2.0 (http://www.apache.org/licenses/)

This code is designed to fit Mössbauer temperature profiles at all temperatures simultaneously.

This is a steering file, which uses the model functions that are defined in the utils.curve file.

"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import re
from nussbaum.utils import fold as fold
from nussbaum.utils import s2n as s2n
from nussbaum.utils import curve as moss
import time
from datetime import datetime
import matplotlib.animation as animation
import pandas as pd

#define the directory containing the data to be fitted
file_home= r"..\Data\2L-Fh\profile\"

#LOAD FILES
def load_files(directory, pattern, calfile):
    # Dictionary to store file contents with the unique part of file names as the key
    file_dict = {}
    number_pattern = r"(\d+\.\d+)" #this is designed to import data with names from the automated saving function
    
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
                    print(filename)
                    x,folded=fold.fold(cal,raw)
                    
                    #find background value for velocity domain spectrum to normalise each spectrum to its background count intensity
                    bkg=s2n.bkg(folded)
                    
                    norm_to_one = folded/bkg
                                        
                    file_dict[key] = folded/bkg
    
    print("loaded all files")
                     
    return file_dict, x

# define data files
directory_path=file_home+"\profile"
pattern = r"\d+\.\d+K"  # Pattern to match files like 77.0K.dat or 101.0K.dat
calfile=file_home+'/'+'250624_v12.dat'
cal=fold.calibrate(calfile)

highSNR_directory_path=file_home+"\high_SNR_spectra"
highSNR_pattern = r"\d+\.\d+K_\d+\.\d+h"  # Pattern to match files like 77.0K.dat or 101.0K.dat

# execute
start_time = time.perf_counter()
spectra_dict, velocity = load_files(directory_path, pattern, cal)
highSNR_dict,velocity = load_files(highSNR_directory_path, highSNR_pattern, cal)


#%%

#Velocity-temperature matrix fitting

if __name__ == '__main__':

    print(f"Number of CPU cores reported by os.cpu_count(): {os.cpu_count()}")
    end_time = time.perf_counter()
    loading_time = end_time - start_time
    print(f"Time taken to load data: {loading_time:.4f} seconds")
    
    x_values =velocity

    highSNR_spectra_dict_numeric_keys = {}
    for temp_str, spectrum_data in highSNR_dict.items():
        try:
            temp_numeric = float(temp_str)  # Convert string key to float
            highSNR_spectra_dict_numeric_keys[temp_numeric] = spectrum_data
        except ValueError:
            print(f"Warning: Could not convert temperature key '{temp_str}' to a number. Skipping this spectrum.")
            continue
    highSNR_spectra_dict = highSNR_spectra_dict_numeric_keys # Use the dict with numeric keys

    # Perform a pre-fit of the Debye temperature
    fixed_thD_val, _ = moss.prefit_debye_temp(highSNR_spectra_dict, x_values)

    areas = []
    temps_sorted = sorted(highSNR_spectra_dict.keys())
    for t in temps_sorted:
        # Trapz integration
        abs_spec = 1.0 - highSNR_spectra_dict[t]
        areas.append(np.trapz(abs_spec, x_values))
    
    plt.figure(figsize=(6,4))
    plt.scatter(temps_sorted, areas, label='Measured Area (High SNR)')
    plt.title(f"Area Trend (Calculated thD: {fixed_thD_val:.1f} K)")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Integrated Area")
    plt.grid(True)
    plt.show()

    measured_spectra_dict_numeric_keys = {}
    for temp_str, spectrum_data in spectra_dict.items():
        try:
            temp_numeric = float(temp_str)  # Convert string key to float
            measured_spectra_dict_numeric_keys[temp_numeric] = spectrum_data
        except ValueError:
            print(f"Warning: Could not convert temperature key '{temp_str}' to a number. Skipping this spectrum.")
            continue
    measured_spectra_dict = measured_spectra_dict_numeric_keys # Use the dict with numeric keys
    temperatures = np.array(sorted(measured_spectra_dict.keys())) # Get ordered temperatures from measured_spectra_dict keys


# MODEL NUMBER 1 - Static xVBF

    # Define parameters to optimize, initial guess, and fixed parameters
    params_static = [
        'thD', 
        'del1', 
        'QS_nought', 
        'sigma_QS', 
        'sigma_ep', 
        'B_sat', 
        'sigma_H', 
        'T_Block', 
        'sig_T_Block', 
        'nu' ,
        'theta', 
        'intensity', 

    ]
    initial_guess_static = [
        fixed_thD_val, 0.55, 0.9, 0.31, 0.2, 50, 2.6, 65, 10, 2, 0.7845, 4,  # Initial guess for counts (e.g., 1.0)
    ]
    parameter_bounds_static = [
        (fixed_thD_val*0.9, fixed_thD_val*2), (0.45, 0.6), (0.6, 1.2), (0.2, 0.6), (0,0.5), (45, 60), (1, 4), (58, 78), (1, 30), (0.1, 3.0), (0,1.5708), (3,7.5), 
    ]
    
    fixed_parameters_static = {'counts':1.0,
                               'sigTB_res':100
    }
    
    # Plots of initial guess parametres
    moss.plot_initial_model_vs_measured(
        measured_spectra_dict, x_values, moss.collapsed_static, 
        initial_guess_static, params_static, fixed_parameters_static, # Use initial_params and fixed_parameters
        plot_dir="file_home"
    )

    # Perform 2D fitting
    print("starting optimisation of static model")
    print(f"[{datetime.now().strftime('%H:%M:%S')}]")
    sys.stdout.flush() # Ensure error log is flushed
    optimisation_result_static = moss.fit_spectra_dict_2d(
        measured_spectra_dict, x_values, moss.collapsed_static, params_static,
        initial_guess_static, fixed_parameters_static, bounds=parameter_bounds_static, 
        method='Nelder-Mead', options={'maxiter': 2500, 'fatol':1e-5, 'xatol':1e-5}
    )

    print("static fit complete")
    print(f"[{datetime.now().strftime('%H:%M:%S')}]")
    print("Optimisation Result:")
    print(optimisation_result_static)

    optimised_params_static = optimisation_result_static.x
    print("Optimised Parameters:")
    for i, name in enumerate(params_static):
        print(f"{name}: {optimised_params_static[i]:.4f}")

    print("\n--- Running L-BFGS-B to calculate uncertainties ---")
    lbfgs_guess = optimised_params_static 
    
    # Run the L-BFGS-B fit.
    optimization_result_lbfgs = moss.fit_spectra_dict_2d(
        measured_spectra_dict, x_values, moss.collapsed_static, params_static,
        lbfgs_guess,  # <-- Use the result from Nelder-Mead
        fixed_parameters_static, 
        bounds=parameter_bounds_static, 
        method='L-BFGS-B',  
        options={'maxiter': 100, 'ftol': 1e-9} # Needs few iterations
    )

    print("L-BFGS-B fit complete.")

    # - CALCULATE AND PRINT UNCERTAINTIES -
    try:
        P = len(params_static)
        
        # Get the inverse Hessian matrix from the result. This is the (unscaled) covariance matrix
        inverse_hessian = optimization_result_lbfgs.hess_inv @ np.identity(P)
        
        # Get the final MSE (the "fun" value) and degrees of freedom
        final_mse = optimization_result_lbfgs.fun
        
        # N = total data points, P = parameters
        # Total points = (points per spectrum) * (number of spectra)
        num_spectra = len(measured_spectra_dict)
        points_per_spectrum = len(x_values)
        N = num_spectra * points_per_spectrum
        degrees_of_freedom = N - P
        
        # Scale the matrix by the MSE. This is the "Covariance Matrix"
        covariance_matrix = inverse_hessian * final_mse / degrees_of_freedom
        
        #  The uncertainty (Standard Error) is the sqrt of the diagonal
        uncertainties = np.sqrt(np.diag(covariance_matrix))

        print("\n--- Parameter Uncertainties (Standard Errors) ---")
        print(f"{'Parameter':<12} | {'Value':<12} | {'Uncertainty':<12}")
        print("-" * 38)
        
        for i, name in enumerate(params_static):
            val = optimised_params_static[i]
            err = uncertainties[i]
            print(f"{name:<12} | {val:<12.4f} | {err:<12.4f}")

    except Exception as e:
        print(f"\nCould not calculate uncertainties: {e}")
        print("This often happens if the 'L-BFGS-B' fit fails or the Hessian is not available.")

    # Generate spectra with optimized parameters
    optimised_spectra_matrix_static = moss.generate_model_spectra_matrix(
        temperatures, x_values, moss.collapsed_static, **fixed_parameters_static,
        thD=optimised_params_static[0],
        del1=optimised_params_static[1],
        QS_nought=optimised_params_static[2],
        sigma_QS=optimised_params_static[3],
        sigma_ep=optimised_params_static[4],
        B_sat=optimised_params_static[5],
        sigma_H=optimised_params_static[6], 
        T_Block=optimised_params_static[7],
        sig_T_Block=optimised_params_static[8],
        nu=optimised_params_static[9], 
        theta=optimised_params_static[10],
        intensity=optimised_params_static[11],
    )
  
    print( "Starting dynamic fit")    
      
    static_results = dict(zip(params_static, optimisation_result_static.x))

  #%%
    
# MODEL NUMBER 2 - Static xVBF with Wickman broadening
    
    params_dynamic = params_static + ['log10_f0', 'A', 'B', 'C']
    
    initial_guess_dynamic = list(optimised_params_static) 
    initial_guess_dynamic.append(10.0)  # Initial guess for log10_f0
    initial_guess_dynamic.append(0.01)   # Initial guess for A
    initial_guess_dynamic.append(0.05)   # Initial guess for B
    initial_guess_dynamic.append(0.02)   # Initial guess for C
    
    tight_bounds_dynamic = []
    TOLERANCE = 0.25  # 25% wiggle room 
    for i, (orig_lower, orig_upper) in enumerate(parameter_bounds_static):
        val = optimised_params_static[i]
        orig_lower, orig_upper = parameter_bounds_static[i]
        
        if name == 'sig_T_Block':
            # Give it wide-open bounds to *correct* the static fit
            tight_bounds_dynamic.append((1.0, 30.0)) 
        else:
            # Use standard +/- tolerance for all other static params
            delta = abs(val * TOLERANCE)
            if delta < 1e-3: delta = 1e-2 
            new_lower = val - delta
            new_upper = val + delta
            final_lower = max(orig_lower, new_lower)
            final_upper = min(orig_upper, new_upper)
            tight_bounds_dynamic.append((final_lower, final_upper))

    # 4. Append the wide bounds for the new dynamic parameters
    #    (These have not been fitted yet, so they need space to move)
    tight_bounds_dynamic.append((8.0, 12.5)) # Bounds for log10_f0
    tight_bounds_dynamic.append((0.0, 0.5))  # Bounds for A
    tight_bounds_dynamic.append((0.0, 0.5))  # Bounds for B
    tight_bounds_dynamic.append((0.0, 0.3))  # Bounds for C
    
    fixed_parameters_dynamic = fixed_parameters_static.copy()

  
    print("Starting optimisation of dynamic 'Wickman' model...")
    print(f"[{datetime.now().strftime('%H:%M:%S')}]")
    sys.stdout.flush() 
    optimization_result_dynamic = moss.fit_spectra_dict_2d(
        measured_spectra_dict, x_values, moss.collapsed_wickman, # <-- Use the Wickman model
        params_dynamic,
        initial_guess_dynamic, 
        fixed_parameters_dynamic, 
        bounds=tight_bounds_dynamic, 
        method='Nelder-Mead', 
        options={'maxiter': 2500, 'fatol':1e-6, 'xatol':1e-6} 
    )
    
    print("Dynamic 'Wickman' fit complete")
    print(f"[{datetime.now().strftime('%H:%M:%S')}]")
    print("Optimisation Result:")
    print(optimization_result_dynamic)

    optimised_params_dynamic = optimization_result_dynamic.x
    print("Optimised Dynamic Parameters:")
    for i, name in enumerate(params_dynamic):
        print(f"{name}: {optimised_params_dynamic[i]:.4f}")

    # Run L-BFGS-B (Dynamic) to calculate uncertainties
    print("\n--- Running L-BFGS-B (Dynamic) to calculate uncertainties ---")
    lbfgs_guess_dynamic = optimised_params_dynamic 
    
    optimization_result_lbfgs_dynamic = moss.fit_spectra_dict_2d(
        measured_spectra_dict, x_values, moss.collapsed_wickman, # <-- Use the Wickman model
        params_dynamic,
        lbfgs_guess_dynamic, 
        fixed_parameters_dynamic, 
        bounds=tight_bounds_dynamic, 
        method='L-BFGS-B', 
        options={'maxiter': 10, 'ftol': 1e-7} 
    )

    print("L-BFGS-B (Dynamic) fit complete.")
    
    try:
        P_dynamic = len(params_dynamic)
        
        inverse_hessian_dynamic = optimization_result_lbfgs_dynamic.hess_inv @ np.identity(P_dynamic)
        
        final_mse_dynamic = optimization_result_lbfgs_dynamic.fun
        
        num_spectra = len(measured_spectra_dict)
        points_per_spectrum = len(x_values)
        N = num_spectra * points_per_spectrum
        degrees_of_freedom_dynamic = N - P_dynamic
        
        covariance_matrix_dynamic = inverse_hessian_dynamic * final_mse_dynamic / degrees_of_freedom_dynamic
        
        uncertainties_dynamic = np.sqrt(np.diag(covariance_matrix_dynamic))

        print("\n--- Dynamic Parameter Uncertainties (Standard Errors) ---")
        print(f"{'Parameter':<12} | {'Value':<12} | {'Uncertainty':<12}")
        print("-" * 38)
        
        for i, name in enumerate(params_dynamic):
                val = optimised_params_dynamic[i]
                err = uncertainties_dynamic[i]
                print(f"{name:<12} | {val:<12.4f} | {err:<12.4f}")
    
    except Exception as e:
        print(f"\nCould not calculate dynamic uncertainties: {e}")
    
    print("\nGenerating final dynamic model spectra...")
    optimised_spectra_matrix_dynamic = moss.generate_model_spectra_matrix(
        temperatures, x_values, moss.collapsed_wickman, # <-- Use the Wickman model
        **fixed_parameters_dynamic,
        thD=optimised_params_dynamic[0],
        del1=optimised_params_dynamic[1],
        QS_nought=optimised_params_dynamic[2],
        sigma_QS=optimised_params_dynamic[3],
        sigma_ep=optimised_params_dynamic[4],
        B_sat=optimised_params_dynamic[5],
        sigma_H=optimised_params_dynamic[6], 
        T_Block=optimised_params_dynamic[7],
        sig_T_Block=optimised_params_dynamic[8],
        nu=optimised_params_dynamic[9], 
        theta=optimised_params_dynamic[10],
        intensity=optimised_params_dynamic[11],
        
        # --- Add new dynamic parameters ---
        log10_f0=optimised_params_dynamic[12],
        A=optimised_params_dynamic[13],
        B=optimised_params_dynamic[14],
        C=optimised_params_dynamic[15],
    )
    
    print("\nDynamic 'Wickman' model fitting complete.")
    
  #%%
  
# MODEL NUMBER 3 - Blume Tjon
  
    print("STARTING DYNAMIC 'BLUME-TJON' MODEL FIT")
  
    params_blume = params_static + ['log10_f0', 
                                    ]
    
    # Use the static results as the initial guess
    initial_guess_blume = list(optimised_params_static)
    initial_guess_blume.append(10.0)  # Initial guess for log10_f0
    theta_index = params_static.index('theta')
    initial_guess_blume[theta_index] = 1.0
    
    # Create tight bounds based on the static fit
    tight_bounds_blume = []
    # TOLERANCE is already defined above
    for i, (orig_lower, orig_upper) in enumerate(parameter_bounds_static):
        val = optimised_params_static[i]
        orig_lower, orig_upper = parameter_bounds_static[i]
        
        if name == 'sig_T_Block':
            tight_bounds_dynamic.append((1.0, 30.0)) 
        if name == 'thD':
            tight_bounds_dynamic.append((100, 1000)) 
        else:
            # Use standard +/- tolerance for all other static params
            delta = abs(val * TOLERANCE)
            if delta < 1e-3: delta = 1e-2 
            new_lower = val - delta
            new_upper = val + delta
            final_lower = max(orig_lower, new_lower)
            final_upper = min(orig_upper, new_upper)
            tight_bounds_blume.append((final_lower, final_upper))
    
    # Add bounds for the new 'Blume' parameters
    tight_bounds_blume.append((8.0, 13.0)) # Bounds for log10_f0
    
    # Define fixed parameters
    fixed_parameters_blume = {
        'counts': 1.0,
        'H_STEPS': 1,  # if 1, the shortcut smearing will be implemented
        'EPS_STEPS': 1, # if 1, the shortcut smearing will be implemented
        'sigTB_res':20,
        'linewidth_L':0.135
    }
    
    if fixed_parameters_blume['EPS_STEPS'] == 1: 
        sigma_ep = 0.0
    
    print("Starting optimisation of dynamic 'Blume-Tjon' model...")
    print(f"[{datetime.now().strftime('%H:%M:%S')}]")
    sys.stdout.flush() 
    optimisation_result_blume = moss.fit_spectra_dict_2d(
        measured_spectra_dict, x_values, 
        moss.collapsed_blume, # <-- Use the Blume model
        params_blume,
        initial_guess_blume, 
        fixed_parameters_blume, 
        bounds=tight_bounds_blume, 
        method='Nelder-Mead', 
        options={'maxiter': 2500, 'fatol':1e-5, 'xatol':1e-5} 
    )

    print("Dynamic 'Blume-Tjon' fit complete")
    print(f"[{datetime.now().strftime('%H:%M:%S')}]")
    optimised_params_blume = optimisation_result_blume.x
    blume_results_dict = dict(zip(params_blume, optimised_params_blume))

    print("Optimised Blume-Tjon Parameters:")
    for i, name in enumerate(params_blume):
        print(f"{name}: {optimised_params_blume[i]:.4f}")

    # L-BFGS-B and Uncertainty calculation for Blume not included
    
    print("\nGenerating final dynamic 'Blume-Tjon' model spectra...")
    optimised_spectra_matrix_blume = moss.generate_model_spectra_matrix(
        temperatures, x_values, moss.collapsed_blume, 
        **fixed_parameters_blume,
        **blume_results_dict
    )
#%%
# Save the plots for each temperature spectrum calculated by each method


    plot_directory_static = file_home+"/"+"fit_plots_static"
    if not os.path.exists(plot_directory_static):
        os.makedirs(plot_directory_static)

    plot_temperatures = temperatures # Use all temperatures
    
    # Get the static optimised parameters
    optimized_params_static = optimisation_result_static.x
    param_values_static = {name: optimized_params_static[i] for i, name in enumerate(params_static)}

    for i, temp in enumerate(plot_temperatures):
        temp_index = np.where(temperatures == temp)[0][0]
        plt.figure(figsize=(8, 6))

        # Plot measured and STATIC optimised spectra
        plt.plot(x_values, measured_spectra_dict[temp], label=f'Measured ({temp}K)', linewidth=1)
        plt.plot(x_values, optimised_spectra_matrix_static[temp_index], label=f'xVBF Fit ({temp}K)', linestyle='--', linewidth=1)

        plt.title(f"Measured vs Static Fit at {temp}K\n(Background Normalised)")
        plt.xlabel("Velocity (mm/s)")
        plt.ylabel("Transmission (Background Normalised)")
        plt.legend()

        # --- Calculate and Add STATIC Parameter Text to Plot ---
        cs_val = moss.calculate_CS(param_values_static['thD'], temp, param_values_static['del1'])
        qs_val = moss.calculate_QS(param_values_static['thD'], temp, param_values_static['QS_nought'])
        epsilon_val = (qs_val * (3 * np.cos(param_values_static['theta']) ** 2 -1)) / (4)
        
        try:
            h_val = moss.Temp_H(temp, param_values_static['T_Block'], param_values_static['B_sat'], param_values_static['B_sat'], param_values_static['nu'])
        except RuntimeError:
            h_val = 0.0

        parameter_text = (
            f"Static Model Parameters:\n"
            f"  CS: {cs_val:.4f}\n"
            f"  QS: {qs_val:.4f}\n"
            f"  σ(QS): {param_values_static['sigma_QS']:.4f}\n"
            f"  ε: {epsilon_val:.4f}\n"
            f"  σ(ε): {param_values_static['sigma_ep']:.4f}\n"
            f"  H: {h_val:.4f}\n"
            f"  σ(H): {param_values_static['sigma_H']:.4f}\n"
        )
        plt.text(0.02, 0.95, parameter_text, transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plot_filename = os.path.join(plot_directory_static, f"Static_Fit_Spectrum_{temp}K.png")
        plt.savefig(plot_filename)
        plt.close()

    print(f"Static plots saved to directory: {plot_directory_static}")
    
   
    #----------------
    plot_directory_dynamic = file_home+"/"+"fit_plots_dynamic"
    if not os.path.exists(plot_directory_dynamic):
        os.makedirs(plot_directory_dynamic)
        
    param_values_dynamic = {name: optimised_params_dynamic[i] for i, name in enumerate(params_dynamic)}
    
    for i, temp in enumerate(plot_temperatures):
        temp_index = np.where(temperatures == temp)[0][0]
        plt.figure(figsize=(8, 6))

        # Plot measured and DYNAMIC optimised spectra
        plt.plot(x_values, measured_spectra_dict[temp], label=f'Measured ({temp}K)', linewidth=1)
        plt.plot(x_values, optimised_spectra_matrix_dynamic[temp_index], label=f'xVBF-Wickman Fit ({temp}K)', linestyle='--', linewidth=1)

        plt.title(f"Measured vs Dynamic Fit at {temp}K\n(Background Normalised)")
        plt.xlabel("Velocity (mm/s)")
        plt.ylabel("Transmission (Background Normalised)")
        plt.legend()

        # --- Calculate and Add DYNAMIC Parameter Text to Plot ---
        cs_val = moss.calculate_CS(param_values_dynamic['thD'], temp, param_values_dynamic['del1'])
        qs_val = moss.calculate_QS(param_values_dynamic['thD'], temp, param_values_dynamic['QS_nought'])
        epsilon_val = (qs_val * (3 * np.cos(param_values_dynamic['theta']) ** 2 -1)) / (4)
        
        # This H calculation uses the DYNAMIC logic (H depends on T_Crit)
        # T_Crit is in the fixed_parameters_dynamic dictionary
        try:
            h_val = moss.Temp_H(temp, param_values_dynamic['T_Block'], param_values_dynamic['B_sat'], param_values_dynamic['B_sat'], param_values_dynamic['nu'])
        except RuntimeError:
            h_val = 0.0

        parameter_text = (
            f"Dynamic Model Parameters:\n"
            f"  CS: {cs_val:.4f})\n"
            f"  QS: {qs_val:.4f}  (σ(QS): {param_values_dynamic['sigma_QS']:.4f})\n"
            f"  ε: {epsilon_val:.4f}  (σ(ε): {param_values_dynamic['sigma_ep']:.4f})\n"
            f"  H: {h_val:.4f}  (σ(H): {param_values_dynamic['sigma_H']:.4f})\n"
            f"  log10(f0): {param_values_dynamic['log10_f0']:.4f}\n"
            f"  T_Block: {param_values_dynamic['T_Block']:.2f}K (σ(T_B): {param_values_dynamic['sig_T_Block']:.2f})\n"
        )
        plt.text(0.02, 0.95, parameter_text, transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plot_filename = os.path.join(plot_directory_dynamic, f"Dynamic_Fit_Spectrum_{temp}K.png")
        plt.savefig(plot_filename)
        plt.close()

    print(f"Dynamic plots saved to directory: {plot_directory_dynamic}")
    

    #----------------
    plot_directory_blume = file_home+"/"+"fit_plots_blume"
    if not os.path.exists(plot_directory_blume):
        os.makedirs(plot_directory_blume)
        
    # Use the dictionary we created earlier
    param_values = blume_results_dict 
    
    for i, temp in enumerate(plot_temperatures):
        temp_index = np.where(temperatures == temp)[0][0]
        plt.figure(figsize=(8, 6))
   
        # Plot measured and BLUME optimised spectra
        plt.plot(x_values, measured_spectra_dict[temp], label=f'Measured ({temp}K)', linewidth=1)
        plt.plot(x_values, optimised_spectra_matrix_blume[temp_index], label=f'Blume-Tjon Fit ({temp}K)', linestyle='--', linewidth=1)
   
        plt.title(f"Measured vs Blume-Tjon Fit at {temp}K\n(Background Normalised)")
        plt.xlabel("Velocity (mm/s)")
        plt.ylabel("Transmission (Background Normalised)")
        plt.legend()
   

        cs_val = moss.calculate_CS(param_values['thD'], temp, param_values['del1'])
        qs_val = moss.calculate_QS(param_values['thD'], temp, param_values['QS_nought'])
        epsilon_val = (qs_val * (3 * np.cos(param_values['theta']) ** 2 -1)) / (4.0)
        
        # Calculate H using the dictionary values
        try:
            h_val = moss.Temp_H(temp, param_values['T_Block'], param_values['B_sat'], param_values['B_sat'], param_values['nu'])
        except RuntimeError:
            h_val = 0.0
   
        parameter_text = (
            f"Blume Model Parameters:\n"
            f"  CS: {cs_val:.4f}\n"
            f"  QS: {qs_val:.4f}  (σ(QS): {param_values['sigma_QS']:.4f})\n"
            f"  ε: {epsilon_val:.4f}  (σ(ε): {param_values['sigma_ep']:.4f})\n"
            f"  H: {h_val:.4f}  (σ(H): {param_values['sigma_H']:.4f})\n"
            f"  log10(f0): {param_values['log10_f0']:.4f}\n"
            f"  T_Block: {param_values['T_Block']:.2f}K (σ(T_B): {param_values['sig_T_Block']:.2f})\n"
        )
        plt.text(0.02, 0.95, parameter_text, transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
   
        plot_filename = os.path.join(plot_directory_blume, f"Blume_Fit_Spectrum_{temp}K.png")
        plt.savefig(plot_filename)
        plt.close()
   
    print(f"Blume plots saved to directory: {plot_directory_blume}")

#%%
# recombine the plots of the fitting results into an animation
   
    print("\n--- GENERATING FIT ANIMATIONS ---")

    animation_directory = "C:/Users/grigga/Documents/python_output/fit_animations"
    if not os.path.exists(animation_directory):
        os.makedirs(animation_directory)

    # GENERATE STATIC FIT ANIMATION 
    print("Generating STATIC fit animation...")

    # Create the figure and axes
    fig_static, ax_static = plt.subplots(figsize=(10, 7))
    
    # Initialize empty plot lines
    line_measured_stat, = ax_static.plot([], [], label='Measured', linewidth=1, color='black')
    line_optimised_stat, = ax_static.plot([], [], label='Static Fit', linestyle='--', linewidth=1, color='red')
    
    # Initialize text
    title_text_stat = ax_static.set_title("")
    ax_static.set_xlabel("Velocity (mm/s)")
    ax_static.set_ylabel("Transmission (Background Normalised)")
    ax_static.legend()
    
    # Set fixed axis limits
    ax_static.set_xlim(x_values.min(), x_values.max())
    y_min_overall = min(np.min([np.min(measured_spectra_dict[t]) for t in temperatures]), np.min(optimised_spectra_matrix_static))
    y_max_overall = max(np.max([np.max(measured_spectra_dict[t]) for t in temperatures]), np.max(optimised_spectra_matrix_static))
    ax_static.set_ylim(y_min_overall * 0.9, y_max_overall * 1.1)
    
    param_text_box_stat = ax_static.text(0.02, 0.95, '', transform=ax_static.transAxes, verticalalignment='top',
                                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Define the animation function for the STATIC model
    def animate_static(i):
        current_temp = temperatures[i]
        measured_spectrum = measured_spectra_dict[current_temp]
        optimised_spectrum = optimised_spectra_matrix_static[i]

        line_measured_stat.set_data(x_values, measured_spectrum)
        line_optimised_stat.set_data(x_values, optimised_spectrum)
        
        title_text_stat.set_text(f"Measured vs Static Fit at {current_temp:.1f}K\n(Background Normalised)")
        
        # Use the STATIC parameters
        param_values = {name: optimized_params_static[j] for j, name in enumerate(params_static)}
        
        cs_val = moss.calculate_CS(param_values['thD'], current_temp, param_values['del1'])
        qs_val = moss.calculate_QS(param_values['thD'], current_temp, param_values['QS_nought']) # Fixed bug: was 'temp'
        epsilon_val = (qs_val * (3 * np.cos(param_values['theta']) ** 2 -1)) / (4)
        
        # STATIC H-field logic: H depends on T_Block
        try:
            h_val = moss.Temp_H(current_temp, param_values['T_Block'], param_values['B_sat'], param_values['B_sat'], param_values['nu'])
        except RuntimeError:
            h_val = 0.0
        
        parameter_text = (
            f"Static Model (T_B-based H):\n"
            f"  CS: {cs_val:.4f}\n"
            f"  QS: {qs_val:.4f}\n"
            f"  σ(QS): {param_values['sigma_QS']:.4f}\n"
            f"  ε: {epsilon_val:.4f}\n"
            f"  H: {h_val:.4f}\n"
            f"  σ(H): {param_values['sigma_H']:.4f}\n"
        )
        param_text_box_stat.set_text(parameter_text)
        
        return line_measured_stat, line_optimised_stat, title_text_stat, param_text_box_stat

    # Create the static animation
    ani_static = animation.FuncAnimation(
        fig_static, animate_static, frames=len(temperatures), interval=200, blit=True, repeat=False
    )

    # Save the static animation
    try:
        writer_name = 'pillow'
        file_extension = '.gif'
        animation_filename = os.path.join(animation_directory, f"mosssbauer_STATIC_fit_animation{file_extension}")
        ani_static.save(animation_filename, writer=writer_name, fps=5)
        print(f"Static animation saved successfully to: {animation_filename}")
    except Exception as e:
        print(f"Error saving STATIC animation: {e}")
        print("Please ensure the Pillow library is installed (pip install Pillow).")

    plt.close(fig_static) # Close the figure

    # GENERATE DYNAMIC FIT ANIMATION 
    print("Generating DYNAMIC fit animation...")

    # Create the figure and axes
    fig_dynamic, ax_dynamic = plt.subplots(figsize=(10, 7))
    
    # Initialize empty plot lines
    line_measured_dyn, = ax_dynamic.plot([], [], label='Measured', linewidth=1, color='black')
    line_optimised_dyn, = ax_dynamic.plot([], [], label='Dynamic Fit', linestyle='--', linewidth=1, color='blue')
    
    # Initialize text
    title_text_dyn = ax_dynamic.set_title("")
    ax_dynamic.set_xlabel("Velocity (mm/s)")
    ax_dynamic.set_ylabel("Transmission (Background Normalised)")
    ax_dynamic.legend()
    
    # Set fixed axis limits
    ax_dynamic.set_xlim(x_values.min(), x_values.max())
    y_min_overall_dyn = min(np.min([np.min(measured_spectra_dict[t]) for t in temperatures]), np.min(optimised_spectra_matrix_dynamic))
    y_max_overall_dyn = max(np.max([np.max(measured_spectra_dict[t]) for t in temperatures]), np.max(optimised_spectra_matrix_dynamic))
    ax_dynamic.set_ylim(y_min_overall_dyn * 0.9, y_max_overall_dyn * 1.1)
    
    param_text_box_dyn = ax_dynamic.text(0.02, 0.95, '', transform=ax_dynamic.transAxes, verticalalignment='top',
                                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Define the animation function for the DYNAMIC model
    def animate_dynamic(i):
        current_temp = temperatures[i]
        measured_spectrum = measured_spectra_dict[current_temp]
        optimised_spectrum = optimised_spectra_matrix_dynamic[i]

        line_measured_dyn.set_data(x_values, measured_spectrum)
        line_optimised_dyn.set_data(x_values, optimised_spectrum)
        
        title_text_dyn.set_text(f"Measured vs Dynamic Fit at {current_temp:.1f}K\n(Background Normalised)")
        
        # Use the DYNAMIC parameters
        param_values = {name: optimised_params_dynamic[j] for j, name in enumerate(params_dynamic)}
        
        cs_val = moss.calculate_CS(param_values['thD'], current_temp, param_values['del1'])
        qs_val = moss.calculate_QS(param_values['thD'], current_temp, param_values['QS_nought']) # Fixed bug: was 'temp'
        epsilon_val = (qs_val * (3 * np.cos(param_values['theta']) ** 2 -1)) / (4)
        
        # DYNAMIC H-field logic: H depends on T_Crit
        try:
            h_val = moss.Temp_H(current_temp, param_values['T_Block'], param_values['B_sat'], param_values['B_sat'], param_values['nu'])
        except RuntimeError:
            h_val = 0.0
        
        parameter_text = (
            f"Dynamic Model (T_C-based H):\n"
            f"  CS: {cs_val:.4f} \n"
            f"  QS: {qs_val:.4f} (σ(QS): {param_values['sigma_QS']:.4f})\n"
            f"  ε: {epsilon_val:.4f} (σ(ε): {param_values['sigma_ep']:.4f})\n"
            f"  H: {h_val:.4f} (σ(H): {param_values['sigma_H']:.4f})\n"
            f"  log10(f0): {param_values['log10_f0']:.4f}\n"
            f"  T_Block: {param_values['T_Block']:.2f}K\n"
        )
        param_text_box_dyn.set_text(parameter_text)
        
        return line_measured_dyn, line_optimised_dyn, title_text_dyn, param_text_box_dyn

    # Create the dynamic animation
    ani_dynamic = animation.FuncAnimation(
        fig_dynamic, animate_dynamic, frames=len(temperatures), interval=200, blit=True, repeat=False
    )

    # Save the dynamic animation
    try:
        writer_name = 'pillow'
        file_extension = '.gif'
        animation_filename = os.path.join(animation_directory, f"mosssbauer_DYNAMIC_fit_animation{file_extension}")
        ani_dynamic.save(animation_filename, writer=writer_name, fps=5)
        print(f"Dynamic animation saved successfully to: {animation_filename}")
    except Exception as e:
        print(f"Error saving DYNAMIC animation: {e}")
        print("Please ensure the Pillow library is installed (pip install Pillow).")

    plt.close(fig_dynamic) # Close the figure

    print("\nAnimation generation complete.")
    
    
 #%%  

# plot key temperatures on a single axis

    print("--- PLOTTING COMPARISON WATERFALL ---")

    # List of temperatures to plot
    plot_temperatures_list = [5, 50, 55, 60, 65, 70, 75, 80, 140] 
    
    # How much to shift each spectrum vertically
    waterfall_offset = 0.35
    
    plt.figure(figsize=(10, 20))
    ax = plt.gca()
    
    # We plot from high-T to low-T, so 140K is at the bottom
    current_offset_level = 0.0
    
    # Use reversed() to plot 140K first, at the bottom of the stack
    for i, temp in enumerate(reversed(plot_temperatures_list)):
        
        # Safety Check
        if temp not in measured_spectra_dict:
            print(f"Warning: Temperature {temp}K not found in measured data. Skipping.")
            continue
            
        if temp not in temperatures:
             print(f"Warning: Temperature {temp}K not found in matrix. Skipping.")
             continue
        
        # Get the measured spectrum
        measured_spec = measured_spectra_dict[temp]
        
        # Find the index for the matrix rows
        temp_index = np.where(temperatures == temp)[0][0]
        
        # Get the two optimised spectra
        static_spec = optimised_spectra_matrix_static[temp_index]
        dynamic_spec = optimised_spectra_matrix_dynamic[temp_index]
        blume_spec = optimised_spectra_matrix_blume[temp_index]
        
        # Plot
        label_m = "Measured" if i == 0 else None
        label_s = "xVBF Fit" if i == 0 else None
        label_d = "xVBF-Wickman Fit" if i == 0 else None
        label_b = "Blume-Tjon Fit" if i == 0 else None
        
        ax.plot(x_values, measured_spec + current_offset_level, 
                label=label_m, color='black', linewidth=1.5)       
        ax.plot(x_values, static_spec + current_offset_level, 
                label=label_s, color='red', linestyle='--', linewidth=1.5)       
        ax.plot(x_values, dynamic_spec + current_offset_level, 
                label=label_d, color='blue', linestyle='--', linewidth=1.5)       
        ax.plot(x_values, blume_spec + current_offset_level, 
                label=label_b, color='darkorange', linestyle='--', linewidth=1.5)
        
        # Add a text label for the temperature
        ax.text(x_values[-50] * 1.05, 1.04 + current_offset_level, f"{temp} K", fontsize=20)
        
        # Increase the offset for the next loop
        current_offset_level += waterfall_offset

    # Final Plot Formatting
    ax.set_title("Model Fitting Comparison", fontsize=20)
    ax.set_xlabel("Velocity (mm/s)", fontsize=20)
    ax.set_ylabel("Transmission (Vertically Offset)", fontsize=20)
    ax.legend(fontsize=20)
    
    ax.set_yticks([]) 
    ax.tick_params(axis='x', which='major', labelsize=20)
    
    # Ensure plot is tight
    plt.tight_layout()
    
    # Save the final comparison plot
    plot_filename_comparison = os.path.join(plot_directory_dynamic, "Comparison_Waterfall_Plot.png")
    plt.savefig(plot_filename_comparison)
    plt.close()

    print(f"Comparison waterfall plot saved to: {plot_filename_comparison}")
    
#%%
   
# Plot the fitting results as a data matrix 

def plot_four_panel_matrix(velocity, temperatures, measured_dict, 
                           static_matrix, wickman_matrix, blume_matrix,
                           vmin=0.65, vmax=1.01):
    """
    Plots Experimental Data vs. 3 Models in a 2x2 Heatmap Grid.
    """
    
    # 1. PREPARE DATA
    # Convert the Measured Dictionary to a Matrix to match the models
    measured_matrix = []
    # Ensure we sort by temperature to align with the model matrices
    sorted_temps = sorted(temperatures) 
    
    for t in sorted_temps:
        if t in measured_dict:
            measured_matrix.append(measured_dict[t])
        else:
            # Fallback if a temp is missing (shouldn't happen based on your script)
            measured_matrix.append(np.ones_like(velocity))
    measured_matrix = np.array(measured_matrix)

    # 2. SETUP PLOT
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    
    # Define the datasets and titles for the loop
    # Order: Top-Left, Top-Right, Bottom-Left, Bottom-Right
    panels = [
        (axes[0, 0], measured_matrix, "Experimental Data"),
        (axes[0, 1], static_matrix,   "Static Model (xVBF)"),
        (axes[1, 0], wickman_matrix,  "Dynamic Model (Wickman)"),
        (axes[1, 1], blume_matrix,    "Dynamic Model (Blume-Tjon)")
    ]

    # Create Meshgrid for mapping (Velocity vs Temperature)
    X, Y = np.meshgrid(velocity, sorted_temps)

    # 3. PLOTTING LOOP
    mesh_ref = None # To store one mesh for the colorbar
    
    for ax, data, title in panels:
        # Pcolormesh
        mesh = ax.pcolormesh(
            X, Y, data, 
            cmap='viridis', 
            shading='auto', 
            vmin=vmin, 
            vmax=vmax
        )
        mesh_ref = mesh # Save reference
        
        # Styling
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.3, color='white')

    # 4. LABELS & LAYOUT
    # Set labels only on the outer edges
    axes[1, 0].set_xlabel('Velocity (mm/s)', fontsize=11)
    axes[1, 1].set_xlabel('Velocity (mm/s)', fontsize=11)
    axes[0, 0].set_ylabel('Temperature (K)', fontsize=11)
    axes[1, 0].set_ylabel('Temperature (K)', fontsize=11)

    # Add a shared Colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    cbar = fig.colorbar(mesh_ref, cax=cbar_ax)
    cbar.set_label('Normalized Transmission', fontsize=11)

    plt.subplots_adjust(wspace=0.1, hspace=0.15, right=0.9)
    
    return fig
  

# --- FINAL 4-PANEL PLOT ---

print("\nGenerating comparison plots...")

# Note: You can adjust vmin to highlight the absorption features better.
# For normalized data, 0.90 to 1.01 often looks best to see the "wings".
fig_comparison = plot_four_panel_matrix(
    velocity=x_values,
    temperatures=temperatures,
    measured_dict=measured_spectra_dict,
    static_matrix=optimised_spectra_matrix_static,
    wickman_matrix=optimised_spectra_matrix_dynamic, 
    blume_matrix=optimised_spectra_matrix_blume,    
    vmin=0.60, 
    vmax=1.01
)

# Save the plot
plot_filename = file_home + "/Four_Panel_Comparison_Matrix.png"
fig_comparison.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Comparison plot saved to: {plot_filename}")
plt.show()
  
#%%
  
# Create Table of paramters


static_opt = dict(zip(params_static, optimised_params_static))
dynamic_opt = dict(zip(params_dynamic, optimised_params_dynamic))
blume_opt = dict(zip(params_blume, optimised_params_blume))

models_data = [
    ('xVBF',         static_opt,  fixed_parameters_static),
    ('xVBF-Wickman', dynamic_opt, fixed_parameters_dynamic),
    ('Blume-Tjon',   blume_opt,   fixed_parameters_blume)
]

parameter_rows = [
    ('thD',         'ThD [K]'),
    ('del1',        'CS₀ [mm/s]'),
    ('QS_nought',   'QS₀ [mm/s]'),
    ('B_sat',       'B_sat [T]'),
    ('T_Block',     'T_B [K]'),
    ('sigma_QS',    'σ(QS) [mm/s]'),
    ('sigma_ep',    'σ(ε) [mm/s]'),
    ('sigma_H',     'σ(H) [T]'),
    ('sig_T_Block', 'σ(T_B) [K]'),
    ('sigTB_res',   'σ(T_B) step resolution'),
    ('nu',          'ν'),
    ('theta',       'θ [rad]'),
    ('intensity',   'Intensity'),
    ('log10_f0',    'log₁₀(f₀)'),
    ('A',           'A'),
    ('B',           'B'),
    ('C',           'C'),
    ('linewidth_L', 'Γ [mm/s]'), 
]


table_data = []
for var_name, display_name in parameter_rows:
    row = {'Parameter': display_name}
    
    # Extract values for each model, defaulting to "N/A" if not found
    for model_name, opt_dict, fixed_dict in models_data:
        
        val_str = "N/A" # Default if not found
        
        if var_name in fixed_dict:
            val = fixed_dict[var_name]
            # Format with asterisk
            if isinstance(val, (int, float)):
                val_str = f"{val:.2f}*" 
            else:
                val_str = f"{val}*"
                
        elif var_name in opt_dict:
            val = opt_dict[var_name]
            # Format normally
            if isinstance(val, (int, float)):
                val_str = f"{val:.2f}"
            else:
                val_str = str(val)
                
        row[model_name] = val_str
            
    table_data.append(row)

df_results = pd.DataFrame(table_data)
df_results.set_index('Parameter', inplace=True)
print("\n" + "="*50)
print("FINAL PARAMETER COMPARISON TABLE")
print("="*50)
print(df_results)


excel_path = file_home + "/Parameter_Comparison_Table.xlsx"

try:
    df_results.to_excel(excel_path)
    print(f"\nExcel file saved to: {excel_path}")
except ImportError:
    print("\nError: 'openpyxl' library not found. Saving as CSV instead.")
    # Fallback to CSV if you don't have the excel library installed
    # We use sep=';' which works better for European Excel versions
    df_results.to_csv(file_home + "/Parameter_Comparison_Table.csv", sep=';')
    print(f"CSV saved to: {file_home}/Parameter_Comparison_Table.csv")
