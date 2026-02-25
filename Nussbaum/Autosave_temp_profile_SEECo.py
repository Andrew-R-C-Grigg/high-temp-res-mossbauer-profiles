# -*- coding: utf-8 -*-
"""
Nussbaum - a python package for automated high-resolution Mössbauer spectroscopy temperature profile measurements

@author: Andrew R. C. Grigg*, James M Byrne, Ruben Kretzschmar
*ETH Zurich, Department of Environmental System Science, Institute for Biogeochemistry and Pollutant Dynamics

License: Apache 2.0 (http://www.apache.org/licenses/)

The code to automatically download Mössbauer data from SEECo software 
at a user defined time interval
and control a Lake Shore Model 336 controller
to change temperature when user defined thresholds of signal to noise ratio are met.

Data files are saved in a 'log' file as a backup 
that can be accessed until the code is run the next time, 
and data is additionally saved in .dat format without the metadata from the .ws5 file 
in a directory location of the user's choosing.

"""

import serial.tools.list_ports
import sys
import gc
sys.coinit_flags = 2  # COINIT_APARTMENTTHREADED
from pywinauto.application import Application
from pywinauto.timings import wait_until
import pywinauto
import time
import datetime
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import ttk 
import os
import glob
import shutil
import csv
# import pickle
import math
import numpy as np
import nussbaum.utils.s2n as s2n
import nussbaum.utils.fold as fold
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import comtypes
import subprocess
import json 
import matplotlib.pyplot as plt
from lakeshore import Model336, Model336InputSensorSettings, InstrumentException
from pathlib import Path


import serial.tools.list_ports


#%% Open the application

try:
# check if it is already open
    app = Application('uia').connect(path=r"C:\Users\gac-MB\Desktop\SEECo\W302_20141031.exe")
    wnd = app.top_window()
    print(wnd.texts()[0])
    dlg=app.window(title_re="W302 Resonant Gamma-Ray Spectrometer  sn086 ver.W302_20141031")

except:
# if it is not already open, open now
    app = Application('uia').start(r"C:\Users\gac-MB\Desktop\SEECo\W302_20141031.exe")
    dlg = app['W302']
    app['dlg']['OK'].click()
    dlg=app.window(title_re="W302 Resonant Gamma-Ray Spectrometer  sn086 ver. W302_20141031")



def check_pid_values():
    instrument = None
    try:
        instrument = Model336(com_port='COM4')
        
        for i in [1, 2]:
            # PID? <loop> returns <P value>, <I value>, <D value>
            pid_raw = instrument.query(f'PID? {i}').split(',')
            p, i_val, d = pid_raw[0], pid_raw[1], pid_raw[2]
            
            # Also check if it's in Auto-PID mode (Zone tuning)
            # 0=Manual, 1=Zone, 2=Auto-PID
            tuning_mode = instrument.query(f'CMODE? {i}')

            print(f"\n--- Loop {i} PID Settings ---")
            print(f"P (Proportional): {p}")
            print(f"I (Integral):     {i_val}")
            print(f"D (Derivative):   {d}")
            print(f"Tuning Mode:      {tuning_mode} (0=Manual)")

    except Exception as e:
        print(f"PID check failed: {e}")
    finally:
        if instrument:
            instrument.disconnect_usb()

check_pid_values()


#%%

# =============================================================================
# --- Global Variables ---
# =============================================================================

# Main state variables
running = False
is_ramping = False

Enhanced_SNR = False
normal_spectrum_saved = False
special_case=False
acccount = 0
profile_start_time = None
then = 0

# User-defined paths
savepath = ""
calpath = ""

# Data storage
s2ns = {}
as2ns = {}
temp_timestamps = []
temp_A_values = []
temp_B_values = []

# --- Plotting components & their data ---
# For external 3D plot window
temp_plot_window = None
temp_ax = None
temp_canvas = None
temp_data = []

# For the third plot (temperature trace)
canvas3 = None
ax3 = None

# --- Configuration & constants ---
t_threshold_ESNRP = 10000

# --- GUI / State variables for new feature ---
threshold_mode = None # Will be set to a tk.StringVar in GUI setup
pause_asnr_labels = [] # Will store references to pause point labels


# Create the separate plotting window
def create_temp_plot_window():
    """
    Launches the separate plotting GUI script as an independent process.
    """
    plotting_script_path = r"C:\Nussbaum\Matrix_plotting.py"

    if not savepath or not calpath:
        label6_var.set("Error: Please set save and calibration paths first!")
        print("Error: Save path or calibration path not set.")
        return

    # --- 3. Launch the script with the paths as arguments ---
    try:
        command = ["python", plotting_script_path, savepath, calpath]
        subprocess.Popen(command)
        print(f"Launched plotting GUI with save path: {savepath}")
    except FileNotFoundError:
        label6_var.set("Error: Plotting script not found!")
        print(f"ERROR: Could not find the plotting script at: {plotting_script_path}")


# Set up the log file

LOG_SAVE_PATH = '.'  # Default to the current script's directory
LOG_FILENAME = ""    # This will be constructed by init_logger()
LOG_START_TIME = None
LOG_CURRENT_ACTIVITY = None
LOG_TOTALS = {
    "Temperature Change": 0,
    "Normal Spectra": 0,
    "Enhanced Spectra": 0
}

def _format_duration(seconds):
    """Helper function to format seconds into a consistent string format."""
    return f"{seconds:.2f}"

def log_event(activity_type, status, temp=None, snr=None, asnr=None):
    """
    Logs the start or end of an activity using global state variables.
    """
    # Declare which global variables will be modified by this function.
    global LOG_START_TIME, LOG_CURRENT_ACTIVITY, LOG_TOTALS

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    duration_str = "-"
    temp_str = f"{temp:.2f}" if temp is not None else ""
    snr_str = f"{snr:.1f}" if snr is not None else "-"
    asnr_str = f"{asnr:.1f}" if asnr is not None else "-"

    # When an activity FINISHES, calculate its duration and add to totals.
    if status.lower() == 'finish' and LOG_START_TIME is not None:
        elapsed_time = time.time() - LOG_START_TIME
        if LOG_CURRENT_ACTIVITY in LOG_TOTALS:
            LOG_TOTALS[LOG_CURRENT_ACTIVITY] += elapsed_time
        duration_str = _format_duration(elapsed_time)
        # Clear the start time and activity, ready for the next one.
        LOG_START_TIME = None
        LOG_CURRENT_ACTIVITY = None
        temp_str = ""  # Don't log a target temp on 'finish' events.
        snr_str = f"{snr:.1f}" if snr is not None else "-"
        asnr_str = f"{asnr:.1f}" if asnr is not None else "-"

    # When an activity STARTS, record its start time.
    elif status.lower() == 'start':
        LOG_START_TIME = time.time()
        LOG_CURRENT_ACTIVITY = activity_type

    # Write the event to the log file.
    with open(LOG_FILENAME, 'a') as f:
        f.write(f"{timestamp:<22}{activity_type:<20}{status:<10}{duration_str:<15}{temp_str:<18}{snr_str:<8}{asnr_str:<8}\n")

def write_summary():
    """Writes the final summary of total times to the log file."""
    with open(LOG_FILENAME, 'a') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("EXPERIMENT SUMMARY\n")
        f.write("=" * 80 + "\n")
        total_time = sum(LOG_TOTALS.values())
        f.write(f"Total time changing temperature: {_format_duration(LOG_TOTALS['Temperature Change'])} s\n")
        f.write(f"Total time on Normal Spectra:    {_format_duration(LOG_TOTALS['Normal Spectra'])} s\n")
        f.write(f"Total time on Enhanced Spectra:  {_format_duration(LOG_TOTALS['Enhanced Spectra'])} s\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Logged Experiment Time:    {_format_duration(total_time)} s\n")
        f.write("=" * 80 + "\n")

def init_logger():
    """
    Creates the log file in the chosen directory and writes the header.
    Must be called once at the start of the experiment.
    """
    # <<< MODIFIED: This function now constructs the full filename and path. >>>
    global LOG_FILENAME
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = f"experiment_log_{timestamp}.txt"
    LOG_FILENAME = os.path.join(LOG_SAVE_PATH, log_file)

    try:
        # Create the target directory if it doesn't exist.
        os.makedirs(LOG_SAVE_PATH, exist_ok=True)

        with open(LOG_FILENAME, 'w') as f:
            f.write(f"Log File created at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Log file location: {os.path.abspath(LOG_FILENAME)}\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'Timestamp':<22}{'Activity':<20}{'Status':<10}{'Duration (s)':<15}{'Target Temp (K)':<18}{'SNR':<8}{'aSNR':<8}\n")
            f.write("-" * 96 + "\n") # Increased separator length
        print(f"Logging to file: {os.path.abspath(LOG_FILENAME)}")
    except OSError as e:
        print(f"Error: Could not create log file at {LOG_FILENAME}. Please check permissions. Details: {e}")
        # Optionally, update a GUI label to show this error.


# now code to run the main window and functions

# 
def isfloat(value):
    """
    A useful function to determine if a number is a float

    Parameters
    ----------
    value : number to test

    Returns
    -------
    bool 

    """
    try:
        float(value)
        return True
    except ValueError or IndexError:
        return False

def _clean_val(val_str):
    """Helper to convert string from Entry to a clean int or float."""
    try:
        val_f = float(val_str.strip())
        if val_f.is_integer():
            return int(val_f)
        return val_f
    except ValueError:
        # Not a number, just return the stripped string
        return val_str.strip()

def opensaveloc():
    """A function to define the save location"""
    global savepath
    savepath = filedialog.askdirectory()
    print(savepath)
    label1['text'] = os.path.basename(os.path.normpath(savepath))

def opencalloc():
    """A function to define the calibration file location"""
    global calpath
    calpath = filedialog.askopenfilename()
    print(calpath)
    label2['text'] = os.path.basename(os.path.normpath(calpath))

def save_project():
    """Saves the current GUI configuration to a JSON file."""
    
    # Ask for file name
    filepath = filedialog.asksaveasfilename(
        defaultextension=".json",
        filetypes=[("JSON Project Files", "*.json"), ("All Files", "*.*")],
        title="Save Project As"
    )
    if not filepath:
        print("Save project cancelled.")
        return

    # Gather all data from the GUI, using _clean_val
    project_data = {
        'savepath': savepath,
        'calpath': calpath,
        'main_settings': {
            'refresh_interval': _clean_val(e1.get()),
            'target_threshold': _clean_val(e2.get()),
            'temp_interval': _clean_val(e3.get()),
            'setpoint_temp': _clean_val(e4.get())
        },
        'threshold_mode': threshold_mode.get(),
        'pause_points': []
    }

    # Gather pause point data, using _clean_val
    for i in range(len(pause_vars)):
        point_data = {
            'enabled': pause_vars[i].get(),
            'temp': _clean_val(pause_entries_temp[i].get()),
            'threshold': _clean_val(pause_entries_asnr[i].get())
        }
        project_data['pause_points'].append(point_data)

    # 3. Write to JSON file
    try:
        with open(filepath, 'w') as f:
            json.dump(project_data, f, indent=4)
        print(f"Project saved successfully to {filepath}")
        label6_var.set(f"Project saved to {os.path.basename(filepath)}")
    except Exception as e:
        print(f"Error saving project: {e}")
        label6_var.set("Error: Could not save project.")

def load_project():
    """Loads a configuration from a JSON file into the GUI."""
    global savepath, calpath
    
    # Ask for file name
    filepath = filedialog.askopenfilename(
        filetypes=[("JSON Project Files", "*.json"), ("All Files", "*.*")],
        title="Load Project"
    )
    if not filepath:
        print("Load project cancelled.")
        return

    # Read data from JSON file
    try:
        with open(filepath, 'r') as f:
            project_data = json.load(f)
    except Exception as e:
        print(f"Error loading project: {e}")
        label6_var.set("Error: Could not load project file.")
        return

    # Populate the GUI with the loaded data
    try:
        # Set paths
        savepath = project_data.get('savepath', '')
        calpath = project_data.get('calpath', '')
        label1['text'] = os.path.basename(os.path.normpath(savepath)) if savepath else 'location...'
        label2['text'] = os.path.basename(os.path.normpath(calpath)) if calpath else 'location...'

        # Set main settings
        main_settings = project_data.get('main_settings', {})
        e1.delete(0, tk.END); e1.insert(0, str(main_settings.get('refresh_interval', '60')))
        e2.delete(0, tk.END); e2.insert(0, str(main_settings.get('target_threshold', '100')))
        e3.delete(0, tk.END); e3.insert(0, str(main_settings.get('temp_interval', '1')))
        e4.delete(0, tk.END); e4.insert(0, str(main_settings.get('setpoint_temp', '5')))

        # Set threshold mode
        loaded_mode = project_data.get('threshold_mode', 'SNR')
        threshold_mode.set(loaded_mode)
        if loaded_mode == "Time":
            toggle_button.config(text="Mode: Time")
            l2.config(text="Target Time (min):")
            for label in pause_asnr_labels:
                label.config(text="Time (min):")
        else: # SNR mode
            toggle_button.config(text="Mode: aSNR")
            l2.config(text="Target aSNR:")
            for label in pause_asnr_labels:
                label.config(text="aSNR:")

        # Set pause points
        pause_points = project_data.get('pause_points', [])
        for i, point_data in enumerate(pause_points):
            if i < len(pause_vars): # Ensure we don't go out of bounds
                pause_vars[i].set(point_data.get('enabled', 0))
                pause_entries_temp[i].delete(0, tk.END)
                pause_entries_temp[i].insert(0, str(point_data.get('temp', '1')))
                pause_entries_asnr[i].delete(0, tk.END)
                pause_entries_asnr[i].insert(0, str(point_data.get('threshold', '400')))
        
        print(f"Project loaded successfully from {filepath}")
        label6_var.set(f"Project loaded from {os.path.basename(filepath)}")

    except Exception as e:
        print(f"Error applying loaded project settings: {e}")
        label6_var.set("Error: Project file was corrupt or invalid.")

def save_data(t):
    """A function to save data"""
    interval = e1.get()
    
def on_stop():
    """A function to stop the loop"""
    global running
    running = False
    
def clear():
    """a function to clear the memory of the SEECo software"""
    button = app['dlg']['CLEAR Channel 1']
    button.click() 
    button.wait('visible enabled', timeout=30)
    app['dlg']['OK'].click()
    
def on_closing():
    """Handles the application window being closed."""
    on_stop()       # Stop any running measurements.
    if LOG_FILENAME:
        write_summary()   # Write the final summary block to the log file.
    else:
        print("Closing with no log file written")
    root.destroy()

def close_lakeshore_port(port,baudrate): 
    # List all open ports
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        print(f"Found port: {p.device}")
    
    try:
        ser = serial.Serial(port=port, baudrate=baudrate) 
        time.sleep(2)
        ser.close()  # Close the port
        print(f"Closed port {ser.name}")
    except Exception as e:
        print(f"Could not close port: {e}")

def get_pause_point_data():
    """Retrieves data ONLY for enabled pause points,
       returning lists of temps and aSNRs for enabled points.
    """
    enabled_pause_temps_list = []
    enabled_pause_asnrs_list = []

    for i in range(len(pause_vars)):
        if pause_vars[i].get():  # Check if checkbox is checked (enabled)
            temp_str = pause_entries_temp[i].get()
            asnr_str = pause_entries_asnr[i].get()

            try:
                temp = float(temp_str)
                asnr = float(asnr_str)
                enabled_pause_temps_list.append(temp) # Append only if enabled and valid
                enabled_pause_asnrs_list.append(asnr) # Append only if enabled and valid
            except ValueError:
                print(f"Warning: Invalid input for Pause Point {i+1}. Skipping this point.")
                # You could handle invalid input differently, e.g., display an error in GUI

    return enabled_pause_temps_list, enabled_pause_asnrs_list # Return only enabled lists

def toggle_threshold_mode():
    """Switches the thresholding logic between aSNR and Time."""
    if threshold_mode.get() == "SNR":
        # Switch to Time mode
        threshold_mode.set("Time")
        toggle_button.config(text="Mode: Time")
        l2.config(text="Target Time (min):") # l2 is the label for the main profile
        e2.delete(0, tk.END)
        e2.insert(0, "60") # Default to 60 minutes
        
        # Update all pause point labels
        for label in pause_asnr_labels: 
            label.config(text="Time (min):")
        # Update all pause point entries (example value)
        for entry in pause_entries_asnr:
            entry.delete(0, tk.END)
            entry.insert(0, "240") # Default to 4 hours
            
    else:
        # Switch to SNR mode
        threshold_mode.set("SNR")
        toggle_button.config(text="Mode: aSNR")
        l2.config(text="Target aSNR:")
        e2.delete(0, tk.END)
        e2.insert(0, "100") # Original default
        
        # Update all pause point labels
        for label in pause_asnr_labels:
            label.config(text="aSNR:")
        # Update all pause point entries
        for entry in pause_entries_asnr:
            entry.delete(0, tk.END)
            entry.insert(0, "400") # Original default

def refresh_spec():
    app['dlg']['CLEAR Channel 1'].click() 
    app['dlg']['OK'].click() 


def draw_temp_plot(tempA, tempB):
    """
    A dedicated function that handles drawing the temperature plot.
    It takes temperature values as arguments and does not connect to hardware.
    """
    current_time = time.time()
    
    # --- Store and Prune Data ---
    temp_timestamps.append(current_time)
    temp_A_values.append(tempA)
    temp_B_values.append(tempB)

    thirty_minutes_ago = current_time - (30 * 60)
    start_index = 0
    for i, t in enumerate(temp_timestamps):
        if t >= thirty_minutes_ago:
            start_index = i
            break
            
    plot_timestamps = temp_timestamps[start_index:]
    plot_A_values = temp_A_values[start_index:]
    plot_B_values = temp_B_values[start_index:]
    
    # --- Redraw the Plot ---
    ax3.clear()
    if profile_start_time and plot_timestamps:
        relative_times_min = [(t - profile_start_time) / 60 for t in plot_timestamps]
        ax3.plot(relative_times_min, plot_B_values, marker='.', linestyle='-', markersize=2, color='red', label='Sample (B)')
        ax3.plot(relative_times_min, plot_A_values, marker='.', linestyle='-', markersize=2, color='blue', label='Coldhead (A)')
        ax3.grid(True)
        ax3.legend(loc=('center left'), ncol=1, fontsize=5)
    
    ax3.set_title("Temperature Trace", fontsize='x-small')
    ax3.set_xlabel("Time (minutes)", fontsize='x-small')
    ax3.set_ylabel("Temperature (K)", fontsize='x-small')
    ax3.tick_params(axis='both', which='major', labelsize=6)
    canvas3.draw()

def update_temp_plot():
    """
    Fetches temperature data and updates the third plot every 10 seconds.
    This function runs independently of the main measurement_loop.
    """
    if not running or is_ramping:  # Stop the loop if the main measurement is stopped
        if running:
            root.after(10000, update_temp_plot)
        return
        print('not running, no temperature plot update')

    # Get new temperature data 
    try:
        # NOTE: This assumes your Model336 connection logic is available
        my_model_336 = Model336(com_port='COM4')
        tempA = my_model_336.get_kelvin_reading("A")
        tempB = my_model_336.get_kelvin_reading("B")
        my_model_336.disconnect_usb()
        del my_model_336
        current_time = time.time()
        
        # Store the new data 
        draw_temp_plot(tempA, tempB)
        label5_var.set(f'Coldhead: {round(tempA, 2)}K       Sample: {round(tempB, 2)}K')
        label5.update_idletasks()

    except Exception as e:
        # print(f"Failed to read temperature for trace plot: {e}")
        # Still reschedule to try again later
        root.after(10000, update_temp_plot)
        return
    # Reschedule this function to run again in 10 seconds ---
    root.after(10000, update_temp_plot)

RAMP_RATE_K_PER_MIN = 0.18 


def ramp_temperature(final_setpoint, loop_count, ns, then):
    """
    Connects to the controller and begins a non-linear temperature ramp.

    This function establishes the connection and then schedules the main ramp
    logic to run after a short delay, keeping the GUI responsive.
    """
    print("Connecting to controller for ramp...")
    # Establish a connection with the Lake Shore Model 336 controller.
    # This will raise an error if the connection fails.
    my_model_336 = Model336(com_port='COM4')

    above_trigger=True

    def _start_ramp_logic(my_model_336):
        """Contains the main logic for calculating and executing the ramp."""
        print("Connection stable. Starting ramp logic...")
        log_event("Temperature Change", "start", temp=final_setpoint[1])
        
        # Safely check the heater status.
        try:
            current_range = my_model_336.get_heater_range(1)
        except InstrumentException:
            current_range = 0  # HeaterRange.OFF is 0

        # If the heater is on, configure the output modes for both heaters.
        if current_range != 0:
            my_model_336.set_heater_output_mode(1, 1, "A") # Control Loop 1 -> Sensor A
            my_model_336.set_heater_output_mode(2, 1, "B") # Control Loop 2 -> Sensor B

        my_model_336.set_heater_range(1, 3)
        my_model_336.set_heater_range(2, 3)
        
        # Get the initial temperature to calculate the ramp parameters.
        start_temp = my_model_336.get_kelvin_reading("B")
        final_temp = final_setpoint[1]
        delta_T = start_temp - final_temp

        # If already at or below the target temperature, skip the ramp.
        if delta_T <= 0:
            print("Temperature is already at or below the target. Skipping ramp.")
            my_model_336.disconnect_usb()
            del my_model_336

            # Proceed directly to the stabilization function.
            root.after(1000, lambda: wait_for_temp_drop(final_setpoint, loop_count, ns, then))
            return


        def _ramp_step(current_step, my_model_336):
            """A single step in the ramp process, called repeatedly."""
            # Base case: If all steps are completed, finalize the ramp.
            offset=0.15
            nonlocal above_trigger
            if not above_trigger:
                print("Ramp finished. Setting final setpoint. Stabilising for 1 minute")
                status_msg = f"Stabilising... Target: {final_setpoint[1]:.2f}K."
                label6_var.set(status_msg)
                label6.update_idletasks()
                my_model_336.set_control_setpoint(1, final_setpoint[0])
                my_model_336.set_control_setpoint(2, final_setpoint[1]+offset)                
                my_model_336.disconnect_usb()
                del my_model_336
                # Hand off to the stabilization function.
                root.after(60000, lambda: wait_for_temp_drop(final_setpoint, loop_count, ns, then))
                return

            
            # Recursive step: Calculate the next intermediate setpoint.
            # The formula creates an exponential curve from start_temp to final_temp.
            next_setpoint = [final_temp - 3.0, final_temp+offset] # [Heater A, Heater B]
            
            # Send the new setpoint to the instrument.
            my_model_336.set_control_setpoint(1, next_setpoint[0])
            my_model_336.set_control_setpoint(2, next_setpoint[1])
            tempA = my_model_336.get_kelvin_reading("A")
            tempB = my_model_336.get_kelvin_reading("B")
            draw_temp_plot(tempA, tempB)

            if tempB<final_temp+offset:
                above_trigger=False
            else:
            # Update the GUI with the current status.
                status_msg = f"Ramping... Target: {next_setpoint[1]:.2f}K."
                label6_var.set(status_msg)
                label6.update_idletasks()
                label5_var.set(f'Coldhead: {round(tempA, 2)}K       Sample: {round(tempB, 2)}K')
                label5.update_idletasks()
            
            # Schedule the next ramp step after the defined interval.
            root.after(10000, lambda: _ramp_step(current_step + 1, my_model_336))

        # Start the ramp process at the first step.
        _ramp_step(1, my_model_336)

    # Schedule the main logic to start after a 200ms non-blocking pause.
    root.after(200, lambda: _start_ramp_logic(my_model_336))



def wait_for_temp_drop(setpoint, loop_count, ns, then):
    """
    Polls the instrument until temperature stabilizes at the final setpoint.
    
    This function connects, reads the temperature, and immediately disconnects
    in a loop until the sample temperature is within a defined tolerance.
    """
    global acccount, s2ns, as2ns
    
    my_model_336 = None  # Initialize to None to ensure it exists for the 'finally' block.
    try:
        # Connect and Use
        # A new connection is made for each temperature check.
        my_model_336 = Model336(com_port='COM4')
        tempA = my_model_336.get_kelvin_reading("A")
        tempB = my_model_336.get_kelvin_reading("B")
        my_model_336.set_control_setpoint(1, setpoint[0])
        my_model_336.set_control_setpoint(2, setpoint[1])
    except Exception as e:
        # If connection or reading fails, print an error and reschedule a retry.
        print(f"Failed to read temperature during drop: {e}. Retrying...")
        root.after(1000, lambda: wait_for_temp_drop(setpoint, loop_count, ns, then))
        return # Stop execution of this attempt.
    finally:
        # Guaranteed Disconnect
        # This block runs whether the 'try' succeeds or fails, preventing a locked port.
        if my_model_336:
            my_model_336.disconnect_usb()
            del my_model_336
            
    # Update the GUI with the latest temperature readings.
    label5_var.set(f'Coldhead: {round(tempA, 2)}K       Sample: {round(tempB, 2)}K')
    label5.update_idletasks()

    # Check if the sample temperature is still above the target + tolerance.
    # The tolerance is 5% of the value from the GUI entry field 'e3'.
    if tempB > setpoint[1] + 0.05 * float(e3.get()):
        # If not stable, update the status and schedule the next check.
        # The polling interval is read from the GUI entry field 'e1'.
        label6_var.set(f"Stabilizing... Temp: {round(tempB, 2)}K / Target: {setpoint[1]}K")
        label6.update_idletasks()
        root.after(int(float(e1.get()) * 1000), lambda: wait_for_temp_drop(setpoint, loop_count, ns, then))
    else:
        # If stable, print a confirmation message and start the data collection.
        print(f"Temperature stabilized. Starting data collection at {setpoint[1]}K.")
        refresh_spec()
        ax2.clear()
        log_event("Temperature Change", "finish")
        log_event("Normal Spectra", "start", temp=setpoint[1])
            
        # Reset global variables for the new measurement.
        s2ns = {0: 0}
        as2ns = {0: 0}
        then = time.time()       
        acccount = 0

        # Hand off to the main measurement loop.
        root.after(10, lambda: measurement_loop(loop_count, ns, then))

def download_and_save(loop_count,ns):
    nano = 1000000000

    now = time.time()
    nsdiff = (now * nano) - ns
    t = time.localtime(time.time())
    
    try:
        app['dlg']['SAVE DATA'].wait('visible enabled', timeout=30).click()
        app['dlg']['OK'].wait('visible enabled', timeout=30).click()
    except pywinauto.timings.TimeoutError:
        label6_var.set('spectrum saving function timeout. Please press start to continue program')
        label6.update_idletasks()
    
    files = glob.glob(r'C:\W302_data\W302sn086_1\*.t')
    filename = max(files, key=os.path.getmtime)	

    file_path = filename
    wait_until(5, 0.1, lambda: os.path.exists(file_path))
    
    loops_to_save = round(np.divide(600, float(e1.get())), 0)
    
    with open(file_path) as csvfile:
        dreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        if loops_to_save == loop_count:
            loop_count = 1
            with open(file_path, 'w', newline='') as f1:
                writer1 = csv.writer(f1)
                for n, row in enumerate(dreader):
                    if row and isfloat(row[0]):
                        writer1.writerow([int(row[0])])
            
        data = []
        with open(savepath + '\{}{}{}_{}h{}m{}s__{:.0f}__{:.0f}.dat'.format(t[0], t[1], t[2], t[3], t[4], t[5], ns, nsdiff), 'w', newline='') as f2:
            writer2 = csv.writer(f2)
            for n, row in enumerate(dreader):
                if row and isfloat(row[0]):
                    data.append(row[0])
                    writer2.writerow([row[0]])
            data1 = data
    
    return(data1,now,t,nsdiff)


def measurement_loop(loop_count, ns, then):
    global acccount, s2ns, as2ns, Enhanced_SNR, normal_spectrum_saved, t_threshold_ESNRP, special_case
    nano = 1000000000
    setpoint = [float(e4.get()) - 3, float(e4.get())]
    check_path = os.path.join(savepath, 'profile', '{}K.dat'.format(setpoint[1]))
    if os.path.exists(check_path):
        normal_spectrum_saved=True
        Enhanced_SNR = True
    
    if not running:
        print('not running')
        return
    
    my_model_336 = Model336(com_port='COM4')
    try:       
        tempA = my_model_336.get_kelvin_reading("A")
        tempB = my_model_336.get_kelvin_reading("B")
        my_model_336.disconnect_usb()
        
    except:
        print("Failed to read temperature. Retrying...")
        root.after(500, lambda: measurement_loop(loop_count, ns, then))
        return
    finally:
        # Guaranteed Disconnect     
        del my_model_336
            
    # update label in main plotting window
    label5_var.set('Coldhead: {}K       Sample: {}K'.format(round(tempA, 2), round(tempB, 2)))
    label5.update_idletasks()
    
    # Add new temperature data to the list for matrix plot
    temp_data.append({'time': time.time(), 'temp': tempB})
    
    if abs(tempB - setpoint[1]) > 0.05:
            
        def _send_commands_and_reschedule(my_model_336):
            try:
                current_range = my_model_336.get_heater_range(1)
            except InstrumentException:
                current_range = 0
            if current_range != 0:
                print("Heater is active. Sending correction commands...")
                my_model_336.set_heater_range(1, 3)
                my_model_336.set_heater_range(2, 3)
                my_model_336.set_control_setpoint(1, setpoint[0])
                my_model_336.set_control_setpoint(2, setpoint[1])
            # else:
            #     print("Heater is OFF. Skipping commands to prevent errors.")
                
            print("temperature stabilising...")
            my_model_336.set_heater_range(1, 3)
            my_model_336.set_heater_range(2, 3)
            my_model_336.set_control_setpoint(1, setpoint[0])
            my_model_336.set_control_setpoint(2, setpoint[1])
            my_model_336.disconnect_usb()
            del my_model_336
            
            label6_var.set('Correcting temperature...')
            label6.update_idletasks()
            refresh_spec()
            
            global acccount, s2ns, as2ns, then
            acccount = 0
            s2ns = {0: 0}
            as2ns = {0: 0}
            then = time.time() # Reset the timer for the new accumulation
            ax2.clear() # Clear the aSNR plot
            ax2.set_title('aSNR progress') # Redraw the title
            canvas.draw() # Update the canvas to show the cleared plot
            
            root.after(int(float(e1.get()) * 1000), lambda: measurement_loop(loop_count, ns, then))
            
        def _connect_and_correct():
            my_model_336 = Model336(com_port='COM4')
            root.after(200, lambda: _send_commands_and_reschedule(my_model_336))
                    
        _connect_and_correct()
        # acccount=0
        return

    error_count = 0
    

    try:
        data1,now,t,nsdiff =download_and_save(loop_count,ns)
        
        if var3.get() == 1 and len(data1) == 1024:
            time_step = now - then
            
            cal = fold.calibrate(r'{}'.format(calpath))
            vel, data2 = fold.fold(cal, np.array([data1]).astype(np.float64)[0])
            s2n_M = s2n.s2n(data2)
            s2ns[time_step] = s2n_M
            as2n_M = s2n.as2n(data2) / 1024
            as2ns[time_step] = as2n_M
            
            v, s = vel, data2
            ax.clear()
            ax.plot(v, s, linestyle=' ', marker='.', markersize=0.5, color='black')
            ax.set_title('Latest Folded Spectrum', fontsize='x-small')
            ax.text(-11, min(data2) - 0.001 * min(data2), 'SNR: {}, aSNR: {}'.format(round(s2n_M, 1), round(as2n_M, 1)), fontsize='x-small')
            ax.get_yaxis().set_visible(False)
            ax.set_xlabel("Velocity (mm/s)", fontsize='x-small') 
            ax.tick_params(axis='both', which='major', labelsize=6)
            canvas1.figure.tight_layout()
            canvas1.draw()
            
            global acccount
            acccount += 1
            
            print("{}: s2n is {} and as2n is {}".format(acccount, round(s2n_M, 1), round(as2n_M, 1)))
            label6_var.set('Collected {} accumulations, the latest s2n is {} and as2n is {}'.format(acccount, round(s2n_M, 1), round(as2n_M, 1)))
            label6.update_idletasks()
            
            np.save(savepath+'/log', [time_step, acccount, tempB])
            np.save(savepath+'/s2ns', s2ns)
            np.save(savepath+'/as2ns', as2ns)
            
            k, a = s2n.time_curve_params(list(s2ns.keys()), list(s2ns.values()))
            ak, aa = s2n.time_curve_params(list(as2ns.keys()), list(as2ns.values()))
            s2n_i = e2.get()
            
            time_fut = ((float(s2n_i) - aa) / ak) ** 2
            
            minutes, _ = divmod(time_fut, 60)
            hours, minutes = divmod(minutes, 60)
            days, hours = divmod(hours, 24)
            time_str = f"Est. time: {int(days)}d {int(hours)}h {int(minutes)}m"
            ax2.text(0.05, 0.05, time_str, transform=ax2.transAxes, fontsize='x-small')
            
            va, sa = list(as2ns.keys()), list(as2ns.values())
            vb, sb = as2ns.keys(), s2n.time_curve(list(as2ns.keys()), ak, aa)
            if acccount == 1:
                ax2.cla()
                os.remove(savepath + '/s2ns.npy')
                os.remove(savepath + '/as2ns.npy')
            ax2.clear()
            va_p,vb_p=[x/60 for x in va],[y/60 for y in vb]
            ax2.plot(va_p, sa, linestyle=' ', marker='.', markersize=0.5, color='black')
            ax2.plot(vb_p, sb, linestyle='-', color='black')
            ax2.plot(time_fut/60, float(s2n_i), linestyle=' ', marker='x',)
            ax2.set_title('aSNR progress', fontsize='x-small')
            ax2.set_xlabel("Time (min)", fontsize='x-small')
            ax2.tick_params(axis='both', which='major', labelsize=6)
            ax2.set_ylim([-1, float(s2n_i) + 5])
            canvas.figure.tight_layout()
            canvas.draw()
            
            normal_spectrum_saved = False
            t_threshold = float(e2.get())
            pause_temps, pause_asnrs = get_pause_point_data()
        
                                
            if Enhanced_SNR: # only runs if we are in enhanced SNR mode
            
                pause_temps, _ = get_pause_point_data()
                current_setpoint = float(e4.get())
                if current_setpoint in pause_temps: #Check if the current setpoint is still a valid pause point.
                    
                    pause_index = pause_temps.index(current_setpoint)       
                    live_threshold_str = pause_entries_asnr[pause_index].get() #Get the LIVE threshold value from the entry box.
                    t_threshold_ESNRP = float(live_threshold_str) # Update the threshold
                    
                    # Mark the new threshold on the plot for visual feedback
                    k, a = s2n.time_curve_params(list(s2ns.keys()), list(s2ns.values()))
                    ak, aa = s2n.time_curve_params(list(as2ns.keys()), list(as2ns.values()))
                    time_fut = ((t_threshold_ESNRP - aa) / ak) ** 2
                    
                    last_time_step_min = list(as2ns.keys())[-1] / 60
                    ax2.plot(time_fut/60, t_threshold_ESNRP, 'bx', markersize=4)
                    ax2.set_ylim([-1, t_threshold_ESNRP + 5])
                    canvas.draw()
                
                t_threshold_ESNRP_value = t_threshold_ESNRP # This is either SNR or Minutes
                current_duration_seconds = time.time() - then
                current_duration_minutes = current_duration_seconds / 60.0

                trigger_condition_met_esnrp = False
                if threshold_mode.get() == "SNR":
                    if as2n_M > t_threshold_ESNRP_value and (as2n_M < t_threshold_ESNRP_value + 40):
                        trigger_condition_met_esnrp = True
                else: # "Time" mode
                    if current_duration_minutes >= t_threshold_ESNRP_value:
                        trigger_condition_met_esnrp = True
                         
                if trigger_condition_met_esnrp: #test if we have met the threshold
                    
                    data1,now,t,nsdiff=download_and_save(loop_count,ns) # check if it is a glitch that is removed simply by refreshing the spectrum
                    if len(data1) == 1024:
                        cal = fold.calibrate(r'{}'.format(calpath))
                        vel, data2 = fold.fold(cal, np.array([data1]).astype(np.float64)[0])
                        s2n_M = s2n.s2n(data2)
                        s2ns[time_step] = s2n_M
                        as2n_M = s2n.as2n(data2) / 1024
                                        
                       
                        current_duration_seconds = time.time() - then # Recalculate duration
                        current_duration_minutes = current_duration_seconds / 60.0

                        trigger_condition_still_met_esnrp = False
                        if threshold_mode.get() == "SNR":
                            if as2n_M > t_threshold_ESNRP_value and (as2n_M < t_threshold_ESNRP_value + 40):
                                trigger_condition_still_met_esnrp = True
                        else: # "Time" mode
                            if current_duration_minutes >= t_threshold_ESNRP_value:
                                trigger_condition_still_met_esnrp = True
                        

                        if trigger_condition_still_met_esnrp:#test if we still meet the threshold
    
                            try:
                                os.mkdir(os.path.join(savepath, 'high_SNR_spectra'))
                            except FileExistsError:
                                pass
                            
                            duration_seconds = time.time() - then
                            minutes, seconds = divmod(duration_seconds, 60)
                            hours, minutes = divmod(minutes, 60)
                            
                            file_name = '{}{}{}_{}h{}m{}s__{:.0f}__{:.0f}.dat'.format(t[0], t[1], t[2], t[3], t[4], t[5], ns, nsdiff)
                            with open(os.path.join(savepath, file_name), 'r') as csvfile:
                                reader = csv.reader(csvfile)
                                with open(os.path.join(savepath, 'high_SNR_spectra', '{}K_{}h{}m.dat'.format(setpoint[1], hours, minutes)), 'w', newline='') as dest_file:
                                    writer = csv.writer(dest_file)
                                    for row in reader:
                                        writer.writerow(row)
                        
                            print("Final HD spectrum at {}K saved with s2n of {}".format(setpoint[1], s2n_M))
                            
                            was_setpoint = setpoint[1]
                            was_setpoint0 = setpoint[0]
                            Enhanced_SNR = False
                            normal_spectrum_saved = False
                            
                            setpoint_int = [was_setpoint0 - float(e3.get()), was_setpoint - float(e3.get())]
                            print("New setpoint is {} and current temperature is {}".format(setpoint_int[1], tempB))
                            
                            e4.delete(0, tk.END)
                            e4.insert(0, str(setpoint_int[1]))
                            
                            log_event("Enhanced Spectra", "finish", temp=setpoint[1], snr=s2n_M, asnr=as2n_M)
                            
                            ramp_temperature(setpoint_int, loop_count, ns, then)
                            return

        
            elif not normal_spectrum_saved:  # only runs if we are in normal mode
                
                t_threshold_value = float(e2.get()) # This is either SNR or Minutes
                current_duration_seconds = time.time() - then
                current_duration_minutes = current_duration_seconds / 60.0

                trigger_condition_met = False
                if threshold_mode.get() == "SNR":
                    if as2n_M > t_threshold_value and (as2n_M < t_threshold_value + 75):
                        trigger_condition_met = True
                else: # "Time" mode
                    if current_duration_minutes >= t_threshold_value:
                        trigger_condition_met = True

                if trigger_condition_met: #test if we meet the threshold yet 
                    data1,now,t,nsdiff=download_and_save(loop_count,ns) # check if it is a glitch that is removed simply by refreshing the spectrum
                    if len(data1) == 1024:
                        cal = fold.calibrate(r'{}'.format(calpath))
                        vel, data2 = fold.fold(cal, np.array([data1]).astype(np.float64)[0])
                        s2n_M = s2n.s2n(data2)
                        s2ns[time_step] = s2n_M
                        as2n_M = s2n.as2n(data2) / 1024
                        
                        current_duration_seconds = time.time() - then # Recalculate duration
                        current_duration_minutes = current_duration_seconds / 60.0
                        
                        trigger_condition_still_met = False
                        if threshold_mode.get() == "SNR":
                             if as2n_M > t_threshold_value and (as2n_M < t_threshold_value + 75):
                                trigger_condition_still_met = True
                        else: # "Time" mode
                            if current_duration_minutes >= t_threshold_value:
                                trigger_condition_still_met = True

                        if trigger_condition_still_met: #test if we meet the threshold again 
                            
                            try:
                                os.mkdir(os.path.join(savepath, 'profile'))
                            except FileExistsError:
                                pass
                            
                            file_name = '{}{}{}_{}h{}m{}s__{:.0f}__{:.0f}.dat'.format(t[0], t[1], t[2], t[3], t[4], t[5], ns, nsdiff)
                            shutil.copyfile(os.path.join(savepath, file_name), os.path.join(savepath, 'profile', '{}K.dat'.format(setpoint[1])))
                            print("Final spectrum at {}K saved with s2n of {}".format(setpoint[1], s2n_M))
                            
                            normal_spectrum_saved = True
                            
                            # Check if this is an enhanced SNR point
                            if float(e4.get()) in pause_temps:
                                print("Reached enhanced SNR point at {}K".format(e4.get()))
                                pause_index = pause_temps.index(float(e4.get()))           
                                t_threshold_ESNRP_str = pause_entries_asnr[pause_index].get() # Read the CURRENT threshold from the corresponding Entry widget
                                t_threshold_ESNRP = float(t_threshold_ESNRP_str)
                                last_time_step_min = list(as2ns.keys())[-1] / 60
                                ax2.plot(last_time_step_min, t_threshold_ESNRP, 'bx', markersize=4) 
                                canvas.draw()
                                print("Temperature threshold is now {}".format(t_threshold_ESNRP))
                                Enhanced_SNR = True
                                log_event("Normal Spectra", "finish", temp=setpoint[1], snr=s2n_M, asnr=as2n_M)
                                log_event("Enhanced Spectra", "start", temp=setpoint[1])
                            else:
                                # Not an enhanced SNR point, so drop temperature as usual
                                if special_case: #special case where enhanced SNR point is not in main profile
                                    setpoint_int = [was_setpoint0 - float(e3.get()), was_setpoint - float(e3.get())]
                                    print("New setpoint is {} and current temperature is {}".format(setpoint_int[1], tempB)) 
                                    special_case=False
                                if not special_case:
                                    was_setpoint = setpoint[1]
                                    was_setpoint0 = setpoint[0]
                                    setpoint_int = [was_setpoint0 - float(e3.get()), was_setpoint - float(e3.get())]
                                    print("New setpoint is {} and current temperature is {}".format(setpoint_int[1], tempB))
                                    special_case=False
                                
                                for p in pause_temps:
                                    if p>setpoint_int[1] and p<was_setpoint:
                                        setpoint_int = [was_setpoint0 - float(e3.get()), was_setpoint - float(e3.get())]
                                        special_case=True
                                
                        
                                
                                e4.delete(0, tk.END)
                                e4.insert(0, str(setpoint_int[1]))
                                
                                log_event("Normal Spectra", "finish", temp=setpoint[1], snr=s2n_M, asnr=as2n_M)
                                
                                # my_model_336.disconnect_usb()
                                root.after(1000, lambda: ramp_temperature(setpoint_int, loop_count, ns, then))
                                return
                        else:
                            print("we enoundered a glitch, but it was no longer present after refreshing the download")
                    elif len(data1)<1024:
                        print("data length less than expected, trying again")
        
        elif len(data1)<1024:
            print("data length less than expected, trying again")
        
        gc.collect()
        loop_count += 1
        error_count = 0
        ns = now * nano
        ready = time.time()
        save_interval = float(e1.get())
        remaining_time_ms = max(0, int((save_interval - (ready - now)) * 1000))
        root.after(remaining_time_ms, lambda: measurement_loop(loop_count, ns, then))

    except comtypes.COMError:
        sub.Cancel.click()
        error_count += 1
        print('Exception COMError ignored at {}h{}m{}s - check if script is running faster than computer can manage'.format(t[3], t[4], t[5]))

        if error_count == 4:
            clear_button.app['dlg']['CLEAR Channel 1'].click() 
            clear_button.wait('visible enabled', timeout=30)
            root.after(4000, lambda: app['dlg']['OK'].click())
            print('Stop-Start refresh at {}h{}m{}s'.format(t[3], t[4], t[5]))

        print("Continuing to next measurement cycle after COMError.")
        root.after(int(float(e1.get()) * 1000), lambda: measurement_loop(loop_count, ns, then))


def wait_before_first_measurement(loop_count, ns, then, start_time):
    global running
    if not running:
        print("Stopping due to user request.")
        return

    save_interval = float(e1.get())
    
    # Calculate the time that has passed since start_measurement was called
    setup_time = time.time() - start_time
    
    # Calculate the remaining time to wait, ensuring it's not negative
    time_to_wait = max(0, save_interval - setup_time)
    
    print(f"Initial setup took {setup_time:.2f}s.")
    print(f"Waiting for {time_to_wait:.2f}s before first measurement.")
    label6_var.set(f"Waiting for first measurement cycle ({time_to_wait:.0f}s)...")
    label6.update_idletasks()

    # Schedule the next call to measurement_loop after the calculated delay.
    root.after(int(time_to_wait * 1000), lambda: measurement_loop(loop_count, ns, then))


#%%

def start_measurement():
    global acccount, s2ns, as2ns, then, running, Enhanced_SNR, normal_spectrum_saved, profile_start_time, LOG_SAVE_PATH
    parent_directory = os.path.dirname(savepath)
    LOG_SAVE_PATH = parent_directory
    init_logger() #initialise the logger
    
    label7_var.set(f"Profile started at: {e4.get()} K")
    
    # Capture the start time as early as possible
    start_time = time.time()
    profile_start_time = start_time
    temp_timestamps.clear()
    temp_A_values.clear()
    temp_B_values.clear()
    acccount=0
    running = True
    Enhanced_SNR=False
    
    label6_var.set('Setting up measurement')
    label6.update_idletasks()
    
    shutil.rmtree(r'C:\W302_data\log', ignore_errors=True)
    os.makedirs(r'C:\W302_data\log', exist_ok=True)
    shutil.rmtree(r'C:\W302_data\W302sn086_1', ignore_errors=True)
    os.makedirs(r'C:\W302_data\W302sn086_1', exist_ok=True)
    
#    try:
#        app['dlg']['Start'].click()
#    except:
#        app['dlg']['Stop'].click()
#        app['dlg']['Start'].click()
    clear()

    t = time.localtime(start_time)
    print('Collection commenced at {}h{}m{}s on {}/{}'.format(t[3], t[4], t[5], t[2], t[1]))
    
    try:
        then = float(np.load(savepath+'/first_collection.npy'))
        s2ns = np.load(savepath+'/s2ns.npy', allow_pickle='TRUE').item()
        as2ns = np.load(savepath+'/as2ns.npy', allow_pickle='TRUE').item()
        log_file = np.load(savepath+'/log.npy', allow_pickle='TRUE').item()
        tt = time.localtime(then)
        print('first collection was at {}h{}m{}s on {}/{}'.format(tt[3], tt[4], tt[5], tt[2], tt[1]))
    except:
        s2ns = {0: 0}
        as2ns = {0: 0}
        then = start_time
        np.save(savepath + '/first_collection.npy', then)
        
    np.save(savepath+'/first_collection', then)
    
    loop_count, data1, nano = 1, np.zeros(1024), 1000000000
    
    wait_before_first_measurement(loop_count, start_time * nano, then, start_time)
    
    update_temp_plot()
 

 
#%%
# GUI set-up

# Main Window Setup
root = tk.Tk()
threshold_mode = tk.StringVar(value="SNR") # Default to SNR mode
root.title('Temperature Profile Controller')
root.geometry('1000x800')  # Adjusted for a two-column layout
background_color = '#001f3f'
foreground_color = 'white'
root.configure(bg=background_color)

# Style Configuration
style = ttk.Style()
style.configure('TFrame', background=background_color)
style.configure('TSeparator', background=foreground_color)
style.configure('TLabel', background=background_color, foreground=foreground_color)
style.configure('TCheckbutton', background=background_color, foreground=foreground_color)
style.configure('TButton', relief='raised')

# Main Column Frames 
left_column_frame = ttk.Frame(root, style='TFrame')
left_column_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5), pady=10)

right_column_frame = ttk.Frame(root, style='TFrame')
right_column_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 10), pady=10)

# =============================================================================
# --- Left Column (Controls) ---
# =============================================================================

# --- File Options Frame ---
file_options_frame = ttk.Frame(left_column_frame, padding=10, style='TFrame')
file_options_frame.pack(fill='x', pady=(10, 2))

# Frame for Save/Load buttons ---
project_button_frame = ttk.Frame(file_options_frame, style='TFrame')
project_button_frame.pack(fill='x', pady=(0, 2))

save_project_button = tk.Button(project_button_frame, text='Save Project', command=save_project, width=15)
save_project_button.pack(side=tk.LEFT, fill='x', expand=True, padx=(0, 2), pady=(2, 0))

load_project_button = tk.Button(project_button_frame, text='Load Project', command=load_project, width=15)
load_project_button.pack(side=tk.LEFT, fill='x', expand=True, padx=(2, 0), pady=(2, 0))

open_save_button = tk.Button(file_options_frame, text='Open save location', command=opensaveloc)
open_save_button.pack(fill='x', pady=(2, 0))
label1 = ttk.Label(file_options_frame, text='location...', anchor='w')
label1.pack(fill='x', pady=(0, 2))

open_calloc_button = tk.Button(file_options_frame, text='Open calibration in .dat format', command=opencalloc)
open_calloc_button.pack(fill='x', pady=(2, 0))
label2 = ttk.Label(file_options_frame, text='location...', anchor='w')
label2.pack(fill='x', pady=(0, 0))

separator_file = ttk.Separator(left_column_frame, orient='horizontal')
separator_file.pack(fill='x', padx=10, pady=5)

# Temperature Profile Settings Frame
temp_settings_frame = ttk.Frame(left_column_frame, padding=10, style='TFrame')
temp_settings_frame.pack(fill='x', padx=10, pady=(2, 2))

temp_label_frame = ttk.Frame(temp_settings_frame, style='TFrame')
temp_label_frame.pack(fill='x')
temp_group_label = ttk.Label(temp_label_frame, text='Temperature Profile Settings:', font=('TkDefaultFont', 10, 'bold'), anchor='w')
temp_group_label.pack(fill='x', pady=(0, 2))

temp_row1_frame = ttk.Frame(temp_settings_frame, style='TFrame')
temp_row1_frame.pack(fill='x')

l1 = ttk.Label(temp_row1_frame, text='Refresh interval:', width=15, anchor='w')
l1.pack(side=tk.LEFT, pady=(2, 0), padx=(0, 2))
e1 = tk.Entry(temp_row1_frame, width=5)
e1.insert(0, '60')
e1.pack(side=tk.LEFT, pady=(0, 2), padx=(0, 2))
l1_unit = ttk.Label(temp_row1_frame, text='s', width=2)
l1_unit.pack(side=tk.LEFT, padx=(0, 15))

l4 = ttk.Label(temp_row1_frame, text='Setpoint temp:', width=16, anchor='w')
l4.pack(side=tk.LEFT, pady=(2, 0), padx=(0, 2))
e4 = tk.Entry(temp_row1_frame, width=5)
e4.insert(0, '5')
e4.pack(side=tk.LEFT, pady=(0, 2), padx=(0, 2))
l4_unit = ttk.Label(temp_row1_frame, text='K', width=2)
l4_unit.pack(side=tk.LEFT, pady=(0, 0), padx=(0, 0))

temp_row2_frame = ttk.Frame(temp_settings_frame, style='TFrame')
temp_row2_frame.pack(fill='x')

l3 = ttk.Label(temp_row2_frame, text='Temperature interval:', width=20, anchor='w')
l3.pack(side=tk.LEFT, pady=(2, 0), padx=(0, 2))
e3 = tk.Entry(temp_row2_frame, width=5)
e3.insert(0, '1')
e3.pack(side=tk.LEFT, pady=(0, 2), padx=(0, 2))
l3_unit = ttk.Label(temp_row2_frame, text='K', width=2)
l3_unit.pack(side=tk.LEFT, padx=(0, 15))

l2 = ttk.Label(temp_row2_frame, text='Target aSNR:', width=12, anchor='w')
l2.pack(side=tk.LEFT, pady=(2, 0), padx=(0, 2))
e2 = tk.Entry(temp_row2_frame, width=5)
e2.insert(0, '100')
e2.pack(side=tk.LEFT, pady=(0, 2), padx=(0, 2))

temp_row3_frame = ttk.Frame(temp_settings_frame, style='TFrame')
temp_row3_frame.pack(fill='x', pady=(5,0))

toggle_button = tk.Button(temp_row3_frame, text='Mode: aSNR', width=15, command=toggle_threshold_mode)
toggle_button.pack(side=tk.LEFT, padx=(0, 5))

l_mode_desc = ttk.Label(temp_row3_frame, text="<-- Toggle threshold mode (aSNR or Time)")
l_mode_desc.pack(side=tk.LEFT, anchor='w')

separator_temp = ttk.Separator(left_column_frame, orient='horizontal')
separator_temp.pack(fill='x', padx=10, pady=5)

# --- Pause Points Frame ---
pause_points_frame = ttk.Frame(left_column_frame, padding=10, style='TFrame')
pause_points_frame.pack(fill='x', padx=10, pady=(2, 2))

pause_label_frame = ttk.Frame(pause_points_frame, style='TFrame')
pause_label_frame.pack(fill='x')
pause_points_group_label = ttk.Label(pause_label_frame, text='Enhanced SNR Measurements (Optional):', font=('TkDefaultFont', 10, 'bold'), anchor='w')
pause_points_group_label.pack(fill='x', pady=(0, 2))

def create_pause_point_row(parent_frame, point_number):
    pause_row_frame = ttk.Frame(parent_frame, style='TFrame')
    pause_row_frame.pack(fill='x', pady=(1, 1))
    check_var = tk.IntVar(value=0)
    checkbox = ttk.Checkbutton(pause_row_frame, variable=check_var)
    checkbox.pack(side=tk.LEFT, padx=(0, 2))
    label = ttk.Label(pause_row_frame, text=f'Enhanced SNR Point {point_number}:', width=23, anchor='w')
    label.pack(side=tk.LEFT, padx=(0, 1))
    entry_temp = tk.Entry(pause_row_frame, width=5)
    entry_temp.insert(0, '1')
    entry_temp.pack(side=tk.LEFT, padx=(0, 1))
    k_label = ttk.Label(pause_row_frame, text='K', width=0.5)
    k_label.pack(side=tk.LEFT, padx=(0, 2))
    asnr_label = ttk.Label(pause_row_frame, text='aSNR:', width=10, anchor='w') # Changed width to 10
    asnr_label.pack(side=tk.LEFT, padx=(30, 1)) # Changed padx
    entry_asnr = tk.Entry(pause_row_frame, width=5)
    entry_asnr.insert(0, '400')
    entry_asnr.pack(side=tk.LEFT)
    return check_var, entry_temp, entry_asnr, asnr_label # Return the label

pause_vars = []
pause_entries_temp = []
pause_entries_asnr = []
# pause_asnr_labels is already a global list

for i in range(1, 16):
    check_var, entry_temp, entry_asnr, asnr_label = create_pause_point_row(pause_points_frame, i)
    pause_vars.append(check_var)
    pause_entries_temp.append(entry_temp)
    pause_entries_asnr.append(entry_asnr)
    pause_asnr_labels.append(asnr_label) # Store the label reference

# --- Button Frame ---
button_frame = ttk.Frame(left_column_frame, padding=10, style='TFrame')
button_frame.pack(fill='x', padx=10, pady=20, side=tk.BOTTOM)

close_button = tk.Button(button_frame, width=10, text='Close', command=on_closing)
close_button.pack(side=tk.RIGHT, padx=(5, 0))

clear_button = tk.Button(button_frame, width=10, text=' Clear Data ', command=clear)
clear_button.pack(side=tk.RIGHT, padx=(5, 0))

save_button = tk.Button(button_frame, width=10, text=' Start ', command=start_measurement)
save_button.pack(side=tk.RIGHT, padx=(5, 0))

plot_button = tk.Button(button_frame, width=10, text='Plot preview', command=create_temp_plot_window)
plot_button.pack(side=tk.RIGHT, padx=(5, 0))


# =============================================================================
# --- Right Column (Monitoring) ---
# =============================================================================

# --- Status Frame (Packed to the BOTTOM first) ---
status_frame = ttk.Frame(right_column_frame, padding=10, style='TFrame')
status_frame.pack(side=tk.BOTTOM, fill='x', padx=10, pady=(2, 2))

label5_var = tk.StringVar()
label5_var.set("Current temperature: not yet available")
label5 = ttk.Label(status_frame, textvariable=label5_var, anchor='w')
label5.pack(fill='x', pady=(1, 1))

label6_var = tk.StringVar()
label6_var.set("Status: not yet running")
label6 = ttk.Label(status_frame, textvariable=label6_var, anchor='w')
label6.pack(fill='x', pady=(1, 2))

label7_var = tk.StringVar()
label7_var.set("Starting temperature: TBC")
label7 = ttk.Label(status_frame, textvariable=label7_var, anchor='w')
label7.pack(fill='x', pady=(1, 2))

# --- Output Text Frame (Packed to the BOTTOM, above status) ---
output_frame = ttk.Frame(right_column_frame, padding=10, style='TFrame')
output_frame.pack(side=tk.BOTTOM, fill='x', padx=10, pady=(2, 2))

output_label = ttk.Label(output_frame, text='Output Messages:', font=('TkDefaultFont', 10, 'bold'), anchor='w')
output_label.pack(fill='x', pady=(0, 2))

output_text = tk.Text(output_frame, wrap="word", height=5, state='disabled', bg=background_color, fg=foreground_color)
output_text.pack(fill='both', expand=True)
output_scrollbar = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=output_text.yview)
output_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
output_text.config(yscrollcommand=output_scrollbar.set)

# --- Spectrum Analysis Frame (Packed at the TOP, fills remaining space) ---
spectrum_frame = ttk.Frame(right_column_frame, padding=10, style='TFrame')
spectrum_frame.pack(side=tk.TOP, fill='both', expand=True, padx=10, pady=(2, 2))

var3 = tk.IntVar(value=1)
c3 = ttk.Checkbutton(spectrum_frame, text="Start analysing spectrum", variable=var3)
c3.pack(anchor='w', pady=(0, 5))

plots_frame = ttk.Frame(spectrum_frame, style='TFrame')
plots_frame.pack(fill='both', expand=True)

# create left and right frames for plots
left_plot_frame = ttk.Frame(plots_frame)
left_plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

right_plot_frame = ttk.Frame(plots_frame)
right_plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

# Plot 1: Folded Spectrum
fig = Figure(figsize=(2.0,4), dpi=100)
ax = fig.add_subplot(111)
ax.set_title('Folded Spectrum', color='white', fontsize=10)
# ... additional plot setup ...
canvas1 = FigureCanvasTkAgg(fig, master=left_plot_frame)
canvas1.draw()
canvas1.get_tk_widget().pack(side=tk.TOP, pady=(0, 5), fill=tk.BOTH, expand=1)

# Plot 2: aSNR progress
fig2 = Figure(figsize=(2.0,1.2), dpi=100)
ax2 = fig2.add_subplot(111)
ax2.set_title('aSNR progress', color='white', fontsize=10)
# ... additional plot setup ...
canvas = FigureCanvasTkAgg(fig2, master=right_plot_frame)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, pady=(0, 5), fill=tk.BOTH, expand=1)

# Plot 3: Temperature Trace
fig3 = Figure(figsize=(2.0,1.2), dpi=100, constrained_layout=True)
ax3 = fig3.add_subplot(111)
ax3.set_title('Temperature Trace', color='white', fontsize=10)
# ... additional plot setup ...
canvas3 = FigureCanvasTkAgg(fig3, master=right_plot_frame)
canvas3.draw()
canvas3.get_tk_widget().pack(side=tk.TOP, pady=(0, 5), fill=tk.BOTH, expand=1)

# --- Text Redirector Class ---
class TextRedirector(object):
    def __init__(self, widget):
        self.widget = widget
    def write(self, text):
        self.widget.config(state=tk.NORMAL)
        self.widget.insert(tk.END, text)
        self.widget.see(tk.END)
        self.widget.config(state=tk.DISABLED)
    def flush(self):
        pass

# --- Redirect stdout to the Text widget ---
sys.stdout = TextRedirector(output_text)

# Connect the on_closing function to the window's 'X' button.
root.protocol("WM_DELETE_WINDOW", on_closing)    

# --- Start the main GUI loop ---
root.mainloop()