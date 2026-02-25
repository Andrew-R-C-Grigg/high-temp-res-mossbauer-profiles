# -*- coding: utf-8 -*-
"""

Nussbaum - a python package for automated high-resolution Mössbauer spectroscopy temperature profile measurements

@author: Andrew R. C. Grigg*, James M Byrne, Ruben Kretzschmar
*ETH Zurich, Department of Environmental System Science, Institute for Biogeochemistry and Pollutant Dynamics

License: Apache 2.0 (http://www.apache.org/licenses/)

This opens a GUI that allows the user to select data to be included into 
a 2D plot of a temperature profile.

"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import re
from nussbaum.utils import fold as fold
from nussbaum.utils import s2n as s2n
import matplotlib.animation as animation

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


if len(sys.argv) >= 3:
    # Use the paths provided by the main script
    folder_to_plot = sys.argv[1]
    calibration_file = sys.argv[2]
    print(f"Received folder to plot: {folder_to_plot}")
    print(f"Received calibration file: {calibration_file}")
else:
    # Fallback if the script is run by itself without arguments
    folder_to_plot = "."  # Default to the current directory
    calibration_file = "" # Default to no calibration file
    print("No paths provided on launch. Please select them manually.")

file_paths = folder_to_plot
calfile_path = calibration_file


def load_files(file_paths, calfile_path):
    """
    Loads selected .dat files and a calibration file, then processes them.
    
    Args:
        file_paths (list): A list of full file paths to the .dat spectra files.
        calfile_path (str): The full file path to the calibration file.

    Returns:
        tuple: A tuple containing the processed data dictionary and the velocity array.
    """
    file_dict = {}
    number_pattern = r"(\d+\.\d+)"
    
    try:
        cal = fold.calibrate(calfile_path)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load calibration file: {e}")
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
                    
                    # Find background value for velocity domain spectrum to normalise each spectrum
                    bkg = s2n.bkg(folded)
                    
                    file_dict[key] = folded / bkg
                    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process file {os.path.basename(filename)}: {e}")
            continue

    if not file_dict:
        return None, None
        
    # Assuming 'x' (velocity) is the same for all files.
    velocity = x
    return file_dict, velocity


# plotting functions
def plot_data_with_velocity(file_dict, velocity, title, cmap, vmin, vmax):
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_keys = sorted(file_dict.keys(), key=lambda x: float(x))
    sorted_keys_float = [float(key) for key in sorted_keys]
    
    all_data_values = [float(value) for key in sorted_keys for value in file_dict[key]]
    
    grid_data = []
    for key in sorted_keys:
        data_values = [float(value) for value in file_dict[key]]
        grid_data.append(data_values)
        
    grid_data = np.array(grid_data)
    x, y = np.meshgrid(velocity, sorted_keys_float)
    
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
    fig.colorbar(mesh, ax=ax, label='counts normalised to background')
    ax.grid(True, linestyle=' ', alpha=0.7)
    plt.show()
    return fig

def plot_data_3d_animated(file_dict, velocity, title, cmap, elev, azim, animation_filename='temperature_velocity_matrix_3d_animation.gif'):
    """
    Generates a 3D surface plot of Mössbauer spectra and creates an animation.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    sorted_keys = sorted(file_dict.keys(), key=lambda x: float(x))
    sorted_keys_float = [float(key) for key in sorted_keys]
    X = np.array(velocity)
    Y = np.array(sorted_keys_float)
    Z = []
    for key in sorted_keys:
        data_values = [float(value) for value in file_dict[key]]
        Z.append(data_values)
    Z = np.array(Z)
    X_mesh, Y_mesh = np.meshgrid(X, Y)
    surf = ax.plot_surface(X_mesh, Y_mesh, Z, cmap=cmap, edgecolor='none')
    ax.set_xlabel('Velocity (mm/s)')
    ax.set_ylabel('Temperature (K)')
    ax.set_zlabel('Counts Normalised to Background')
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='counts normalised to background')
    fixed_elev = elev
    start_azim = azim
    end_azim = azim + 50
    num_animation_frames = 90
    azimuth_sequence = np.zeros(num_animation_frames)
    for i in range(num_animation_frames):
        progress = i / (num_animation_frames - 1)
        normalized_triangle_wave = 1 - abs(2 * progress - 1)
        azimuth_sequence[i] = start_azim + (end_azim - start_azim) * normalized_triangle_wave
    
    def update_angle(frame):
        ax.view_init(elev=fixed_elev, azim=azimuth_sequence[frame])
        return fig,
    
    anim = animation.FuncAnimation(fig, update_angle, frames=num_animation_frames, interval=100, blit=False)
    print(f"Saving animation to {animation_filename}...")
    try:
        anim.save(animation_filename, writer='pillow', fps=10)
        print("Animation saved successfully!")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Please ensure you have the necessary writer installed (e.g., 'pillow' for GIF, 'ffmpeg' for MP4).")
    return fig, ax

def plot_offset_spectra_3d(file_dict, velocity, title, elev, azim, temp_spacing_factor=1.0):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    sorted_keys = sorted(file_dict.keys(), key=lambda x: float(x))
    sorted_keys_float = [float(key) for key in sorted_keys]
    min_temp = sorted_keys_float[0]
    max_temp = sorted_keys_float[-1]
    effective_y_positions = [(t - min_temp) * temp_spacing_factor + min_temp for t in sorted_keys_float]
    temp_to_effective_y_map = {sorted_keys_float[i]: effective_y_positions[i] for i in range(len(sorted_keys_float))}
    gray_cmap = plt.cm.Greys
    for i, key in enumerate(sorted_keys):
        temp_value = float(key)
        data_values = np.array([float(value) for value in file_dict[key]])
        y_pos_for_plot = temp_to_effective_y_map[temp_value]
        normalized_temp = (temp_value - min_temp) / (max_temp - min_temp)
        ax.plot(velocity, np.full_like(velocity, y_pos_for_plot), data_values,
                color=gray_cmap(normalized_temp),
                alpha=0.8,
                linewidth=0.5,
                label=f'{temp_value} K')
    ax.set_xlabel('Velocity (mm/s)')
    ax.set_ylabel('Temperature (K)')
    ax.set_zlabel('Counts Normalised to Background')
    ax.set_title(title)
    ax.set_ylim(min(effective_y_positions), max(effective_y_positions))
    ax.set_yticks(effective_y_positions)
    ax.set_yticklabels([f'{t:.0f}' for t in sorted_keys_float])
    ax.view_init(elev=elev, azim=azim)
    plt.show()
    return fig, ax


# Main GUI Application Class
class MossbauerApp:
    def __init__(self, master):
        self.master = master
        master.title("Mössbauer Spectrum Plotter")
        
        self.spectra_files = []
        self.cal_file = ""
        self.spectra_dict = None
        self.velocity = None
        
        # Define some common colormaps
        self.colormaps = sorted(['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'])

        # Create a frame for the file selection buttons
        self.file_frame = tk.LabelFrame(master, text="File Selection", padx=10, pady=10)
        self.file_frame.pack(padx=10, pady=10, fill="x")
        
        self.cal_button = tk.Button(self.file_frame, text="Select Calibration File", command=self.select_cal_file)
        self.cal_button.pack(pady=5)
        
        self.cal_label = tk.Label(self.file_frame, text="No calibration file selected.")
        self.cal_label.pack()
        
        self.spectra_button = tk.Button(self.file_frame, text="Select Spectra Files", command=self.select_spectra_files)
        self.spectra_button.pack(pady=5)
        
        self.spectra_label = tk.Label(self.file_frame, text="No spectra files selected.")
        self.spectra_label.pack()

        # --- Plot Customization Section ---
        self.customization_frame = tk.LabelFrame(master, text="Plot Customization", padx=10, pady=10)
        self.customization_frame.pack(padx=10, pady=10, fill="x")
        
        # Plot Title
        tk.Label(self.customization_frame, text="Plot Title:").pack(pady=(5,0))
        self.title_entry = tk.Entry(self.customization_frame, width=50)
        self.title_entry.insert(0, 'Mössbauer Spectra: Temperature-Velocity Matrix')
        self.title_entry.pack()

        # Color Scheme Dropdown
        tk.Label(self.customization_frame, text="Color Scheme:").pack(pady=(5,0))
        self.cmap_var = tk.StringVar(self.master)
        self.cmap_var.set('viridis') # default value
        self.cmap_dropdown = ttk.Combobox(self.customization_frame, textvariable=self.cmap_var, values=self.colormaps, state="readonly")
        self.cmap_dropdown.pack()

        # Color Map Saturation Values
        saturation_frame = tk.Frame(self.customization_frame)
        saturation_frame.pack(pady=5)
        
        tk.Label(saturation_frame, text="Min Saturation (vmin):").pack(side="left", padx=(0, 5))
        self.vmin_entry = tk.Entry(saturation_frame, width=5)
        self.vmin_entry.insert(0, '0.88')
        self.vmin_entry.pack(side="left")

        tk.Label(saturation_frame, text="Max Saturation (vmax):").pack(side="left", padx=(10, 5))
        self.vmax_entry = tk.Entry(saturation_frame, width=5)
        self.vmax_entry.insert(0, '1.01')
        self.vmax_entry.pack(side="left")

        # 3D Plot Angles
        angles_frame = tk.Frame(self.customization_frame)
        angles_frame.pack(pady=5)
        
        tk.Label(angles_frame, text="Elevation:").pack(side="left", padx=(0, 5))
        self.elev_entry = tk.Entry(angles_frame, width=5)
        self.elev_entry.insert(0, '15')
        self.elev_entry.pack(side="left")

        tk.Label(angles_frame, text="Azimuth:").pack(side="left", padx=(10, 5))
        self.azim_entry = tk.Entry(angles_frame, width=5)
        self.azim_entry.insert(0, '250')
        self.azim_entry.pack(side="left")
        
        # --- End of Plot Customization Section ---

        # Create a frame for the plotting buttons
        self.plot_frame = tk.LabelFrame(master, text="Plotting Options", padx=10, pady=10)
        self.plot_frame.pack(padx=10, pady=10, fill="x")
        
        self.plot_2d_button = tk.Button(self.plot_frame, text="Plot 2D Matrix", command=self.run_plot_2d)
        self.plot_2d_button.pack(pady=5)
        
        self.plot_3d_button = tk.Button(self.plot_frame, text="Plot 3D Surface", command=self.run_plot_3d)
        self.plot_3d_button.pack(pady=5)
        
        self.plot_offset_button = tk.Button(self.plot_frame, text="Plot Offset Spectra", command=self.run_plot_offset)
        self.plot_offset_button.pack(pady=5)
        
        self.plot_animated_button = tk.Button(self.plot_frame, text="Generate 3D Animation", command=self.run_plot_animated)
        self.plot_animated_button.pack(pady=5)

    def select_cal_file(self):
        """Opens a dialog to select a single calibration file."""
        file_path = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select a Calibration File",
            filetypes=(("DAT files", "*.dat"), ("All files", "*.*"))
        )
        if file_path:
            self.cal_file = file_path
            self.cal_label.config(text=f"Calibration File: {os.path.basename(self.cal_file)}")
            self.master.update_idletasks()
            self.process_files()

    def select_spectra_files(self):
        """Opens a dialog to select multiple spectra files."""
        file_paths = filedialog.askopenfilenames(
            initialdir=os.getcwd(),
            title="Select Spectra Files",
            filetypes=(("DAT files", "*.dat"), ("All files", "*.*"))
        )
        if file_paths:
            self.spectra_files = list(file_paths)
            self.spectra_label.config(text=f"{len(self.spectra_files)} files selected.")
            self.master.update_idletasks()
            self.process_files()

    def show_processing_dialogue(self):
        """Creates and displays a non-blocking processing dialogue."""
        self.processing_dialogue = tk.Toplevel(self.master)
        self.processing_dialogue.title("Processing")
        
        # Make the window modal and non-resizable
        self.processing_dialogue.grab_set()
        self.processing_dialogue.resizable(False, False)
        
        tk.Label(self.processing_dialogue, text="Loading and processing files. This may take a moment.", padx=20, pady=20).pack()

        # Center the dialogue on the main window
        self.processing_dialogue.update_idletasks()
        width = self.processing_dialogue.winfo_width()
        height = self.processing_dialogue.winfo_height()
        x = self.master.winfo_x() + (self.master.winfo_width() // 2) - (width // 2)
        y = self.master.winfo_y() + (self.master.winfo_height() // 2) - (height // 2)
        self.processing_dialogue.geometry(f"{width}x{height}+{x}+{y}")
        
    def hide_processing_dialogue(self):
        """Hides and destroys the processing dialogue."""
        if hasattr(self, 'processing_dialogue') and self.processing_dialogue.winfo_exists():
            self.processing_dialogue.destroy()

    def process_files(self):
        """Loads and processes files after both have been selected."""
        if self.spectra_files and self.cal_file:
            self.show_processing_dialogue()
            
            # Use after() to allow the GUI to update and show the dialogue before processing
            self.master.after(100, self._actual_processing)
            
    def _actual_processing(self):
        """The actual processing logic, called after a small delay."""
        self.spectra_dict, self.velocity = load_files(self.spectra_files, self.cal_file)
        
        # Hide the processing dialogue once done
        self.hide_processing_dialogue()

        if self.spectra_dict and self.velocity is not None:
            messagebox.showinfo("Ready", "Files processed successfully. Ready to plot!")
        else:
            messagebox.showerror("Error", "Failed to process files. Please check the file formats.")

    def run_plot_2d(self):
        """Executes the 2D matrix plot."""
        if self.spectra_dict and self.velocity is not None:
            try:
                title = self.title_entry.get()
                cmap = self.cmap_var.get()
                vmin = float(self.vmin_entry.get())
                vmax = float(self.vmax_entry.get())
                fig = plot_data_with_velocity(self.spectra_dict, self.velocity, title, cmap, vmin, vmax)
                save_path = filedialog.asksaveasfilename(defaultextension=".png", 
                                                         filetypes=[("PNG files", "*.png")], 
                                                         initialfile='temperature-velocity_matrix.png')
                if save_path:
                    fig.savefig(save_path, dpi=300, bbox_inches='tight')
                    messagebox.showinfo("Saved", f"2D plot saved to {save_path}")
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid numbers for vmin and vmax.")
        else:
            messagebox.showwarning("Warning", "Please select and process files first.")

    def run_plot_3d(self):
        """Executes the static 3D plot."""
        if self.spectra_dict and self.velocity is not None:
            title = self.title_entry.get()
            elev = int(self.elev_entry.get())
            azim = int(self.azim_entry.get())
            # Reusing the animated function's core but saving a static image
            fig, ax = plot_data_3d_animated(self.spectra_dict, self.velocity, title, self.cmap_var.get(), elev, azim, animation_filename='_temp_anim.gif')
            
            # Use a static view for the static save
            ax.view_init(elev=elev, azim=azim)
            
            save_path = filedialog.asksaveasfilename(defaultextension=".png", 
                                                     filetypes=[("PNG files", "*.png")], 
                                                     initialfile='temperature-velocity_matrix_3d_static.png')
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Saved", f"3D static plot saved to {save_path}")
            
            # Clean up the temporary animated plot window
            # plt.close(fig)
            try:
                os.remove('_temp_anim.gif')
            except OSError:
                pass
        else:
            messagebox.showwarning("Warning", "Please select and process files first.")
            
    def run_plot_animated(self):
        """Executes the 3D animated plot and saves the GIF."""
        if self.spectra_dict and self.velocity is not None:
            title = self.title_entry.get()
            cmap = self.cmap_var.get()
            elev = int(self.elev_entry.get())
            azim = int(self.azim_entry.get())
            save_path = filedialog.asksaveasfilename(defaultextension=".gif", 
                                                     filetypes=[("GIF files", "*.gif")], 
                                                     initialfile='temperature_velocity_matrix_3d_animation.gif')
            if save_path:
                fig, ax = plot_data_3d_animated(self.spectra_dict, self.velocity, title, cmap, elev, azim, animation_filename=save_path)
                messagebox.showinfo("Complete", f"3D animation saved to {save_path}")
                # You might need to close the plot window after the animation is done
                # or let it stay open for a live preview
                # plt.close(fig)
        else:
            messagebox.showwarning("Warning", "Please select and process files first.")

    def run_plot_offset(self):
        """Executes the 3D offset spectra plot."""
        if self.spectra_dict and self.velocity is not None:
            title = self.title_entry.get()
            elev = int(self.elev_entry.get())
            azim = int(self.azim_entry.get())
            fig, ax = plot_offset_spectra_3d(self.spectra_dict, self.velocity, title, elev, azim, temp_spacing_factor=2.0)
            save_path = filedialog.asksaveasfilename(defaultextension=".png", 
                                                     filetypes=[("PNG files", "*.png")], 
                                                     initialfile='offset_spectra_3d.png')
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Saved", f"Offset spectra plot saved to {save_path}")
        else:
            messagebox.showwarning("Warning", "Please select and process files first.")
            
def main():
    root = tk.Tk()
    app = MossbauerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
