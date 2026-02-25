Nussbaum: Automated Acquisition, Processing & Fitting of High-Resolution Mössbauer Spectroscopy

Developed at: 
    Department of Environmental Systems Science 
    Institute of Biogeochemistry and Pollutant Dynamics
    ETH Zurich, Switzerland.

Available from Github in the 'high-temp-res-mossbauer-profiles' repository (https://github.com/Andrew-R-C-Grigg/high-temp-res-mossbauer-profiles

Publication to cite: 
    Andrew R. C. Grigg, James M. Byrne, Ruben Kretzschmar (in preparation) Collection and analysis of high temperature-resolution 57Fe Mössbauer spectroscopy for detailed iron mineral characterisation.


Overview

Nussbaum is a Python package designed for the entire lifecycle of high temperature-resolution Mössbauer temperature profile experiments. It automates hardware control for automated profile measurements and provides a mathematical framework for data reduction (folding/calibration) and physics-based modelling simultaneously in energy and temperature domains.

Core Components & Utility Modules

1. Automated Acquisition & Hardware Control

    Autosave_temp_profile_Wissoft.py or Autosave_temp_profile_Wissoft.py: The main GUI applications. They interface with Wissel Wissoft or SEE CO software to download data and use the lakeshore library to command Model 335 or 336 controllers.

    Logic-Driven Ramping: Instead of fixed time intervals, the system calculates the adjusted Signal-to-Noise Ratio (aSNR) in real-time. Once a statistical threshold is met, the program automatically increments the temperature to the next setpoint.

2. Data Processing (utils/fold.py & utils/s2n.py)

    Calibration: The calibrate function identifies peaks in standard reference spectra (e.g., α-Fe) to map hardware channels to physical velocity (mm/s).

    Folding: Converts raw drive data into folded spectra using linear regression and interpolation to ensure zero-velocity alignment.

    SNR Metrics:

        s2n: Standard signal-to-noise calculation.

        as2n: Adjusted SNR which accounts for background levels, used as the primary trigger for temperature stepping.

3. Modelling of high temperature resolution profiles (utils/curve.py)

    The curve module provides the physical backbone for fitting complex mineralogical samples:

    Voigt-based Profiles: Uses numerical approximations for Voigt-based peaks to account for both Lorentzian (lifetime) and Gaussian (site distribution) broadening.

    Physical Models:

        Magnetic Sextets & Quadrupole Doublets.

        Debye Temperature (ThD​): Modeling the Second-Order Doppler Shift (SODS) and Recoilless Fraction across temperature profiles.

        Blocking Temperature: Fitting superparamagnetic behavior in nanoparticles.

    Parallel Processing: Utilizes joblib for high-performance fitting of large data matrices.

4. Visualization & Analysis

    Matrix_plotting.py: A GUI for generating 2D intensity maps (heatmaps) and 3D stacked "offset" plots of temperature profiles.

    Temperature_profile_fitting.py: A "steering" script that performs simultaneous global fits across all measured temperatures, ensuring that parameters like the Debye temperature remain physically consistent across the entire dataset.

5. Reproducing Figures

The scripts used to generate the figures for the accompanying manuscript are located in the /Scripts directory. These scripts utilise the nussbaum library and the raw data provided in /Data. 

Installation & Requirements

Dependencies

    Core: numpy, scipy, matplotlib, pandas

    Hardware/UI: pywinauto, comtypes, pyserial, tkinter

    Fitting/Optimisation: joblib, sklearn, tqdm

Directory Structure

To ensure the scripts run correctly, maintain the following structure:
Plaintext

nussbaum_project/
├── LICENSE                # 
├── DATA_LICENSE           # CC-BY 4.0 (Data License)
├── README.md              # Project documentation
├── pyproject.toml         # Package metadata and dependencies
├── Data/                  # Research data files (CC-BY 4.0)
    ├── 2L-Fh/
    ├── 6L-Fh/
    ├── 2Lc-Fh/
    ├── Fh-Gt mix/
├── Scripts/               # Scripts for reproducing Figures (Apache 2.0)
    ├── Figure1.py
    ├── Figure2.py
    ├── Figure3.py
    ├── Figure4_(Temperature_profile_fitting).py
    ├── FigureS5.py
└── Nussbaum/              # Source code directory (Apache 2.0)
    ├── __init__.py
    ├── Autosave_temp_profile_Wissoft.py
    ├── Autosave_temp_profile_SEECo.py
    ├── Matrix_plotting.py
    └── utils/
        ├── __init__.py
        ├── fold.py        # Calibration and folding logic
        ├── s2n.py         # Signal-to-Noise metrics
        └── curve.py       # fitting line shapes

Usage

    Calibrate: Ensure you have a .dat (or .txt) calibration file.

    Acquire: Run the Autosave script, connect your Lake Shore controller, and define your temperature sequence.

    Process: Use Matrix_plotting.py to inspect the data.

    Fit: Edit the paths in Temperature_profile_fitting.py to point to your data folder and run the global optimization. Results are exported as high-resolution plots and an Excel summary table.

