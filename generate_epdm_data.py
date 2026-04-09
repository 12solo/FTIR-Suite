import streamlit as st
import numpy as np
import pandas as pd
import os

# ==========================================
# 1. SETUP & CONSTANTS
# ==========================================
OUTPUT_DIR = "Synthetic_EPDM_Data"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Standard mid-IR wavenumber range (4000 to 400 cm⁻¹, resolution ~2 cm⁻¹)
wavenumbers = np.linspace(4000, 400, 1800)

def gaussian(x, amplitude, center, width):
    """Generates a Gaussian peak"""
    return amplitude * np.exp(-((x - center) ** 2) / (2 * width ** 2))

# ==========================================
# 2. DATA GENERATION ENGINE
# ==========================================
def generate_spectrum(days, temp_c, koh_m):
    # Initialize zero array
    absorbance = np.zeros_like(wavenumbers)
    
    # --- STATIC EPDM BACKBONE PEAKS ---
    absorbance += gaussian(wavenumbers, 0.85, 2920, 15) # CH2 asym stretch
    absorbance += gaussian(wavenumbers, 0.65, 2850, 12) # CH2 sym stretch
    absorbance += gaussian(wavenumbers, 0.45, 1460, 10) # CH2 bending (REFERENCE)
    absorbance += gaussian(wavenumbers, 0.35, 1375, 8)  # CH3 bending
    absorbance += gaussian(wavenumbers, 0.25, 720, 12)  # CH2 rocking
    
    # --- KINETIC DEGRADATION LOGIC ---
    # Convert Temp to Kelvin for basic Arrhenius acceleration
    T_K = temp_c + 273.15
    T_ref = 65 + 273.15 
    
    # Simulated activation energy acceleration factor
    # Higher temp and higher KOH drive the reaction much faster
    temp_factor = np.exp(4500 * ((1/T_ref) - (1/T_K)))
    conc_factor = (koh_m / 0.5) ** 0.8 
    
    # Total degradation severity
    severity = days * temp_factor * conc_factor * 0.005
    
    # --- DYNAMIC AGING PEAKS ---
    # Carbonyl formation (Oxidation)
    absorbance += gaussian(wavenumbers, 0.02 + (severity * 0.8), 1715, 20) 
    
    # Hydroxyl / Water uptake (Hydrolysis/Oxidation)
    # Broader peak, grows slightly faster
    absorbance += gaussian(wavenumbers, 0.05 + (severity * 1.2), 3400, 70) 
    
    # --- REALISM (Noise & Baseline Drift) ---
    # Random instrumental noise
    noise = np.random.normal(0, 0.002, len(wavenumbers))
    
    # Slight baseline slope (scattering effect)
    baseline = 0.03 * np.exp((wavenumbers - 400) / 3000)
    
    final_absorbance = absorbance + noise + baseline
    
    # Ensure no negative absorbance from noise
    final_absorbance = np.clip(final_absorbance, a_min=0, a_max=None)
    
    return final_absorbance

# ==========================================
# 3. EXPERIMENTAL MATRIX
# ==========================================
# We will simulate two temperatures and two concentrations over 21 days
time_points = [0, 7, 14, 21]  # Days
conditions = [
    {"temp": 65, "koh": 0.5},
    {"temp": 65, "koh": 2.0},
    {"temp": 80, "koh": 0.5},
    {"temp": 80, "koh": 2.0}
]

print("🔬 Generating Synthetic EPDM Spectra...")

for cond in conditions:
    t = cond["temp"]
    k = cond["koh"]
    
    for d in time_points:
        # Day 0 is baseline, so it doesn't need temp/koh variations, but we make copies for the matrix
        if d == 0:
            filename = f"EPDM_Control_Day0_Rep{t}_{k}.csv"
        else:
            filename = f"EPDM_Day{d}_{t}C_{k}M.csv"
            
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Generate the y-data
        abs_y = generate_spectrum(days=d, temp_c=t, koh_m=k)
        
        # Create DataFrame and export
        df = pd.DataFrame({
            "Wavenumber": wavenumbers,
            "Absorbance": abs_y
        })
        
        # Round data to typical FTIR precision
        df = df.round(4)
        df.to_csv(filepath, index=False, header=False)
        print(f"  ✓ Saved: {filename}")

print(f"\n✅ Success! {len(time_points) * len(conditions)} files created in the '{OUTPUT_DIR}' folder.")
print("Upload these into your Streamlit app, make sure 'Data Format' is set to 'Absorbance', and assign the correct metadata in Tab 9.")
