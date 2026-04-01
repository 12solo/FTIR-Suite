# 📖 Documentation: Solomon FTIR spectra analyser

## 1. Overview
The **Solomon FTIR Suite 5.0** is an advanced, interactive scientific application built with Python and Streamlit. It is designed for researchers, manufacturing engineers, and material scientists to rapidly process, visualize, and analyze Fourier Transform Infrared (FTIR) spectroscopy data. 

It transitions raw machine exports into publication-ready figures by automating complex pre-processing steps, including baseline correction, Savitzky-Golay smoothing, 2nd derivative deconvolution, and dynamic peak assignment against a massive built-in library of polymers and elastomers.

---

## 2. Installation & Setup
To run this application locally, you must have Python installed. 

**Step 1: Install Required Libraries**
Open your terminal or command prompt and install the necessary dependencies by running:
```bash
pip install streamlit pandas numpy plotly scipy xlrd openpyxl
```
*(Note: `xlrd` and `openpyxl` are specifically required to allow the app to read `.xls` and `.xlsx` files).*

**Step 2: Launch the App**
Save the provided Python code as `app.py`. In your terminal, navigate to the folder containing the file and run:
```bash
streamlit run app.py
```
This will automatically open the application in your default web browser.

---

## 3. Step-by-Step User Guide

### Step 1: Data Configuration
Before uploading data, look at the **⚙️ Data Processing & View** section in the left sidebar:
* **Uploaded Data Format:** Select how your raw machine data was exported (Absorbance or Transmittance). The app will perform the mathematical conversion ($A = 2 - \log_{10}T\%$) automatically in the background.
* **Output Display Mode:** Choose how you want to *view* the data. You can toggle this setting at any time to instantly swap your plots and tables between Absorbance and Transmittance.

### Step 2: Uploading Data
* Go to the **📂 Data Input** section.
* Provide a "Sample Group ID" (optional, useful for batch tracking).
* Drag and drop your files (`.csv`, `.txt`, `.xls`, `.xlsx`).
* *Note: The app will instantly process the files as soon as they are dropped. There is no submit button.*

### Step 3: Spectral Adjustments
Use the **🎨 Plot Formatting** section to perfect your visualization:
* **Label Peaks:** Select the materials you expect in your samples from the comprehensive dropdown library. The app will automatically search your uploaded spectra for these specific functional groups.
* **Vertical Offset Cushion:** Adjust this slider to push the stacked spectra further apart or closer together. 
* **Clear Memory:** If you want to start a new analysis, click the red **🗑️ Clear Memory** button. Do not just upload new files over the old ones, as this can cause overlapping data.

---

## 4. Understanding the Analysis Tabs

The main dashboard features four distinct views for your data:

### 📉 Tab 1: Primary Spectra
Generates a publication-ready "Waterfall Plot."
* **Smart Stacking:** The app measures the absolute maximum height of every individual spectrum and guarantees the next spectrum floats exactly above it. They will never overlap, regardless of peak intensity.
* **Inline Labels:** Spectrum names are anchored neatly to the left side (3900 $cm^{-1}$) to avoid bulky legend boxes.

### 🔬 Tab 2: 2nd Derivative Deconvolution
This tab zooms into the "Fingerprint Region" (2000–600 $cm^{-1}$).
* 2nd derivatives are strictly calculated in *Absorbance* (required for linear mathematical accuracy). 
* Minima (valleys) on these curves correspond precisely to the hidden peak maxima in your original spectrum, helping resolve overlapping bands (e.g., separating $C=O$ esters from $C=O$ acids).

### 📋 Tab 3: Peak Summary Table
An automated report that scans your actual data against the theoretical values of the polymer library you selected.
* It reports the *actual* wavenumber where the peak was found in your sample and its exact intensity.
* Click **Download Peak Summary** to export this table for your thesis or research paper.

### 📊 Tab 4: Data Matrix
Different FTIR machines often export data with different step sizes (e.g., one file might have a data point at 1000.1, another at 1000.4). 
* This tab uses mathematical interpolation to perfectly align all your uploaded spectra onto a single, unified wavenumber scale (4000 to 400 $cm^{-1}$ at exact intervals).
* This matrix is essential if you plan to export the data for Principal Component Analysis (PCA) or external machine learning.

---

## 5. Scientific Methodology (The Engine)
If you need to write a methodology section for a journal article regarding how this software treats data, you can cite the following processes:

1. **ATR Correction (Optional):** Applies a wavelength-dependent correction factor ($A_{corr} = A \times (\nu / 1000)$) to account for varying penetration depths when using an Attenuated Total Reflectance crystal.
2. **Savitzky-Golay Smoothing:** Applies a moving polynomial fit (Degree=3) to remove high-frequency detector noise without altering the area or fundamental shape of the chemical peaks. The window size dynamically shrinks to protect small datasets from crashing.
3. **Baseline Subtraction:** Shifts the global minimum of the spectrum to zero, correcting for broadband scattering often seen in manufacturing samples.
4. **Min-Max Normalization:** Scales the entire spectrum from 0 to 1, allowing samples of different thicknesses to be visually compared on the same axes.

---

## 6. Troubleshooting

* **Error: `KeyError: 'Absorbance_Norm'` or similar:** This happens when Streamlit remembers data from a previous session where settings were different. **Fix:** Click the red **🗑️ Clear Memory** button in the sidebar and re-upload your files.
* **Error: `utf-8 codec can't decode...` on XLS files:** You are missing the Excel reader engine. **Fix:** Run `pip install xlrd openpyxl` in your terminal.
* **Spectra look completely flat:** You likely selected "Transmittance" as your input format, but your raw data is actually in Absorbance. Switch the "Uploaded Data Format" toggle in the sidebar.
