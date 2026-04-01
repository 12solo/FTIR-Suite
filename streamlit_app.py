import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks, savgol_filter
import os

# --- 1. Page Configuration (Browser Tab) ---
st.set_page_config(page_title="Solomon FTIR Suite 4.0", layout="wide", page_icon="🧪")

# --- ACTUAL PAGE HEADING ---
st.title("🧪 Solomon FTIR Suite 4.0: Advanced Spectral Analysis")
st.markdown("---")

# Initialize Session States
if 'ftir_master_df' not in st.session_state:
    st.session_state['ftir_master_df'] = pd.DataFrame()
if 'spectra_storage' not in st.session_state:
    st.session_state['spectra_storage'] = {}

# Journal Style Config
FTIR_STYLE = dict(
    showline=True, mirror=True, ticks='outside', 
    linecolor='black', linewidth=2.5,
    title_font=dict(family="Arial", size=18, color="black"),
    tickfont=dict(family="Arial", size=14, color="black"),
    showgrid=False,
)

# Reference Library
POLYMER_DB = {
    "PLA": {1750: "C=O", 1180: "C-O-C", 1080: "C-O"},
    "PBAT": {1715: "C=O (Arom.)", 1270: "C-O", 720: "CH2-bend"},
    "PBS": {1710: "C=O", 1150: "C-O-C"},
    "PP": {2950: "CH3-str", 1455: "CH2-bend", 1376: "CH3-bend"},
    "General": {3300: "O-H", 1640: "H2O", 2920: "C-H"}
}

def clean_name(filename):
    return os.path.splitext(filename)[0]

# --- 2. Sidebar Logic ---
with st.sidebar:
    st.header("⚙️ Data Pre-Processing")
    raw_data_format = st.radio("Raw Data Format", ["Absorbance", "Transmittance (%)"])
    
    with st.expander("🔬 Advanced Scientific Corrections", expanded=True):
        apply_atr = st.checkbox("Apply ATR Correction", help="Corrects for wavelength-dependent penetration depth in ATR-FTIR.")
        calc_deriv = st.checkbox("Calculate 2nd Derivative", help="Resolves overlapping peaks.")
        smooth_val = st.slider("Savitzky-Golay Window", 5, 101, 15, step=2)

    st.header("🎨 Plot Formatting")
    stack_offset = st.slider("Vertical Offset", 0.0, 2.0, 0.5, step=0.1)
    
    st.header("📂 Data Input")
    with st.form("ftir_upload", clear_on_submit=True):
        group_id = st.text_input("Sample Group ID", "Experimental_Batch")
        files = st.file_uploader("Upload FTIR (.csv, .txt)", accept_multiple_files=True)
        submit = st.form_submit_button("Process Spectra")

    if st.button("🗑️ Clear Memory", type="primary"):
        st.session_state['ftir_master_df'] = pd.DataFrame()
        st.session_state['spectra_storage'] = {}
        st.rerun()

# --- 3. Scientific Processing Engine ---
def process_scientific_ftir(file, s_val):
    try:
        # Load Data
        df = pd.read_csv(file, header=None, sep=None, engine='python', on_bad_lines='skip')
        df = df.apply(pd.to_numeric, errors='coerce').dropna().iloc[:, :2]
        df.columns = ['Wavenumber', 'Raw_Intensity']
        df = df.sort_values('Wavenumber', ascending=True)

        # 1. Convert Transmittance to Absorbance if needed
        if raw_data_format == "Transmittance (%)":
            # Avoid log(0) errors
            df['Raw_Intensity'] = df['Raw_Intensity'].clip(lower=0.001)
            df['Intensity'] = 2 - np.log10(df['Raw_Intensity'])
        else:
            df['Intensity'] = df['Raw_Intensity']

        # 2. ATR Correction
        if apply_atr:
            df['Intensity'] = df['Intensity'] * (df['Wavenumber'] / 1000)

        # 3. Smoothing
        data_len = len(df)
        if data_len > 5:
            actual_window = s_val if s_val < data_len else (data_len - 1 if (data_len - 1) % 2 != 0 else data_len - 2)
            if actual_window >= 3:
                df['Intensity'] = savgol_filter(df['Intensity'], actual_window, 3)

        # 4. Baseline Min-Subtraction & Normalization
        df['Intensity'] = df['Intensity'] - df['Intensity'].min()
        max_val = df['Intensity'].max()
        if max_val > 0:
            df['Intensity'] = df['Intensity'] / max_val

        # 5. 2nd Derivative Calculation (Optional)
        if calc_deriv:
            # Deriv window usually needs to be slightly smaller than smoothing window
            d_window = max(3, actual_window - 2)
            df['2nd_Deriv'] = savgol_filter(df['Intensity'], d_window, 3, deriv=2)

        return df
    except Exception as e:
        st.error(f"Error processing {file.name}: {e}")
        return None

if submit and files:
    for f in files:
        df_proc = process_scientific_ftir(f, smooth_val)
        if df_proc is not None:
            name = clean_name(f.name)
            st.session_state['spectra_storage'][name] = df_proc
            new_entry = pd.DataFrame({"Group": [group_id], "File": [name]})
            st.session_state['ftir_master_df'] = pd.concat([st.session_state['ftir_master_df'], new_entry], ignore_index=True)

# --- 4. Dashboard View ---
master = st.session_state['ftir_master_df']
spectra = st.session_state['spectra_storage']

if not master.empty:
    tab1, tab2, tab3 = st.tabs(["📉 Primary Spectra", "🔬 2nd Derivative Deconvolution", "📊 Data Matrix"])

    with tab1:
        fig = go.Figure()
        for i, name in enumerate(master['File'].unique()):
            df = spectra[name]
            offset = i * stack_offset
            fig.add_trace(go.Scatter(
                x=df['Wavenumber'], y=df['Intensity'] + offset,
                mode='lines', line=dict(width=2.0), name=name
            ))

        fig.update_layout(
            template="simple_white", height=700,
            xaxis=dict(title="<b>Wavenumber (cm⁻¹)</b>", range=[4000, 400], **FTIR_STYLE),
            yaxis=dict(title="<b>Absorbance (Normalized) + Offset</b>", showticklabels=False, **FTIR_STYLE),
            legend=dict(x=1.01, y=1, bordercolor="Black", borderwidth=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if calc_deriv:
            st.markdown("**2nd Derivative Spectra:** Used to find hidden peaks. Minima (valleys) correspond to peak maxima in the original spectrum.")
            fig_deriv = go.Figure()
            for name in master['File'].unique():
                df = spectra[name]
                fig_deriv.add_trace(go.Scatter(
                    x=df['Wavenumber'], y=df['2nd_Deriv'],
                    mode='lines', line=dict(width=1.5), name=name
                ))
            
            fig_deriv.update_layout(
                template="simple_white", height=600,
                xaxis=dict(title="<b>Wavenumber (cm⁻¹)</b>", range=[2000, 600], **FTIR_STYLE), # Zoomed in for detail
                yaxis=dict(title="<b>d²A/dν²</b>", **FTIR_STYLE)
            )
            # Add a zero-line for reference
            fig_deriv.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_deriv, use_container_width=True)
        else:
            st.info("Check 'Calculate 2nd Derivative' in the sidebar to view peak deconvolution.")

    with tab3:
        common_wn = np.linspace(4000, 400, 1500)
        export_data = {"Wavenumber": common_wn}
        for name, df in spectra.items():
            export_data[name] = np.interp(common_wn, df['Wavenumber'], df['Intensity'])
        
        matrix_df = pd.DataFrame(export_data)
        st.download_button(
            "📥 Download Standardized Matrix", 
            matrix_df.to_csv(index=False).encode('utf-8'), 
            "FTIR_Scientific_Matrix.csv"
        )
        st.dataframe(matrix_df.head(15))

else:
    st.info("Upload your raw data files to begin.")
