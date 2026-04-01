import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks, savgol_filter
import os

# --- 1. Page Configuration ---
st.set_page_config(page_title="Solomon FTIR Suite 4.0", layout="wide", page_icon="🧪")

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
        calc_deriv = st.checkbox("Show 2nd Derivative Tab", help="Resolves overlapping peaks.")
        smooth_val = st.slider("Savitzky-Golay Window", 5, 101, 15, step=2)

    st.header("🎨 Plot Formatting")
    stack_offset = st.slider("Vertical Offset", 0.0, 2.0, 0.5, step=0.1)
    line_w = st.slider("Line Weight", 1.0, 4.0, 2.0)
    selected_ref = st.multiselect("Label Peaks", list(POLYMER_DB.keys()), default=["General"])
    
    st.header("📂 Data Input")
    group_id = st.text_input("Sample Group ID", "Experimental_Batch")
    files = st.file_uploader("Upload FTIR (.csv, .txt, .xls, .xlsx)", accept_multiple_files=True)

    # INSTANT PROCESSING
    if files:
        for f in files:
            name = clean_name(f.name)
            if name not in st.session_state['spectra_storage']:
                with st.spinner(f"Processing {name}..."):
                    try:
                        # --- PROPERLY INDENTED EXCEL/CSV LOADER ---
                        if f.name.lower().endswith(('.xls', '.xlsx')):
                            df = pd.read_excel(f, header=None)
                        else:
                            df = pd.read_csv(f, header=None, sep=None, engine='python', on_bad_lines='skip')

                        # Clean data
                        df = df.apply(pd.to_numeric, errors='coerce').dropna().iloc[:, :2]
                        df.columns = ['Wavenumber', 'Raw_Intensity']
                        df = df.sort_values('Wavenumber', ascending=True)

                        # 1. Convert Transmittance to Absorbance
                        if raw_data_format == "Transmittance (%)":
                            df['Raw_Intensity'] = df['Raw_Intensity'].clip(lower=0.001)
                            df['Intensity'] = 2 - np.log10(df['Raw_Intensity'])
                        else:
                            df['Intensity'] = df['Raw_Intensity']

                        # 2. ATR Correction
                        if apply_atr:
                            df['Intensity'] = df['Intensity'] * (df['Wavenumber'] / 1000)

                        # 3. Smoothing (Mathematical Edge-Case Fix)
                        data_len = len(df)
                        actual_window = 3 # Safe minimum fallback
                        if data_len > 3:
                            # Ensure window is odd and fits data length
                            actual_window = smooth_val if smooth_val < data_len else (data_len - 1 if (data_len - 1) % 2 != 0 else data_len - 2)
                            
                            if actual_window > 3:
                                df['Intensity'] = savgol_filter(df['Intensity'], actual_window, 3)
                            elif actual_window == 3:
                                # If window shrinks to 3, polyorder MUST be 2 or lower
                                df['Intensity'] = savgol_filter(df['Intensity'], actual_window, 2)

                        # 4. Baseline Correction & Normalization
                        df['Intensity'] = df['Intensity'] - df['Intensity'].min()
                        max_val = df['Intensity'].max()
                        if max_val > 0:
                            df['Intensity'] = df['Intensity'] / max_val

                        # 5. 2nd Derivative (Mathematical Edge-Case Fix)
                        d_window = max(3, actual_window - 2)
                        if d_window % 2 == 0:  
                            d_window += 1 
                        
                        # Polyorder must strictly be less than window length
                        d_poly = min(3, d_window - 1)
                        df['2nd_Deriv'] = savgol_filter(df['Intensity'], d_window, d_poly, deriv=2)

                        # Save to memory
                        st.session_state['spectra_storage'][name] = df
                        new_entry = pd.DataFrame({"Group": [group_id], "File": [name]})
                        st.session_state['ftir_master_df'] = pd.concat([st.session_state['ftir_master_df'], new_entry], ignore_index=True)
                        st.success(f"Successfully loaded: {name}")

                    except Exception as e:
                        st.error(f"Error processing {f.name}: {e}")

    if st.button("🗑️ Clear Memory", type="primary"):
        st.session_state['ftir_master_df'] = pd.DataFrame()
        st.session_state['spectra_storage'] = {}
        st.rerun()

# --- 3. Dashboard View ---
master = st.session_state['ftir_master_df']
spectra = st.session_state['spectra_storage']

if not master.empty:
    tab1, tab2, tab3 = st.tabs(["📉 Primary Spectra", "🔬 2nd Derivative Deconvolution", "📊 Data Matrix"])

    with tab1:
        fig = go.Figure()
        
        # Smart Stacking: Track the dynamic baseline
        current_baseline = 0.0 
        
        for i, name in enumerate(master['File'].unique()):
            df = spectra[name]
            
            # 1. Plot the current spectrum at the current baseline
            fig.add_trace(go.Scatter(
                x=df['Wavenumber'], y=df['Intensity'] + current_baseline,
                mode='lines', line=dict(width=line_w), name=name
            ))

            # 2. Auto-label peaks for this spectrum
            for poly in selected_ref:
                for wn, label in POLYMER_DB[poly].items():
                    if df['Wavenumber'].min() <= wn <= df['Wavenumber'].max():
                        idx = (df['Wavenumber'] - wn).abs().idxmin()
                        py = df.loc[idx, 'Intensity'] + current_baseline
                        fig.add_annotation(
                            x=wn, y=py, text=label, showarrow=True, 
                            arrowhead=1, ay=-30, font=dict(size=10)
                        )
            
            # 3. Calculate the baseline for the NEXT spectrum
            # It takes the absolute highest peak of the current curve and adds your slider offset
            spectrum_max_height = df['Intensity'].max()
            current_baseline += spectrum_max_height + (stack_offset * 0.2) # 0.2 multiplier keeps the slider sensitivity manageable

        fig.update_layout(
            template="simple_white", height=700 + (len(master['File'].unique()) * 50), # Auto-expands plot height based on file count
            xaxis=dict(title="<b>Wavenumber (cm⁻¹)</b>", range=[4000, 400], **FTIR_STYLE),
            yaxis=dict(title="<b>Absorbance (Normalized) + Offset</b>", showticklabels=False, **FTIR_STYLE),
            legend=dict(x=1.01, y=1, bordercolor="Black", borderwidth=1)
        )
        st.plotly_chart(fig, use_container_width=True)
            # Auto-label peaks
            for poly in selected_ref:
                for wn, label in POLYMER_DB[poly].items():
                    if df['Wavenumber'].min() <= wn <= df['Wavenumber'].max():
                        idx = (df['Wavenumber'] - wn).abs().idxmin()
                        py = df.loc[idx, 'Intensity'] + offset
                        fig.add_annotation(
                            x=wn, y=py, text=label, showarrow=True, 
                            arrowhead=1, ay=-30, font=dict(size=10)
                        )

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
                if '2nd_Deriv' in df.columns:
                    fig_deriv.add_trace(go.Scatter(
                        x=df['Wavenumber'], y=df['2nd_Deriv'],
                        mode='lines', line=dict(width=1.5), name=name
                    ))
            
            fig_deriv.update_layout(
                template="simple_white", height=600,
                xaxis=dict(title="<b>Wavenumber (cm⁻¹)</b>", range=[2000, 600], **FTIR_STYLE), 
                yaxis=dict(title="<b>d²A/dν²</b>", **FTIR_STYLE)
            )
            fig_deriv.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_deriv, use_container_width=True)
        else:
            st.info("Check 'Show 2nd Derivative Tab' in the sidebar to view peak deconvolution.")

    with tab3:
        # Standardize matrix for Excel export
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
    st.info("Upload your raw data files in the sidebar to begin.")
