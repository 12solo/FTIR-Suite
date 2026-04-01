import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks, savgol_filter
import io
import os

# --- 1. Page Configuration ---
st.set_page_config(page_title="Solomon FTIR Suite 3.0", layout="wide")

# Initialize Session States
if 'ftir_master_df' not in st.session_state:
    st.session_state['ftir_master_df'] = pd.DataFrame()
if 'spectra_storage' not in st.session_state:
    st.session_state['spectra_storage'] = {}

# Journal Style Config: Mirror Box
FTIR_STYLE = dict(
    showline=True, mirror=True, ticks='outside', 
    linecolor='black', linewidth=2.5,
    title_font=dict(family="Arial", size=18, color="black"),
    tickfont=dict(family="Arial", size=14, color="black"),
    showgrid=False,
)

# --- 2. Enhanced Reference Library ---
POLYMER_DB = {
    "PLA": {1750: "C=O", 1180: "C-O-C", 1080: "C-O"},
    "PBAT": {1715: "C=O (arom.)", 1270: "C-O", 720: "CH2-bend"},
    "PP": {2950: "CH3", 1455: "CH2", 1376: "CH3-bend"},
    "General": {3300: "O-H", 1640: "H2O", 2920: "C-H"}
}

def clean_name(filename):
    return os.path.splitext(filename)[0]

# --- 3. Sidebar Logic ---
with st.sidebar:
    st.header("🔬 Processing & View")
    y_mode = st.radio("Y-Axis Mode", ["Absorbance", "Transmittance (%)"])
    smooth_val = st.slider("Smoothing (Savitzky-Golay)", 5, 51, 11, step=2)
    
    st.header("🎨 Waterfall Stacking")
    stack_offset = st.slider("Vertical Offset", 0.0, 2.0, 0.4)
    line_w = st.slider("Line Weight", 1.0, 4.0, 2.0)
    
    st.header("🧬 Assignment")
    selected_ref = st.multiselect("Label Peaks", list(POLYMER_DB.keys()), default=["General"])

    st.header("📂 Data Input")
    with st.form("ftir_upload", clear_on_submit=True):
        group_id = st.text_input("Series Name", "Batch 1")
        files = st.file_uploader("Upload Files", accept_multiple_files=True)
        submit = st.form_submit_button("Analyze & Plot")

    if st.button("Clear All Data"):
        st.session_state['ftir_master_df'] = pd.DataFrame()
        st.session_state['spectra_storage'] = {}
        st.rerun()

# --- 4. Scientific Processing Engine ---
def process_scientific_ftir(file):
    try:
        # Robust loading
        if file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file, header=None)
        else:
            df = pd.read_csv(file, header=None, sep=None, engine='python')
        
        df = df.apply(pd.to_numeric, errors='coerce').dropna().iloc[:, :2]
        df.columns = ['Wavenumber', 'Intensity']
        df = df.sort_values('Wavenumber', ascending=True)

        # A. Smoothing
        df['Intensity'] = savgol_filter(df['Intensity'], smooth_val, 3)

        # B. Baseline Correction (Min-subtraction)
        df['Intensity'] = df['Intensity'] - df['Intensity'].min()

        # C. Normalization
        df['Intensity'] = (df['Intensity'] - df['Intensity'].min()) / (df['Intensity'].max() - df['Intensity'].min())
        
        return df
    except Exception as e:
        st.error(f"Failed to process {file.name}: {e}")
        return None

if submit and files:
    for f in files:
        df_proc = process_scientific_ftir(f)
        if df_proc is not None:
            name = clean_name(f.name)
            st.session_state['spectra_storage'][name] = df_proc
            new_entry = pd.DataFrame([{"Group": group_id, "File": name}])
            st.session_state['ftir_master_df'] = pd.concat([st.session_state['ftir_master_df'], new_entry], ignore_index=True)

# --- 5. Multi-Tab Dashboard ---
master = st.session_state['ftir_master_df']
spectra = st.session_state['spectra_storage']

if not master.empty:
    tab_plot, tab_peaks, tab_data = st.tabs(["📉 Waterfall Plot", "📍 Peak Report", "💾 Export"])

    with tab_plot:
        fig = go.Figure()
        unique_files = master['File'].unique()
        
        for i, name in enumerate(unique_files):
            df = spectra[name]
            offset = i * stack_offset
            
            # Plot Curve
            fig.add_trace(go.Scatter(
                x=df['Wavenumber'], y=df['Intensity'] + offset,
                mode='lines', line=dict(width=line_w), name=name
            ))

            # Auto-label from library
            for poly in selected_ref:
                for wn, label in POLYMER_DB[poly].items():
                    # Find closest point in current spectrum
                    mask = (df['Wavenumber'] - wn).abs() < 10
                    if any(mask):
                        idx = (df['Wavenumber'] - wn).abs().idxmin()
                        py = df.loc[idx, 'Intensity'] + offset
                        fig.add_annotation(x=wn, y=py, text=label, showarrow=True, ay=-40)

        fig.update_layout(
            template="simple_white", height=700,
            xaxis=dict(title="Wavenumber (cm⁻¹)", range=[4000, 400], **FTIR_STYLE),
            yaxis=dict(title=f"Intensity + Offset ({y_mode})", showticklabels=False, **FTIR_STYLE),
            legend=dict(x=0.02, y=0.98, bordercolor="Black", borderwidth=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_peaks:
        st.subheader("Significant Peaks Identified (>5% Intensity)")
        all_peaks = []
        for name, df in spectra.items():
            p_idx, _ = find_peaks(df['Intensity'], height=0.05, distance=30)
            for p in p_idx:
                all_peaks.append({
                    "Sample": name,
                    "Wavenumber": round(df.iloc[p]['Wavenumber'], 1),
                    "Rel. Intensity": round(df.iloc[p]['Intensity'], 3)
                })
        st.dataframe(pd.DataFrame(all_peaks), use_container_width=True)

    with tab_data:
        # Merge all into one CSV
        export_list = [df.set_index('Wavenumber')['Intensity'].rename(n) for n, df in spectra.items()]
        combined = pd.concat(export_list, axis=1)
        st.download_button("Download Matrix CSV", combined.to_csv().encode('utf-8'), "FTIR_Matrix.csv")

else:
    st.info("Upload your spectra files (CSV/Excel) to begin the analysis.")
