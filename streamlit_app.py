import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import os
from scipy.signal import find_peaks

# --- 1. Page Configuration ---
st.set_page_config(page_title="Solomon FTIR Suite 2.0", layout="wide")

if 'ftir_master_df' not in st.session_state:
    st.session_state['ftir_master_df'] = pd.DataFrame()
if 'spectra_storage' not in st.session_state:
    st.session_state['spectra_storage'] = {}

# Journal Style Config
FTIR_STYLE = dict(
    showline=True, mirror=True, ticks='outside', 
    linecolor='black', linewidth=2.5,
    title_font=dict(family="Times New Roman", size=22, color="black"),
    tickfont=dict(family="Times New Roman", size=18, color="black"),
    showgrid=False,
)

# --- 2. Polymer Reference Library ---
# Dictionary of characteristic peaks (Wavenumber: Label)
POLYMER_DB = {
    "PLA": {1750: "C=O (Ester)", 1180: "C-O-C", 1080: "C-O", 1450: "CH3 Bend"},
    "PBAT": {1715: "C=O (Aromatic Ester)", 1270: "C-O", 720: "CH2 Rock (Aromatic)"},
    "PBS": {1710: "C=O", 1150: "C-O-C", 1330: "CH2 Wag"},
    "PET": {1715: "C=O", 1240: "C-O (Ester)", 1100: "Aromatic C-H", 725: "Aromatic Ring"},
    "PP": {2950: "CH3 Stretch", 2917: "CH2 Stretch", 1455: "CH2 Bend", 1376: "CH3 Bend"},
    "PMMA": {1725: "C=O (Ester)", 1240: "C-O", 1145: "C-O-C", 2950: "a-methyl CH3"},
    "PCL": {1720: "C=O", 1293: "C-O", 1165: "C-O-C", 730: "CH2 Rock"},
    "EPDM": {2920: "CH2 Stretch", 2850: "CH2 Stretch", 1460: "CH2 Bend", 1375: "CH3 (Propylene)", 720: "-(CH2)n-"},
    "TPV": {2920: "PP Phase", 1460: "EPDM/PP Bend", 1376: "CH3", 973: "PP Crystallinity"},
    "General": {3300: "O-H (Broad)", 1650: "C=C / Amide", 2920: "C-H Alkane"}
}

def clean_name(filename):
    return os.path.splitext(filename)[0]

# --- 3. Header ---
st.title("🧪 Solomon FTIR Suite 2.0")
st.markdown("**Advanced Polymer Identification & Spectral Analysis**")

# --- 4. Sidebar ---
with st.sidebar:
    st.header("🔬 Analysis Mode")
    y_mode = st.radio("Y-Axis", ["Transmittance (%)", "Absorbance"])
    
    st.header("🧬 Polymer Reference")
    selected_ref = st.multiselect("Apply Peak Assignment Library", list(POLYMER_DB.keys()), default=["General"])
    
    st.header("🎨 Plot Styling")
    line_w = st.slider("Line Thickness", 1.0, 5.0, 2.0)
    leg_x = st.slider("Legend X", 0.0, 1.0, 0.75)
    leg_y = st.slider("Legend Y", 0.0, 1.0, 0.95)

    st.header("📂 Data Input")
    with st.form("ftir_upload", clear_on_submit=True):
        group_id = st.text_input("Sample Group", "Bioplastic Blend")
        files = st.file_uploader("Upload Spectra", accept_multiple_files=True)
        submit = st.form_submit_button("Process Batch")

    if st.button("Clear All Data", type="primary"):
        st.session_state['ftir_master_df'] = pd.DataFrame()
        st.session_state['spectra_storage'] = {}
        st.rerun()

# --- 5. Processing Engine ---
def process_spectrum(df, mode):
    df.columns = ['Wavenumber', 'Intensity']
    df = df.sort_values('Wavenumber', ascending=False).reset_index(drop=True)
    
    # Simple Baseline Correction
    y = df['Intensity'].values
    slope = (y[-1] - y[0]) / (len(y) - 1)
    df['Intensity'] = y - (slope * np.arange(len(y)) + y[0])
    
    if mode == "Absorbance":
        if df['Intensity'].max() > 2: # Convert %T to A
            df['Intensity'] = 2 - np.log10(np.clip(df['Intensity'], 0.1, 100))
        else:
            df['Intensity'] = -np.log10(np.clip(df['Intensity'], 0.001, 1))

    # Min-Max Normalize to keep 0-1 range for journal style
    imin, imax = df['Intensity'].min(), df['Intensity'].max()
    df['Intensity'] = (df['Intensity'] - imin) / (imax - imin)
    return df

if submit and files:
    for f in files:
        try:
            content = f.getvalue().decode('utf-8', errors='ignore')
            sep = '\t' if '\t' in content else ','
            df_raw = pd.read_csv(io.StringIO(content), sep=sep, header=None).apply(pd.to_numeric, errors='coerce').dropna().iloc[:, :2]
            df_proc = process_spectrum(df_raw, y_mode)
            name = clean_name(f.name)
            st.session_state['spectra_storage'][name] = df_proc
            new_entry = pd.DataFrame([{"Group": group_id, "File": name}])
            st.session_state['ftir_master_df'] = pd.concat([st.session_state['ftir_master_df'], new_entry], ignore_index=True)
        except Exception as e:
            st.error(f"Error: {e}")

# --- 6. Dashboard ---
master = st.session_state['ftir_master_df']
spectra = st.session_state['spectra_storage']

if not master.empty:
    tabs = st.tabs(["📊 Dataset", "📈 Overlay & Assignments", "💾 Export"])

    with tabs[1]:
        st.subheader("Spectral Analysis with Smart Labels")
        fig = go.Figure()
        
        # Plot Spectra
        for name in master['File'].unique():
            df = spectra[name]
            fig.add_trace(go.Scatter(x=df['Wavenumber'], y=df['Intensity'], mode='lines', 
                                     line=dict(width=line_w), name=f"<b>{name}</b>"))

        # Apply Reference Library Assignments
        for polymer in selected_ref:
            peaks_to_tag = POLYMER_DB[polymer]
            for wn, label in peaks_to_tag.items():
                # Find the closest wavenumber in the actual data to place the tag
                fig.add_annotation(
                    x=wn, y=1.05, # Place at the top
                    text=f"<b>{label}</b><br>{wn}",
                    showarrow=True, arrowhead=2, arrowcolor="gray",
                    ax=0, ay=-40,
                    font=dict(family="Times New Roman", size=12, color="blue")
                )
                # Vertical indicator line
                fig.add_shape(type="line", x0=wn, y0=0, x1=wn, y1=1,
                              line=dict(color="rgba(0,0,255,0.1)", width=1, dash="dot"))

        fig.update_layout(
            template="simple_white", height=850,
            xaxis=dict(title="<b>Wavenumber (cm⁻¹)</b>", range=[4000, 400], **FTIR_STYLE),
            yaxis=dict(title=f"<b>Normalized {y_mode}</b>", range=[0, 1.2], **FTIR_STYLE),
            showlegend=True,
            legend=dict(x=leg_x, y=leg_y, xanchor='left', yanchor='top', bgcolor="rgba(255,255,255,0.5)", font=dict(family="Times New Roman", size=16)),
            margin=dict(l=80, r=40, t=100, b=80)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[0]:
        st.dataframe(master, use_container_width=True)

    with tabs[2]:
        export_dfs = [df.set_index('Wavenumber').rename(columns={'Intensity': name}) for name, df in spectra.items()]
        combined = pd.concat(export_dfs, axis=1).sort_index(ascending=False)
        st.download_button("📥 Download Processed Matrix", combined.to_csv().encode('utf-8'), "FTIR_Summary.csv")
else:
    st.info("👋 Upload FTIR data files to begin spectral assignment.")
