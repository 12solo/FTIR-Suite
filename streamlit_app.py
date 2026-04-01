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

# Journal Style Config: Mirror Box
FTIR_STYLE = dict(
    showline=True, mirror=True, ticks='outside', 
    linecolor='black', linewidth=2.5,
    title_font=dict(family="Times New Roman", size=22, color="black"),
    tickfont=dict(family="Times New Roman", size=18, color="black"),
    showgrid=False,
)

# --- 2. Polymer & Rubber Reference Library ---
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
    "Polyethylene (PE)": {2915: "CH2 Sym Stretch", 2848: "CH2 Asym Stretch", 1470: "CH2 Bend", 718: "Rocking"},
    "Starch/Cellulose": {3300: "O-H (Broad)", 1015: "C-O Stretch", 1640: "Bound Water"},
    "General": {3300: "O-H (Broad)", 1650: "C=C / Amide", 2920: "C-H Alkane"}
}

def clean_name(filename):
    return os.path.splitext(filename)[0]

# --- 3. Header ---
st.title("🧪 Solomon FTIR Suite 2.0")
st.markdown("**Polymer Identification & Spectral Analysis (Robust Loader)**")

# --- 4. Sidebar ---
with st.sidebar:
    st.header("🔬 Analysis Mode")
    y_mode = st.radio("Y-Axis", ["Transmittance (%)", "Absorbance"])
    
    st.header("🧬 Polymer Reference")
    selected_ref = st.multiselect("Assign Peaks From Library", list(POLYMER_DB.keys()), default=["General"])
    
    st.header("🎨 Plot Styling")
    line_w = st.slider("Line Thickness", 1.0, 5.0, 2.0)
    leg_x = st.slider("Legend X", 0.0, 1.0, 0.75)
    leg_y = st.slider("Legend Y", 0.0, 1.0, 0.95)

    st.header("📂 Data Input")
    with st.form("ftir_upload", clear_on_submit=True):
        group_id = st.text_input("Sample Group", "Bioplastic Blend")
        files = st.file_uploader("Upload Spectra", accept_multiple_files=True)
        submit = st.form_submit_button("Process Batch")

    if st.button("Reset App Data", type="primary"):
        st.session_state['ftir_master_df'] = pd.DataFrame()
        st.session_state['spectra_storage'] = {}
        st.rerun()

# --- 5. Robust Processing Engine ---
def robust_ftir_load(file):
    """Skips headers and handles inconsistent column counts."""
    try:
        # Step 1: Try reading as a raw string to find the data start
        raw_bytes = file.read()
        content = raw_bytes.decode('utf-8', errors='ignore')
        file.seek(0) # Reset pointer
        
        # Determine separator
        sep = '\t' if '\t' in content else ','
        
        # Step 2: Use on_bad_lines='skip' to avoid the "Expected X fields" crash
        # Also use comment='#' to ignore common header lines
        df = pd.read_csv(
            io.BytesIO(raw_bytes), 
            sep=sep, 
            on_bad_lines='skip', 
            engine='python',
            comment='#'
        )
        
        # Step 3: Clean up - Keep only numeric rows
        df = df.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
        
        # Step 4: Map columns (Wavenumber vs Intensity)
        if len(df.columns) >= 2:
            df = df.iloc[:, :2]
            df.columns = ['Wavenumber', 'Intensity']
            
            # Standardization
            df = df.sort_values('Wavenumber', ascending=False).reset_index(drop=True)
            
            # Baseline Correction
            y = df['Intensity'].values
            slope = (y[-1] - y[0]) / (len(y) - 1)
            df['Intensity'] = y - (slope * np.arange(len(y)) + y[0])
            
            # Normalization
            imin, imax = df['Intensity'].min(), df['Intensity'].max()
            df['Intensity'] = (df['Intensity'] - imin) / (imax - imin)
            
            return df
        return None
    except Exception as e:
        st.error(f"Error loading {file.name}: {e}")
        return None

if submit and files:
    for f in files:
        df_proc = robust_ftir_load(f)
        if df_proc is not None:
            name = clean_name(f.name)
            st.session_state['spectra_storage'][name] = df_proc
            new_entry = pd.DataFrame([{"Group": group_id, "File": name}])
            st.session_state['ftir_master_df'] = pd.concat([st.session_state['ftir_master_df'], new_entry], ignore_index=True)

# --- 6. Dashboard ---
master = st.session_state['ftir_master_df']
spectra = st.session_state['spectra_storage']

if not master.empty:
    tabs = st.tabs(["📊 Dataset", "📈 Overlay & Assignments", "💾 Export"])

    with tabs[1]:
        st.subheader("Scientific Spectral Analysis")
        fig = go.Figure()
        
        for name in master['File'].unique():
            df = spectra[name]
            fig.add_trace(go.Scatter(x=df['Wavenumber'], y=df['Intensity'], mode='lines', 
                                     line=dict(width=line_w), name=f"<b>{name}</b>"))

        # Reference Assignments
        for polymer in selected_ref:
            if polymer in POLYMER_DB:
                for wn, label in POLYMER_DB[polymer].items():
                    fig.add_annotation(
                        x=wn, y=1.02, text=f"<b>{label}</b>",
                        showarrow=True, arrowhead=2, arrowcolor="gray", ay=-40,
                        font=dict(family="Times New Roman", size=12, color="blue")
                    )
                    fig.add_shape(type="line", x0=wn, y0=0, x1=wn, y1=1, line=dict(color="rgba(0,0,255,0.1)", dash="dot"))

        fig.update_layout(
            template="simple_white", height=800,
            xaxis=dict(title="<b>Wavenumber (cm⁻¹)</b>", range=[4000, 400], **FTIR_STYLE),
            yaxis=dict(title=f"<b>Normalized Intensity</b>", range=[0, 1.15], **FTIR_STYLE),
            showlegend=True,
            legend=dict(x=leg_x, y=leg_y, xanchor='left', yanchor='top', borderwidth=0, bgcolor="rgba(255,255,255,0.5)"),
            margin=dict(l=80, r=40, t=100, b=80)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[0]:
        st.dataframe(master, use_container_width=True)

    with tabs[2]:
        export_dfs = [df.set_index('Wavenumber').rename(columns={'Intensity': name}) for name, df in spectra.items()]
        combined = pd.concat(export_dfs, axis=1).sort_index(ascending=False)
        st.download_button("📥 Download Summary Matrix", combined.to_csv().encode('utf-8'), "FTIR_Analysis.csv")
else:
    st.info("👋 Upload FTIR data files to begin. The loader will automatically handle file headers.")
