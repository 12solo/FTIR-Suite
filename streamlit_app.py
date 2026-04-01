import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks, savgol_filter
import io
import os

# --- 1. Page Configuration ---
st.set_page_config(page_title="Solomon FTIR Suite 3.0", layout="wide", page_icon="🧪")

# Initialize Session States for data persistence
if 'ftir_master_df' not in st.session_state:
    st.session_state['ftir_master_df'] = pd.DataFrame()
if 'spectra_storage' not in st.session_state:
    st.session_state['spectra_storage'] = {}

# Journal Style Config: Mirror Box / Times New Roman Style
FTIR_STYLE = dict(
    showline=True, mirror=True, ticks='outside', 
    linecolor='black', linewidth=2.5,
    title_font=dict(family="Arial", size=18, color="black"),
    tickfont=dict(family="Arial", size=14, color="black"),
    showgrid=False,
)

# --- 2. Reference Library ---
POLYMER_DB = {
    "PLA": {1750: "C=O (Ester)", 1180: "C-O-C", 1080: "C-O"},
    "PBAT": {1715: "C=O (Arom.)", 1270: "C-O", 720: "CH2-bend"},
    "PBS": {1710: "C=O", 1150: "C-O-C"},
    "PP": {2950: "CH3-str", 1455: "CH2-bend", 1376: "CH3-bend"},
    "General": {3300: "O-H (Broad)", 1640: "H2O / C=C", 2920: "C-H (Aliphatic)"}
}

def clean_name(filename):
    return os.path.splitext(filename)[0]

# --- 3. Sidebar Logic ---
with st.sidebar:
    st.header("⚙️ Spectral Logic")
    y_mode = st.radio("Y-Axis Mode", ["Absorbance", "Transmittance (%)"])
    smooth_val = st.slider("Smoothing Window", 5, 101, 15, step=2, help="Higher = Smoother curve")
    
    st.header("🎨 Visual Stacking")
    stack_offset = st.slider("Vertical Offset", 0.0, 3.0, 0.5, step=0.1)
    line_w = st.slider("Line Weight", 0.5, 5.0, 2.0)
    
    st.header("🧬 Assignment")
    selected_ref = st.multiselect("Label Peaks", list(POLYMER_DB.keys()), default=["General"])

    st.header("📂 Data Input")
    with st.form("ftir_upload", clear_on_submit=True):
        group_id = st.text_input("Series ID", "Study_01")
        files = st.file_uploader("Upload FTIR (.csv, .xlsx, .txt)", accept_multiple_files=True)
        submit = st.form_submit_button("Process & Generate Report")

    if st.button("🗑️ Reset All Data", type="primary"):
        st.session_state['ftir_master_df'] = pd.DataFrame()
        st.session_state['spectra_storage'] = {}
        st.rerun()

# --- 4. Scientific Processing Engine ---
def process_scientific_ftir(file, s_val):
    try:
        # Load Data
        if file.name.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file, header=None)
        else:
            df = pd.read_csv(file, header=None, sep=None, engine='python', on_bad_lines='skip')
        
        # Clean non-numeric rows (headers/metadata)
        df = df.apply(pd.to_numeric, errors='coerce').dropna().iloc[:, :2]
        df.columns = ['Wavenumber', 'Intensity']
        df = df.sort_values('Wavenumber', ascending=True)

        # Dynamic Smoothing Safety Check (Fixes your window_length error)
        data_len = len(df)
        if data_len > 5:
            actual_window = s_val if s_val < data_len else (data_len - 1 if (data_len - 1) % 2 != 0 else data_len - 2)
            if actual_window >= 3:
                df['Intensity'] = savgol_filter(df['Intensity'], actual_window, 3)

        # Baseline Correction & Normalization
        df['Intensity'] = df['Intensity'] - df['Intensity'].min()
        max_val = df['Intensity'].max()
        if max_val > 0:
            df['Intensity'] = df['Intensity'] / max_val
        
        return df
    except Exception as e:
        st.error(f"Error processing {file.name}: {e}")
        return None

# Handle Form Submission
if submit and files:
    for f in files:
        df_proc = process_scientific_ftir(f, smooth_val)
        if df_proc is not None:
            name = clean_name(f.name)
            st.session_state['spectra_storage'][name] = df_proc
            new_entry = pd.DataFrame([{"Group": group_id, "File": name}])
            st.session_state['ftir_master_df'] = pd.concat([st.session_state['ftir_master_df'], new_entry], ignore_index=True)

# --- 5. Output Dashboard ---
master = st.session_state['ftir_master_df']
spectra = st.session_state['spectra_storage']

if not master.empty:
    tab_plot, tab_peaks, tab_data = st.tabs(["📉 Waterfall Plot", "📍 Peak Analysis", "💾 Export Data"])

    with tab_plot:
        fig = go.Figure()
        unique_files = master['File'].unique()
        
        for i, name in enumerate(unique_files):
            df = spectra[name]
            offset = i * stack_offset
            
            # Plot the spectrum
            fig.add_trace(go.Scatter(
                x=df['Wavenumber'], y=df['Intensity'] + offset,
                mode='lines', 
                line=dict(width=line_w), 
                name=f"<b>{name}</b>"
            ))

            # Automated Labeling from DB
            for poly in selected_ref:
                for wn, label in POLYMER_DB[poly].items():
                    # Check if wavenumber exists in range
                    if df['Wavenumber'].min() <= wn <= df['Wavenumber'].max():
                        idx = (df['Wavenumber'] - wn).abs().idxmin()
                        py = df.loc[idx, 'Intensity'] + offset
                        fig.add_annotation(
                            x=wn, y=py, text=label, showarrow=True, 
                            arrowhead=1, ay=-30, font=dict(size=10)
                        )

        fig.update_layout(
            template="simple_white", height=800,
            xaxis=dict(title="<b>Wavenumber (cm⁻¹)</b>", range=[4000, 400], **FTIR_STYLE),
            yaxis=dict(title="<b>Normalized Intensity + Offset</b>", showticklabels=False, **FTIR_STYLE),
            legend=dict(x=1.01, y=1, bordercolor="Black", borderwidth=1),
            margin=dict(l=50, r=150, t=50, b=50)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_peaks:
        st.subheader("Automated Peak Detection")
        peak_results = []
        for name, df in spectra.items():
            # Finds peaks with at least 5% height
            p_idx, props = find_peaks(df['Intensity'], height=0.05, distance=50)
            for p in p_idx:
                peak_results.append({
                    "Sample": name,
                    "Peak Wavenumber (cm⁻¹)": round(df.iloc[p]['Wavenumber'], 1),
                    "Relative Intensity": round(df.iloc[p]['Intensity'], 3)
                })
        
        if peak_results:
            st.dataframe(pd.DataFrame(peak_results), use_container_width=True)
        else:
            st.warning("No significant peaks detected. Adjust smoothing or check data.")

    with tab_data:
        # Create a single Matrix for Excel/CSV export
        if spectra:
            # Re-index to a common wavenumber scale for clean comparison
            common_wn = np.linspace(4000, 400, 1000)
            export_data = {"Wavenumber": common_wn}
            for name, df in spectra.items():
                f_interp = np.interp(common_wn, df['Wavenumber'], df['Intensity'])
                export_data[name] = f_interp
            
            matrix_df = pd.DataFrame(export_data)
            st.download_button(
                "📥 Download Combined Matrix (CSV)", 
                matrix_df.to_csv(index=False).encode('utf-8'), 
                "FTIR_Stacked_Matrix.csv"
            )
            st.dataframe(matrix_df)

else:
    st.info("👋 Welcome! Please upload your FTIR data files in the sidebar to generate a scientific waterfall plot.")
