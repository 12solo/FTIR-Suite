import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import os

# --- 1. Page Configuration ---
st.set_page_config(page_title="Solomon FTIR Suite 2.2", layout="wide")

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

# --- 2. Enhanced Polymer Reference Library ---
POLYMER_DB = {
    "PLA": {1750: "C=O", 1180: "C-O-C", 1080: "C-O", 1450: "δas CH3"},
    "PBAT": {1715: "C=O (arom.)", 1270: "C-O", 720: "CH2-bend"},
    "PBS": {1710: "C=O", 1150: "C-O-C", 1330: "CH2-wag"},
    "PET": {1715: "C=O", 1240: "C-O", 1100: "arom. C-H", 725: "ring-def."},
    "PP": {2950: "CH3-str", 2917: "CH2-str", 1455: "CH2-bend", 1376: "CH3-bend"},
    "PMMA": {1725: "C=O", 1240: "C-O", 1145: "C-O-C"},
    "PCL": {1720: "C=O", 1293: "C-O", 1165: "C-O-C", 730: "CH2-rock"},
    "EPDM": {2920: "CH2-str", 2850: "CH2-str", 1460: "CH2-bend", 1375: "CH3", 720: "-(CH2)n-"},
    "TPV": {2920: "PP/EPDM", 1460: "bend", 1376: "CH3", 973: "cryst."},
    "General": {3300: "O-H", 1640: "H2O", 2920: "C-H"}
}

def clean_name(filename):
    return os.path.splitext(filename)[0]

# --- 3. Header ---
st.title("🧪 Solomon FTIR Suite 2.2")
st.markdown("**Stacked Spectral Analysis (Waterfall Plot) for Scientific Publication**")

# --- 4. Sidebar ---
with st.sidebar:
    st.header("🔬 Spectral Logic")
    y_mode = st.radio("Y-Axis Mode", ["Transmittance (%)", "Absorbance"])
    
    st.header("🧬 Assignment Library")
    selected_ref = st.multiselect("Select Polymers to Label", list(POLYMER_DB.keys()), default=["General"])
    
    st.header("🎨 Waterfall Stacking")
    # THE KEY LOGIC: Y-Offset for stacking
    stack_offset = st.slider("Vertical Stack Offset", 0.0, 2.0, 0.5, help="Increase to separate spectra vertically")
    line_w = st.slider("Curve Thickness", 1.0, 5.0, 2.0)
    
    st.subheader("📍 Legend Position")
    leg_x = st.slider("Horizontal", 0.0, 1.0, 0.85)
    leg_y = st.slider("Vertical", 0.0, 1.0, 0.95)

    st.header("📂 Data Input")
    with st.form("ftir_upload", clear_on_submit=True):
        group_id = st.text_input("Sample Group", "Waterfall Series 1")
        files = st.file_uploader("Upload (.xls, .xlsx, .csv, .txt)", accept_multiple_files=True)
        submit = st.form_submit_button("Process & Stack")

    if st.button("Reset Study", type="primary"):
        st.session_state['ftir_master_df'] = pd.DataFrame()
        st.session_state['spectra_storage'] = {}
        st.rerun()

# --- 5. Robust Processing Engine ---
def robust_ftir_load(file):
    try:
        if file.name.lower().endswith(('.xls', '.xlsx')):
            engine = 'xlrd' if file.name.lower().endswith('.xls') else 'openpyxl'
            df = pd.read_excel(file, header=None, engine=engine)
        else:
            raw_bytes = file.read()
            content = raw_bytes.decode('utf-8', errors='ignore')
            sep = '\t' if '\t' in content else ','
            df = pd.read_csv(io.StringIO(content), sep=sep, header=None, on_bad_lines='skip')
        
        df = df.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
        
        if len(df.columns) >= 2:
            df = df.iloc[:, :2]
            df.columns = ['Wavenumber', 'Intensity']
            df = df.sort_values('Wavenumber', ascending=False).reset_index(drop=True)
            
            # Baseline Correction
            y = df['Intensity'].values
            slope = (y[-1] - y[0]) / (len(y) - 1)
            df['Intensity'] = y - (slope * np.arange(len(y)) + y[0])
            
            # Normalization (0-1)
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
    tabs = st.tabs(["📊 Dataset", "📉 Stacked Scientific Figure", "💾 Export"])

    with tabs[1]:
        fig = go.Figure()
        unique_files = master['File'].unique()
        
        # Plot Stacked Spectra
        for i, name in enumerate(unique_files):
            df = spectra[name]
            # APPLY OFFSET BASED ON INDEX
            current_offset = i * stack_offset
            
            fig.add_trace(go.Scatter(
                x=df['Wavenumber'], 
                y=df['Intensity'] + current_offset, 
                mode='lines', 
                line=dict(width=line_w), 
                name=f"<b>{name}</b>"
            ))

            # Apply Assignments LOCALLY to each stacked curve
            for polymer in selected_ref:
                if polymer in POLYMER_DB:
                    for wn, label in POLYMER_DB[polymer].items():
                        # Find value on this specific offset curve
                        closest_idx = (df['Wavenumber'] - wn).abs().idxmin()
                        point_y = df.loc[closest_idx, 'Intensity'] + current_offset
                        
                        fig.add_annotation(
                            x=wn, y=point_y,
                            text=f"<b>{label}</b>",
                            showarrow=True, arrowhead=1, arrowsize=1, arrowcolor="black",
                            ax=0, ay=-30,
                            font=dict(family="Times New Roman", size=12, color="black"),
                            bgcolor="rgba(255,255,255,0.6)"
                        )

        fig.update_layout(
            template="simple_white", height=900,
            xaxis=dict(title="<b>Wavenumber (cm⁻¹)</b>", range=[4000, 400], **FTIR_STYLE),
            yaxis=dict(
                title=f"<b>Normalized Intensity ({y_mode}) + Offset</b>", 
                showticklabels=False, # Standard for waterfall plots to hide absolute Y values
                **FTIR_STYLE
            ),
            showlegend=True,
            legend=dict(x=leg_x, y=leg_y, xanchor='left', yanchor='top', borderwidth=0, bgcolor="rgba(255,255,255,0.4)"),
            margin=dict(l=80, r=40, t=50, b=80)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[0]:
        st.dataframe(master, use_container_width=True)

    with tabs[2]:
        export_dfs = [df.set_index('Wavenumber').rename(columns={'Intensity': name}) for name, df in spectra.items()]
        combined = pd.concat(export_dfs, axis=1).sort_index(ascending=False)
        st.download_button("📥 Download Summary Matrix", combined.to_csv().encode('utf-8'), "FTIR_Stacked_Analysis.csv")
else:
    st.info("👋 Upload FTIR data files to generate a stacked waterfall plot.")
