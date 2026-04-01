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

# --- 2. Comprehensive Reference Library ---
POLYMER_DB = {
    # --- BIOPOLYMERS & POLYESTERS ---
    "PLA": {1750: "C=O (Ester)", 1180: "C-O-C", 1080: "C-O"},
    "PBAT": {1715: "C=O (Arom.)", 1270: "C-O", 720: "CH2-bend"},
    "PBS": {1710: "C=O", 1150: "C-O-C"},
    "PHA / PHB": {1720: "C=O", 1280: "C-O-C", 1055: "C-O"},
    "PET": {1715: "C=O (Ester)", 1240: "C-O (Ester)", 1100: "Arom. C-H", 725: "Ring-def."},
    "PC (Polycarbonate)": {1775: "C=O (Carbonate)", 1500: "Arom. C=C", 1220: "C-O-C", 1190: "O-C-O"},

    # --- COMMODITY PLASTICS (POLYOLEFINS & STYRENICS) ---
    "PE (HDPE/LDPE)": {2920: "CH2 asym-str", 2850: "CH2 sym-str", 1470: "CH2-bend", 720: "CH2-rock (Cryst.)"},
    "PP": {2950: "CH3-str", 1455: "CH2-bend", 1376: "CH3-bend (Sym)", 973: "Isotactic band"},
    "PS (Polystyrene)": {3026: "Arom. C-H str", 2924: "CH2 str", 1601: "Arom. C=C", 1492: "Arom. ring", 755: "Out-of-plane C-H", 698: "Ring bend"},
    "PVC": {2970: "CH2-str", 1425: "CH2-bend", 1250: "CH-bend", 690: "C-Cl str", 615: "C-Cl str"},
    "PMMA (Acrylic)": {1725: "C=O", 1435: "CH3-bend", 1145: "C-O-C"},

    # --- ENGINEERING PLASTICS ---
    "PA6 / PA66 (Nylon)": {3300: "N-H str (H-bonded)", 1640: "Amide I (C=O)", 1540: "Amide II (N-H bend / C-N str)"},
    "POM (Acetal/Delrin)": {2920: "CH2-str", 1090: "C-O-C asym-str", 900: "C-O-C sym-str"},
    "PTFE (Teflon)": {1200: "CF2 asym-str", 1150: "CF2 sym-str", 640: "CF2 wag"},
    "ABS": {2237: "C≡N (Nitrile)", 1601: "Arom. C=C (Styrene)", 1492: "Arom. ring", 966: "C=C trans (Butadiene)"},

    # --- RUBBERS & ELASTOMERS ---
    "NR (Natural Rubber)": {2960: "CH3 str", 2850: "CH2 str", 1660: "C=C str", 1450: "CH2 bend", 835: "=C-H out-of-plane"},
    "SBR (Styrene-Butadiene)": {2920: "CH2 str", 1600: "Arom. C=C", 1492: "Arom. ring", 965: "C=C trans", 760: "Arom. C-H", 699: "Ring bend"},
    "NBR (Nitrile Rubber)": {2920: "CH2 str", 2237: "C≡N str (Strong)", 1440: "CH2 bend", 966: "C=C trans"},
    "EPDM": {2920: "CH2 str", 2850: "CH2 str", 1460: "CH2 bend", 1375: "CH3 bend", 720: "CH2 rock"},
    "Silicone (PDMS)": {2960: "CH3 str", 1260: "Si-CH3 sym-bend (Sharp)", 1090: "Si-O-Si asym-str (Broad)", 1020: "Si-O-Si", 800: "Si-C str"},
    "CR (Neoprene)": {1660: "C=C str", 1440: "CH2 bend", 1120: "C-C str", 825: "C-Cl str"},
    "FKM (Viton/Fluoroelastomer)": {1200: "C-F str", 1120: "CF2 str", 880: "C-F bend"},
    "PU (Polyurethane)": {3330: "N-H str", 1730: "C=O (Non-H-bonded)", 1700: "C=O (H-bonded)", 1530: "Amide II", 1220: "C-O-C"},

    # --- GENERAL FUNCTIONAL GROUPS ---
    "General / Unknown": {3300: "O-H / N-H (Broad)", 2920: "C-H (Aliphatic)", 2250: "C≡N / C≡C", 1720: "C=O (Carbonyl)", 1640: "C=C / H2O", 1050: "C-O / C-N"}
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
    stack_offset = st.slider("Vertical Offset Cushion", 0.0, 1.0, 0.2, step=0.05)
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
                        # Load Data safely
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
                            actual_window = smooth_val if smooth_val < data_len else (data_len - 1 if (data_len - 1) % 2 != 0 else data_len - 2)
                            
                            if actual_window > 3:
                                df['Intensity'] = savgol_filter(df['Intensity'], actual_window, 3)
                            elif actual_window == 3:
                                df['Intensity'] = savgol_filter(df['Intensity'], actual_window, 2)

                        # 4. Baseline Correction & Normalization
                        df['Intensity'] = df['Intensity'] - df['Intensity'].min()
                        max_val = df['Intensity'].max()
                        if max_val > 0:
                            df['Intensity'] = df['Intensity'] / max_val

                        # 5. 2nd Derivative
                        d_window = max(3, actual_window - 2)
                        if d_window % 2 == 0:  
                            d_window += 1 
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
        current_baseline = 0.0 
        
        for i, name in enumerate(master['File'].unique()):
            df = spectra[name]
            
            fig.add_trace(go.Scatter(
                x=df['Wavenumber'], y=df['Intensity'] + current_baseline,
                mode='lines', line=dict(width=line_w), 
                name=name, hoverinfo="name+x+y", showlegend=False 
            ))

            # Primary Tab Label Anchor (3900 cm⁻¹)
            anchor_x = 3900 
            if anchor_x > df['Wavenumber'].max():
                anchor_x = df['Wavenumber'].max()
                
            idx_anchor = (df['Wavenumber'] - anchor_x).abs().idxmin()
            anchor_y = df.loc[idx_anchor, 'Intensity'] + current_baseline

            fig.add_annotation(
                x=anchor_x, y=anchor_y + 0.03, 
                text=f"<b>{name}</b>", showarrow=False,
                xanchor='left', yanchor='bottom',
                font=dict(family="Arial", size=14, color="black")
            )

            for poly in selected_ref:
                for wn, label in POLYMER_DB[poly].items():
                    if df['Wavenumber'].min() <= wn <= df['Wavenumber'].max():
                        idx = (df['Wavenumber'] - wn).abs().idxmin()
                        py = df.loc[idx, 'Intensity'] + current_baseline
                        fig.add_annotation(
                            x=wn, y=py, text=label, showarrow=True, 
                            arrowhead=1, ay=-30, font=dict(size=10)
                        )
            
            spectrum_max_height = df['Intensity'].max()
            current_baseline += spectrum_max_height + stack_offset 

        dynamic_plot_height = 600 + (len(master['File'].unique()) * 80)

        fig.update_layout(
            template="simple_white", height=dynamic_plot_height,
            xaxis=dict(title="<b>Wavenumber (cm⁻¹)</b>", range=[4000, 400], **FTIR_STYLE),
            yaxis=dict(title="<b>Absorbance (Stacked)</b>", showticklabels=False, **FTIR_STYLE),
            showlegend=False 
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if calc_deriv:
            st.markdown("**2nd Derivative Spectra:** Minima (valleys) correspond to peak maxima in the original spectrum.")
            fig_deriv = go.Figure()
            
            # --- SMART STACKING FOR DERIVATIVES ---
            current_deriv_baseline = 0.0
            
            for i, name in enumerate(master['File'].unique()):
                df = spectra[name]
                if '2nd_Deriv' in df.columns:
                    
                    fig_deriv.add_trace(go.Scatter(
                        x=df['Wavenumber'], y=df['2nd_Deriv'] + current_deriv_baseline,
                        mode='lines', line=dict(width=1.5), 
                        name=name, hoverinfo="name+x+y", showlegend=False
                    ))
                    
                    # Derivative Tab Label Anchor (1950 cm⁻¹ because we zoom in)
                    anchor_x_deriv = 1950 
                    if anchor_x_deriv > df['Wavenumber'].max():
                        anchor_x_deriv = df['Wavenumber'].max()
                        
                    idx_anchor_d = (df['Wavenumber'] - anchor_x_deriv).abs().idxmin()
                    anchor_y_d = df.loc[idx_anchor_d, '2nd_Deriv'] + current_deriv_baseline
                    
                    amp_max = df['2nd_Deriv'].max()
                    amp_min = df['2nd_Deriv'].min()

                    fig_deriv.add_annotation(
                        x=anchor_x_deriv, 
                        y=anchor_y_d + (amp_max * 0.2), # Nudge text just above the line
                        text=f"<b>{name}</b>",
                        showarrow=False,
                        xanchor='left',
                        yanchor='bottom',
                        font=dict(family="Arial", size=14, color="black")
                    )
                    
                    # Advance the baseline using the full peak-to-valley height plus a cushion
                    current_deriv_baseline += (amp_max - amp_min) + (stack_offset * 0.05)
            
            dynamic_plot_height_deriv = 600 + (len(master['File'].unique()) * 80)

            fig_deriv.update_layout(
                template="simple_white", height=dynamic_plot_height_deriv,
                xaxis=dict(title="<b>Wavenumber (cm⁻¹)</b>", range=[2000, 600], **FTIR_STYLE), 
                yaxis=dict(title="<b>d²A/dν² (Stacked)</b>", showticklabels=False, **FTIR_STYLE),
                showlegend=False
            )
            st.plotly_chart(fig_deriv, use_container_width=True)
        else:
            st.info("Check 'Show 2nd Derivative Tab' in the sidebar to view peak deconvolution.")

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
    st.info("Upload your raw data files in the sidebar to begin.")
