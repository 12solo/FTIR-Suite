import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter
import os

# --- 1. Page Configuration ---
st.set_page_config(page_title="Solomon FTIR Suite 5.0", layout="wide", page_icon="🧪")

st.title("🧪 Solomon FTIR Suite 5.0: Advanced Spectral Analysis")
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

# Comprehensive Reference Library
POLYMER_DB = {
    "PLA": {1750: "C=O (Ester)", 1180: "C-O-C", 1080: "C-O"},
    "PBAT": {1715: "C=O (Arom.)", 1270: "C-O", 720: "CH2-bend"},
    "PBS": {1710: "C=O", 1150: "C-O-C"},
    "PHA / PHB": {1720: "C=O", 1280: "C-O-C", 1055: "C-O"},
    "PET": {1715: "C=O (Ester)", 1240: "C-O", 1100: "Arom. C-H", 725: "Ring-def."},
    "PC (Polycarbonate)": {1775: "C=O", 1500: "Arom. C=C", 1220: "C-O-C"},
    "PE (HDPE/LDPE)": {2920: "CH2 asym-str", 2850: "CH2 sym-str", 1470: "CH2-bend", 720: "CH2-rock"},
    "PP": {2950: "CH3-str", 1455: "CH2-bend", 1376: "CH3-bend", 973: "Isotactic"},
    "PS (Polystyrene)": {3026: "Arom. C-H", 2924: "CH2", 1601: "Arom. C=C", 698: "Ring bend"},
    "PVC": {2970: "CH2-str", 1425: "CH2-bend", 1250: "CH-bend", 690: "C-Cl str"},
    "PMMA (Acrylic)": {1725: "C=O", 1435: "CH3-bend", 1145: "C-O-C"},
    "PA6 / PA66 (Nylon)": {3300: "N-H str", 1640: "Amide I (C=O)", 1540: "Amide II"},
    "POM (Acetal)": {2920: "CH2-str", 1090: "C-O-C asym", 900: "C-O-C sym"},
    "PTFE (Teflon)": {1200: "CF2 asym", 1150: "CF2 sym", 640: "CF2 wag"},
    "ABS": {2237: "C≡N (Nitrile)", 1601: "Arom. C=C", 966: "C=C trans"},
    "NR (Natural Rubber)": {2960: "CH3 str", 1660: "C=C str", 1450: "CH2 bend", 835: "=C-H out"},
    "SBR (Styrene-Butadiene)": {2920: "CH2 str", 1600: "Arom. C=C", 965: "C=C trans", 699: "Ring bend"},
    "NBR (Nitrile Rubber)": {2920: "CH2 str", 2237: "C≡N str", 1440: "CH2 bend", 966: "C=C trans"},
    "EPDM": {2920: "CH2 str", 2850: "CH2 str", 1460: "CH2 bend", 1375: "CH3 bend", 720: "CH2 rock"},
    "Silicone (PDMS)": {2960: "CH3 str", 1260: "Si-CH3", 1090: "Si-O-Si", 800: "Si-C"},
    "CR (Neoprene)": {1660: "C=C str", 1440: "CH2 bend", 1120: "C-C str", 825: "C-Cl str"},
    "FKM (Viton)": {1200: "C-F str", 1120: "CF2 str", 880: "C-F bend"},
    "PU (Polyurethane)": {3330: "N-H str", 1730: "C=O (Free)", 1700: "C=O (H-bond)", 1530: "Amide II"},
    "General / Unknown": {3300: "O-H / N-H", 2920: "C-H", 2250: "C≡N", 1720: "C=O", 1640: "C=C / H2O", 1050: "C-O"}
}

def clean_name(filename):
    return os.path.splitext(filename)[0]

# --- 2. Sidebar Logic ---
with st.sidebar:
    st.header("⚙️ Data Processing & View")
    raw_data_format = st.radio("1. Uploaded Data Format", ["Absorbance", "Transmittance (%)"])
    display_mode = st.radio("2. Output Display Mode", ["Absorbance", "Transmittance (%)"], help="Change this anytime to swap the plot style instantly.")
    
    with st.expander("🔬 Advanced Scientific Corrections", expanded=False):
        apply_atr = st.checkbox("Apply ATR Correction")
        calc_deriv = st.checkbox("Show 2nd Derivative Tab")
        smooth_val = st.slider("Savitzky-Golay Window", 5, 101, 15, step=2)

    st.header("🎨 Plot Formatting")
    stack_offset = st.slider("Vertical Offset Cushion", 0.0, 1.0, 0.2, step=0.05)
    line_w = st.slider("Line Weight", 1.0, 4.0, 2.0)
    selected_ref = st.multiselect("Label Peaks", list(POLYMER_DB.keys()), default=["General / Unknown"])
    
    st.header("📂 Data Input")
    group_id = st.text_input("Sample Group ID", "Experimental_Batch")
    files = st.file_uploader("Upload FTIR (.csv, .txt, .xls, .xlsx)", accept_multiple_files=True)

    if files:
        for f in files:
            name = clean_name(f.name)
            if name not in st.session_state['spectra_storage']:
                with st.spinner(f"Processing {name}..."):
                    try:
                        if f.name.lower().endswith(('.xls', '.xlsx')):
                            df = pd.read_excel(f, header=None)
                        else:
                            df = pd.read_csv(f, header=None, sep=None, engine='python', on_bad_lines='skip')

                        df = df.apply(pd.to_numeric, errors='coerce').dropna().iloc[:, :2]
                        df.columns = ['Wavenumber', 'Raw_Intensity']
                        df = df.sort_values('Wavenumber', ascending=True)

                        # CONVERT EVERYTHING TO ABSORBANCE FOR MATH
                        if raw_data_format == "Transmittance (%)":
                            df['Raw_Intensity'] = df['Raw_Intensity'].clip(lower=0.001)
                            abs_y = 2 - np.log10(df['Raw_Intensity'])
                        else:
                            abs_y = df['Raw_Intensity']

                        if apply_atr:
                            abs_y = abs_y * (df['Wavenumber'] / 1000)

                        data_len = len(df)
                        actual_window = 3 
                        if data_len > 3:
                            actual_window = smooth_val if smooth_val < data_len else (data_len - 1 if (data_len - 1) % 2 != 0 else data_len - 2)
                            if actual_window > 3:
                                abs_y = savgol_filter(abs_y, actual_window, 3)
                            elif actual_window == 3:
                                abs_y = savgol_filter(abs_y, actual_window, 2)

                        # Baseline Correction & Normalization
                        abs_y = abs_y - abs_y.min()
                        max_val = abs_y.max()
                        if max_val > 0:
                            abs_y = abs_y / max_val

                        df['Absorbance_Norm'] = abs_y

                        # 2nd Derivative (Always done on Absorbance)
                        d_window = max(3, actual_window - 2)
                        if d_window % 2 == 0:  
                            d_window += 1 
                        d_poly = min(3, d_window - 1)
                        df['2nd_Deriv'] = savgol_filter(df['Absorbance_Norm'], d_window, d_poly, deriv=2)

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
    tab1, tab2, tab3, tab4 = st.tabs(["📉 Primary Spectra", "🔬 2nd Derivative Deconvolution", "📋 Peak Summary Table", "📊 Data Matrix"])

    with tab1:
        fig = go.Figure()
        current_baseline = 0.0 
        
        # Adjust display logic based on Absorbance vs Transmittance
        is_transmittance = display_mode == "Transmittance (%)"
        scaled_offset = stack_offset * 100 if is_transmittance else stack_offset
        arrow_direction = 30 if is_transmittance else -30
        
        for i, name in enumerate(master['File'].unique()):
            df = spectra[name]
            
            # Dynamic Plot Y Calculation
            if is_transmittance:
                plot_y = 100 * (10 ** -df['Absorbance_Norm'])
            else:
                plot_y = df['Absorbance_Norm']

            fig.add_trace(go.Scatter(
                x=df['Wavenumber'], y=plot_y + current_baseline,
                mode='lines', line=dict(width=line_w), 
                name=name, hoverinfo="name+x+y", showlegend=False 
            ))

            # Primary Tab Label Anchor
            anchor_x = 3900 
            if anchor_x > df['Wavenumber'].max():
                anchor_x = df['Wavenumber'].max()
                
            idx_anchor = (df['Wavenumber'] - anchor_x).abs().idxmin()
            anchor_y = plot_y.iloc[idx_anchor] + current_baseline

            # Nudge label up for Absorbance, down for Transmittance
            label_nudge = -3 if is_transmittance else 0.03

            fig.add_annotation(
                x=anchor_x, y=anchor_y + label_nudge, 
                text=f"<b>{name}</b>", showarrow=False,
                xanchor='left', yanchor='bottom' if not is_transmittance else 'top',
                font=dict(family="Arial", size=14, color="black")
            )

            # Assign Peaks
            for poly in selected_ref:
                for wn, label in POLYMER_DB[poly].items():
                    if df['Wavenumber'].min() <= wn <= df['Wavenumber'].max():
                        idx = (df['Wavenumber'] - wn).abs().idxmin()
                        py = plot_y.iloc[idx] + current_baseline
                        fig.add_annotation(
                            x=wn, y=py, text=label, showarrow=True, 
                            arrowhead=1, ay=arrow_direction, font=dict(size=10)
                        )
            
            # Smart Stacking Math
            spectrum_height = plot_y.max() if not is_transmittance else 100
            current_baseline += spectrum_height + scaled_offset 

        dynamic_plot_height = 600 + (len(master['File'].unique()) * 80)

        fig.update_layout(
            template="simple_white", height=dynamic_plot_height,
            xaxis=dict(title="<b>Wavenumber (cm⁻¹)</b>", range=[4000, 400], **FTIR_STYLE),
            yaxis=dict(title=f"<b>{display_mode} (Stacked)</b>", showticklabels=False, **FTIR_STYLE),
            showlegend=False 
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if calc_deriv:
            st.markdown("**2nd Derivative Spectra:** Used to resolve overlapping peaks. *(Note: Derivatives are always calculated and displayed in Absorbance format for mathematical accuracy).*")
            fig_deriv = go.Figure()
            
            current_deriv_baseline = 0.0
            
            for i, name in enumerate(master['File'].unique()):
                df = spectra[name]
                if '2nd_Deriv' in df.columns:
                    
                    fig_deriv.add_trace(go.Scatter(
                        x=df['Wavenumber'], y=df['2nd_Deriv'] + current_deriv_baseline,
                        mode='lines', line=dict(width=1.5), 
                        name=name, hoverinfo="name+x+y", showlegend=False
                    ))
                    
                    # FIXED DECONVOLUTION LABEL OVERLAP
                    amp_max = df['2nd_Deriv'].max()
                    amp_min = df['2nd_Deriv'].min()
                    
                    anchor_x_deriv = 1950 
                    if anchor_x_deriv > df['Wavenumber'].max():
                        anchor_x_deriv = df['Wavenumber'].max()
                        
                    # Strict mathematical placement: Always forced exactly above the highest peak of this specific curve
                    safe_label_y = current_deriv_baseline + amp_max + (stack_offset * 0.05)

                    fig_deriv.add_annotation(
                        x=anchor_x_deriv, 
                        y=safe_label_y,
                        text=f"<b>{name}</b>",
                        showarrow=False,
                        xanchor='left',
                        yanchor='bottom',
                        font=dict(family="Arial", size=14, color="black")
                    )
                    
                    # Push baseline up by the total peak-to-valley height
                    current_deriv_baseline += (amp_max - amp_min) + (stack_offset * 0.1)
            
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
        st.subheader("📋 Comprehensive Peak Assignment Summary")
        st.markdown(f"*Values currently reported in **{display_mode}***")
        
        summary_data = []
        is_transmittance = display_mode == "Transmittance (%)"

        for name in master['File'].unique():
            df = spectra[name]
            
            # Determine output values
            output_y = 100 * (10 ** -df['Absorbance_Norm']) if is_transmittance else df['Absorbance_Norm']

            for poly in selected_ref:
                for wn, label in POLYMER_DB[poly].items():
                    if df['Wavenumber'].min() <= wn <= df['Wavenumber'].max():
                        idx = (df['Wavenumber'] - wn).abs().idxmin()
                        intensity_val = output_y.iloc[idx]
                        
                        summary_data.append({
                            "Sample Name": name,
                            "Material Lib.": poly,
                            "Target Wavenumber (cm⁻¹)": wn,
                            "Actual Peak Wavenumber (cm⁻¹)": round(df['Wavenumber'].iloc[idx], 1),
                            "Assignment": label,
                            f"Intensity ({display_mode})": round(intensity_val, 4)
                        })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            st.download_button(
                "📥 Download Peak Summary (CSV)", 
                summary_df.to_csv(index=False).encode('utf-8'), 
                f"FTIR_Peak_Summary_{display_mode[:3]}.csv"
            )
        else:
            st.warning("No peaks found for the selected libraries.")

    with tab4:
        common_wn = np.linspace(4000, 400, 1500)
        export_data = {"Wavenumber": common_wn}
        is_transmittance = display_mode == "Transmittance (%)"
        
        for name, df in spectra.items():
            output_y = 100 * (10 ** -df['Absorbance_Norm']) if is_transmittance else df['Absorbance_Norm']
            export_data[name] = np.interp(common_wn, df['Wavenumber'], output_y)
        
        matrix_df = pd.DataFrame(export_data)
        st.download_button(
            f"📥 Download Standardized Data Matrix ({display_mode})", 
            matrix_df.to_csv(index=False).encode('utf-8'), 
            f"FTIR_Matrix_{display_mode[:3]}.csv"
        )
        st.dataframe(matrix_df.head(15))

else:
    st.info("Upload your raw data files in the sidebar to begin.")
