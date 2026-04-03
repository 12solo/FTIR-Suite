import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import os
import re

# --- 1. Page Configuration ---
st.set_page_config(page_title="Solomon FTIR Pro Suite 5.0", layout="wide", page_icon="SR.png")

# --- 2. Advanced Mathematical Functions ---
def baseline_als(y, lam=1e5, p=0.01, niter=10):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L-2, L))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.T @ D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def detect_peaks(y, prom=0.02, dist=20):
    peaks, props = find_peaks(y, prominence=prom, distance=dist)
    sorted_indices = np.argsort(props["prominences"])[::-1]
    return peaks[sorted_indices]

def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def match_spectrum(sample, library):
    scores = {}
    for name, ref in library.items():
        score = cosine_similarity([sample], [ref])[0][0]
        scores[name] = score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def clean_name(filename):
    return os.path.splitext(filename)[0]

# --- 3. Database & Styles ---
FTIR_STYLE = dict(
    showline=True, mirror=True, ticks='outside', 
    linecolor='black', linewidth=2.5,
    title_font=dict(family="Arial", size=18, color="black"),
    tickfont=dict(family="Arial", size=14, color="black"),
    showgrid=False,
)

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

# Initialize Session States
if 'ftir_master_df' not in st.session_state:
    st.session_state['ftir_master_df'] = pd.DataFrame()
if 'spectra_storage' not in st.session_state:
    st.session_state['spectra_storage'] = {}

# --- 4. UI Layout ---
title_col1, title_col2 = st.columns([1, 15])
with title_col1:
    if os.path.exists("SR.png"):
        st.image("SR.png", width=60)
with title_col2:
    st.title("Solomon FTIR Pro Suite 5.0")
st.markdown("---")

with st.sidebar:
    # Logo and App Name
    if os.path.exists("SR.png"):
        st.image("SR.png", width=120) # Made slightly larger for the sidebar
    
    # Professional Developer Card
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px; text-align: center; border: 1px solid #e6e6e6;'>
            <p style='color: #333333; font-size: 1em; font-weight: bold; margin-bottom: 0px;'>DEVELOPED BY SOLOMON</p>
            <p style='color: #666666; font-size: 0.85em; margin-top: 2px; margin-bottom: 10px;'>FTIR Pro Suite v5.0</p>
            <hr style='margin: 10px 0; border: 0; border-top: 1px solid #ddd;'>
            <a href='mailto:your.solomon,duf@gmail.com' style='color: #0066cc; text-decoration: none; font-size: 0.85em; font-weight: 500;'>✉️ Contact Developer</a>
            <br>
            <p style='color: #999999; font-size: 0.7em; margin-top: 10px; margin-bottom: 0px;'><i>For Research & Academic Use Only<br>© 2026</i></p>
        </div>
    """, unsafe_allow_html=True)
    
    st.header("⚙️ Data Formatting")
    raw_data_format = st.radio("Uploaded Data Format", ["Absorbance", "Transmittance (%)"])
    display_mode = st.radio("Output Display Mode", ["Absorbance", "Transmittance (%)"])
    
    with st.expander("🔬 Advanced Math Settings", expanded=False):
        apply_atr = st.checkbox("Apply ATR Correction")
        smooth_val = st.slider("Savitzky-Golay Window", 5, 101, 15, step=2)
        apply_baseline = st.checkbox("Apply ALS Baseline Correction", True)
        if apply_baseline:
            als_lam = st.selectbox("Baseline Stiffness (Lambda)", [1e3, 1e4, 1e5, 1e6], index=2)
            als_p = st.selectbox("Baseline Asymmetry (p)", [0.001, 0.01, 0.05, 0.1], index=1)

    st.header("🎨 Plot Formatting")
    stack_offset = st.slider("Vertical Offset Cushion", 0.0, 1.0, 0.2, step=0.05)
    line_w = st.slider("Line Weight", 1.0, 4.0, 2.0)
    selected_ref = st.multiselect("Label Peaks (DB)", list(POLYMER_DB.keys()), default=["General / Unknown"])
    
    st.header("📂 Data Input")
    group_id = st.text_input("Sample Group ID", "Experimental_Batch")
    files = st.file_uploader("Upload FTIR Data", accept_multiple_files=True)

    if files:
        for f in files:
            name = clean_name(f.name)
            if name not in st.session_state['spectra_storage']:
                with st.spinner(f"Processing {name}..."):
                    try:
                        # 1. ULTIMATE FILE PARSING (Handles "Fake" Excel Files)
                        df = None
                        
                        # Attempt 1: Try reading as a true Excel file
                        if f.name.lower().endswith(('.xls', '.xlsx')):
                            try:
                                df = pd.read_excel(f, header=None)
                            except Exception:
                                # If it crashes, it's a text file disguised as an .xls
                                pass 
                        
                        # Attempt 2: Universal Text/CSV/Fake-Excel Parser
                        if df is None:
                            f.seek(0) # Reset file reading pointer
                            content = f.getvalue().decode('utf-8', errors='ignore').split('\n')
                            parsed_data = []
                            for line in content:
                                line = line.strip()
                                if not line: continue
                                # Split by commas, tabs, semicolons, or spaces
                                parts = re.split(r'[,\t;|\s]+', line)
                                if len(parts) >= 2:
                                    try:
                                        # Only grab things that are actually numbers
                                        val_x = float(parts[0])
                                        val_y = float(parts[1])
                                        parsed_data.append([val_x, val_y])
                                    except ValueError:
                                        pass # Silently skip instrument metadata/text headers
                            
                            if not parsed_data:
                                raise ValueError("Could not extract any numerical data from file.")
                                
                            df = pd.DataFrame(parsed_data)

                        # --- ROBUST CLEANUP ---
                        df = df.iloc[:, :2] # Force keep only first two columns
                        df.columns = ['Wavenumber', 'Raw_Intensity']
                        
                        # Force everything to numeric (Turns any remaining text into blank 'NaN's)
                        df['Wavenumber'] = pd.to_numeric(df['Wavenumber'], errors='coerce')
                        df['Raw_Intensity'] = pd.to_numeric(df['Raw_Intensity'], errors='coerce')
                        
                        # Delete the blank rows safely, then sort ascending
                        df = df.dropna().sort_values('Wavenumber', ascending=True).reset_index(drop=True)
                        # ----------------------

                        # 2. CONVERT TO ABSORBANCE FOR MATH
                        raw_y = df['Raw_Intensity'].values
                        if raw_data_format == "Transmittance (%)":
                            raw_y = np.clip(raw_y, a_min=0.001, a_max=None)
                            abs_y = 2 - np.log10(raw_y)
                        else:
                            abs_y = raw_y.copy()

                        if apply_atr:
                            abs_y = abs_y * (df['Wavenumber'].values / 1000)

                        data_len = len(df)
                        actual_window = smooth_val if smooth_val < data_len else (data_len - 1 if (data_len - 1) % 2 != 0 else data_len - 2)
                        if actual_window >= 3:
                            abs_y = savgol_filter(abs_y, actual_window, 3)

                        if apply_baseline:
                            base = baseline_als(abs_y, lam=als_lam, p=als_p)
                            abs_y = abs_y - base

                        abs_y = abs_y - abs_y.min()
                        max_val = abs_y.max()
                        if max_val > 0:
                            abs_y = abs_y / max_val

                        df['Absorbance_Norm'] = abs_y

                        d_window = max(3, actual_window - 2)
                        if d_window % 2 == 0: d_window += 1 
                        d_poly = min(3, d_window - 1)
                        df['2nd_Deriv'] = savgol_filter(df['Absorbance_Norm'].values, d_window, d_poly, deriv=2)

                        st.session_state['spectra_storage'][name] = df
                        new_entry = pd.DataFrame({"Group": [group_id], "File": [name]})
                        st.session_state['ftir_master_df'] = pd.concat([st.session_state['ftir_master_df'], new_entry], ignore_index=True)
                        st.success(f"Successfully loaded: {name}")

                    except Exception as e:
                        st.error(f"Error processing {f.name}: {e}")
# --- 5. Main Dashboard View ---
master = st.session_state['ftir_master_df']
spectra = st.session_state['spectra_storage']

if not master.empty:
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📉 Primary Spectra", 
        "🔬 2nd Derivative", 
        "📋 Peak Assignments", 
        "📈 Gaussian Deconvolution", 
        "🧠 PCA Clustering", 
        "🧪 Spectral Match", 
        "📊 Data Matrix Export"
    ])

    # ---------------------------
    # TAB 1: PRIMARY SPECTRA (Stacked + DB Labels)
    # ---------------------------
    with tab1:
        fig = go.Figure()
        current_baseline = 0.0 
        is_transmittance = display_mode == "Transmittance (%)"
        scaled_offset = stack_offset * 100 if is_transmittance else stack_offset
        
        # Base arrow length
        arrow_direction = 60 if is_transmittance else -60
        
        for i, name in enumerate(master['File'].unique()):
            df = spectra[name]
            wavenumbers = df['Wavenumber'].values
            plot_y = 100 * (10 ** -df['Absorbance_Norm'].values) if is_transmittance else df['Absorbance_Norm'].values

            fig.add_trace(go.Scatter(
                x=wavenumbers, y=plot_y + current_baseline,
                mode='lines', line=dict(width=line_w), 
                name=name, hoverinfo="name+x+y", showlegend=False
            ))

            # --- 1. Inline Legend ---
            anchor_x = 3900 if 3900 <= wavenumbers.max() else wavenumbers.max()
            idx_anchor = np.argmin(np.abs(wavenumbers - anchor_x))
            local_y = plot_y[idx_anchor] 

            nudge = (plot_y.max() - plot_y.min()) * 0.08
            label_y_pos = current_baseline + local_y + (nudge if not is_transmittance else -nudge)

            fig.add_annotation(
                x=anchor_x, y=label_y_pos, 
                text=f"<b>{name}</b>", showarrow=False,
                xanchor='left', yanchor='bottom' if not is_transmittance else 'top',
                font=dict(family="Arial", size=13, color="black")
            )

            # --- 2. Smart Peak Staggering ---
            valid_peaks = []
            for poly in selected_ref:
                for wn, label in POLYMER_DB[poly].items():
                    if wavenumbers.min() <= wn <= wavenumbers.max():
                        idx = np.argmin(np.abs(wavenumbers - wn))
                        valid_peaks.append({"wn": wn, "py": plot_y[idx] + current_baseline, "label": label})
            
            # Sort peaks by Wavenumber to properly stagger adjacent labels
            valid_peaks = sorted(valid_peaks, key=lambda d: d["wn"])

            for p_idx, p_data in enumerate(valid_peaks):
                # Stagger heights to prevent horizontal text collision
                stagger_dist = 25 if p_idx % 2 != 0 else 0
                current_ay = (arrow_direction + stagger_dist) if is_transmittance else (arrow_direction - stagger_dist)

                fig.add_annotation(
                    x=p_data["wn"], y=p_data["py"], 
                    text=f"<b>{p_data['label']}</b>", # BOLD
                    showarrow=True, 
                    arrowhead=1, arrowsize=1, arrowwidth=1.2, arrowcolor="#555555",
                    ay=current_ay, ax=0, 
                    standoff=8, # Force a physical gap between arrow and line
                    font=dict(family="Times New Roman", size=12, color="black"),
                    bgcolor="rgba(255, 255, 255, 0.85)", # Semi-transparent white background to hide lines crossing behind text
                    borderpad=2
                )
            
            # --- 3. Smart Auto-Spacing Math ---
            spectrum_height = plot_y.max() if not is_transmittance else 100
            text_buffer = spectrum_height * 0.35 # Increased buffer for the taller, staggered arrows
            current_baseline += spectrum_height + scaled_offset + text_buffer

        dynamic_plot_height = 600 + (len(master['File'].unique()) * 110)
        fig.update_layout(
            template="simple_white", height=dynamic_plot_height,
            xaxis=dict(title="<b>Wavenumber (cm⁻¹)</b>", range=[4000, 400], **FTIR_STYLE),
            yaxis=dict(title=f"<b>{display_mode} (Stacked)</b>", showticklabels=False, **FTIR_STYLE),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # ---------------------------
    # TAB 2: 2ND DERIVATIVE
    # ---------------------------
    with tab2:
        # --- NEW: Educational Info Box ---
        with st.expander("ℹ️ How to interpret the 2nd Derivative"):
            st.markdown("""
            **Second Derivative Spectroscopy** is a mathematical technique used to artificially enhance the resolution of your spectrum.
            
            * **Revealing Hidden Peaks:** Broad, overlapping absorption bands often hide smaller peaks (which just look like faint "shoulders"). Taking the 2nd derivative dramatically sharpens these features, splitting a single messy bump into distinct, separate signals.
            * **Removing Baselines:** The derivative math automatically cancels out constant (flat) and linear (slanted) baseline errors, making it excellent for highly accurate quantitative analysis.
            * **How to read the plot (⚠️ Important):** Because of how the calculus works, a peak *maximum* in your original absorbance spectrum becomes a **sharp minimum (pointing downwards)** in the 2nd derivative. To find the exact center of a hidden peak, look for the lowest "valleys" on this graph!
            """)

        fig_deriv = go.Figure()
        current_deriv_baseline = 0.0
        
        for name in master['File'].unique():
            df = spectra[name]
            if '2nd_Deriv' in df.columns:
                wavenumbers = df['Wavenumber'].values
                deriv_y = df['2nd_Deriv'].values
                fig_deriv.add_trace(go.Scatter(
                    x=wavenumbers, y=deriv_y + current_deriv_baseline,
                    mode='lines', line=dict(width=1.5), name=name, showlegend=False
                ))
                
                amp_max = deriv_y.max()
                amp_min = deriv_y.min()
                
                anchor_x_deriv = 1950 if 1950 <= wavenumbers.max() else wavenumbers.max()
                idx_anchor_deriv = np.argmin(np.abs(wavenumbers - anchor_x_deriv))
                local_y_deriv = deriv_y[idx_anchor_deriv]
                
                nudge_deriv = (amp_max - amp_min) * 0.08
                label_y_pos_deriv = current_deriv_baseline + local_y_deriv + nudge_deriv

                fig_deriv.add_annotation(
                    x=anchor_x_deriv, y=label_y_pos_deriv, 
                    text=f"<b>{name}</b>", showarrow=False,
                    xanchor='left', yanchor='bottom',
                    font=dict(family="Arial", size=13, color="black")
                )

                total_height = (amp_max - amp_min)
                text_buffer_deriv = total_height * 0.20
                current_deriv_baseline += total_height + (stack_offset * 0.1) + text_buffer_deriv
        
        fig_deriv.update_layout(
            template="simple_white", height=600 + (len(master['File'].unique()) * 90),
            xaxis=dict(title="<b>Wavenumber (cm⁻¹)</b>", range=[2000, 600], **FTIR_STYLE), 
            yaxis=dict(title="<b>d²A/dν² (Stacked)</b>", showticklabels=False, **FTIR_STYLE),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        st.plotly_chart(fig_deriv, use_container_width=True)
    # ---------------------------
    # TAB 3: PEAK SUMMARY TABLE
    # ---------------------------
    with tab3:
        summary_data = []
        is_transmittance = display_mode == "Transmittance (%)"

        for name in master['File'].unique():
            df = spectra[name]
            wavenumbers = df['Wavenumber'].values
            output_y = 100 * (10 ** -df['Absorbance_Norm'].values) if is_transmittance else df['Absorbance_Norm'].values
            for poly in selected_ref:
                for wn, label in POLYMER_DB[poly].items():
                    if wavenumbers.min() <= wn <= wavenumbers.max():
                        idx = np.argmin(np.abs(wavenumbers - wn))
                        summary_data.append({
                            "Sample Name": name, "Material Lib.": poly,
                            "Target Wavenumber (cm⁻¹)": wn,
                            "Actual Peak Wavenumber (cm⁻¹)": round(wavenumbers[idx], 1),
                            "Assignment": label,
                            f"Intensity ({display_mode})": round(output_y[idx], 4)
                        })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            st.download_button("📥 Download Peak Summary (CSV)", summary_df.to_csv(index=False).encode('utf-8'), "Peak_Summary.csv")
        else:
            st.info("Select a Reference Library in the sidebar to generate assignments.")

    # ---------------------------
   # ---------------------------
    # TAB 4: GAUSSIAN DECONVOLUTION
    # ---------------------------
    with tab4:
        st.markdown("### Peak Deconvolution & Gaussian Fitting")
        
        # --- NEW: Educational Info Box ---
        with st.expander("ℹ️ What is Peak Deconvolution?"):
            st.markdown("""
            **Peak Deconvolution** is the mathematical process of separating broad, overlapping spectral bands into individual, underlying peaks.
            
            * **Why Gaussian Fitting?** Infrared absorption bands naturally form bell-shaped curves (Gaussian distributions). By calculating the perfect mathematical curve to fit your raw data, we can isolate overlapping bonds that look like a single messy lump.
            * **How to read the plot:** The **red 'x' marks** show where the algorithm detected a peak crest. The **dashed lines** are the calculated Gaussian curves that best fit the actual shape of your spectrum.
            * **Why it matters:** Finding the exact center of a hidden peak allows for precise molecular identification, and the area under these fitted curves is frequently used to calculate crystallinity or relative concentration.
            """)

        target_fit = st.selectbox("Select Spectrum to Fit", list(spectra.keys()))
        fit_count = st.slider("Number of Major Peaks to Auto-Fit", 1, 10, 3)
        
        df = spectra[target_fit]
        x = df["Wavenumber"].values
        y = df["Absorbance_Norm"].values
        
        fig_fit = go.Figure()
        fig_fit.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Processed Signal", line=dict(color='black')))
        
        peaks = detect_peaks(y, prom=0.03, dist=20)
        if len(peaks) > 0:
            fig_fit.add_trace(go.Scatter(
                x=x[peaks], y=y[peaks], mode="markers",
                marker=dict(size=8, color="red", symbol="x"), name="Detected Peaks"
            ))
            
            peaks_to_fit = peaks[:fit_count]
            for p in peaks_to_fit:
                idx_min = max(0, p - 40)
                idx_max = min(len(x), p + 40)
                
                try:
                    popt, _ = curve_fit(gaussian, x[idx_min:idx_max], y[idx_min:idx_max], p0=[y[p], x[p], 10])
                    curve = gaussian(x, *popt)
                    fig_fit.add_trace(go.Scatter(x=x, y=curve, mode="lines", line=dict(dash="dash"), name=f"Fit {round(x[p],1)} cm⁻¹"))
                except:
                    pass

        fig_fit.update_layout(
            xaxis=dict(title="Wavenumber (cm⁻¹)", range=[4000, 400], autorange="reversed", **FTIR_STYLE),
            yaxis=dict(title="Absorbance", **FTIR_STYLE),
            template="simple_white", height=500
        )
        st.plotly_chart(fig_fit, use_container_width=True)
    # ---------------------------
    # TAB 5: PCA CLUSTERING
    # ---------------------------
    with tab5:
        st.markdown("### Principal Component Analysis (PCA)")
        
        # --- NEW: Educational Info Box ---
        with st.expander("ℹ️ How to interpret this PCA Plot"):
            st.markdown("""
            **Principal Component Analysis (PCA)** is a statistical tool that takes the thousands of data points in your FTIR spectra and compresses them down to the most important underlying trends (Principal Components).
            
            * **Spatial Clustering:** Spectra that are chemically similar will cluster close together on the graph. Spectra that are chemically distinct will be pushed far apart.
            * **What are PC 1 and PC 2?** These are artificial axes created by the algorithm. **PC 1** represents the direction of the *greatest variance* (the biggest differences) among all your samples. **PC 2** represents the second greatest variance.
            * **The Percentages:** The percentages on the axes tell you how much of the total chemical difference in your entire dataset is captured by that specific axis.
            """)

        if len(spectra) < 3:
            st.warning("⚠️ Upload at least 3 spectra to run PCA.")
        else:
            common_x = np.linspace(4000, 400, 1000)
            matrix, labels = [], []
            for name, df in spectra.items():
                matrix.append(np.interp(common_x, df["Wavenumber"].values, df["Absorbance_Norm"].values))
                labels.append(name)

            pca = PCA(n_components=2)
            comps = pca.fit_transform(np.array(matrix))
            var_ratio = pca.explained_variance_ratio_ * 100

            fig_pca = px.scatter(
                x=comps[:,0], y=comps[:,1], text=labels,
                labels={"x": f"PC 1 ({var_ratio[0]:.1f}%)", "y": f"PC 2 ({var_ratio[1]:.1f}%)"}
            )
            fig_pca.update_traces(textposition='top center', marker=dict(size=12))
            fig_pca.update_layout(template="simple_white", height=600)
            fig_pca.update_xaxes(**FTIR_STYLE)
            fig_pca.update_yaxes(**FTIR_STYLE)
            st.plotly_chart(fig_pca, use_container_width=True)
    # ---------------------------
    # TAB 6: SPECTRAL MATCHING
    # ---------------------------
    with tab6:
        st.markdown("### Library Cosine Similarity")
        
        # --- NEW: Educational Info Box ---
        with st.expander("ℹ️ How does Spectral Matching work?"):
            st.markdown("""
            **Cosine Similarity** mathematically compares two spectra by treating their data points as multi-dimensional vectors and measuring the angle between them. 
            
            * **Why it's the standard for FTIR:** It evaluates the *overall shape* and *peak alignment* of the curves. This makes it highly robust against baseline shifts, differences in sample thickness, or varying concentrations.
            * **Interpretation:** A **100%** match indicates the spectral profiles are structurally identical, regardless of their absolute height.
            """)
            
        if len(spectra) < 2:
            st.warning("⚠️ Upload at least 2 spectra to compare.")
        else:
            common_x = np.linspace(4000, 400, 1000)
            library = {name: np.interp(common_x, df["Wavenumber"].values, df["Absorbance_Norm"].values) for name, df in spectra.items()}

            col1, col2 = st.columns([1, 2])
            with col1:
                sample_target = st.selectbox("Select Target Spectrum", list(library.keys()))
                results = match_spectrum(library[sample_target], library)
                matches = [r for r in results if r[0] != sample_target]
                
                st.write("#### Top Matches")
                for r in matches:
                    # --- NEW: Convert decimal to Percentage ---
                    sim_score = float(r[1])
                    sim_pct = sim_score * 100
                    
                    # Ensure progress bar stays within 0.0 - 1.0 limits safely
                    safe_bar_val = max(0.0, min(1.0, sim_score))
                    
                    # Display as a percentage (e.g., 98.5%)
                    st.progress(safe_bar_val, text=f"{r[0]} ({sim_pct:.1f}%)")
                    
            with col2:
                if matches:
                    top_match_name = matches[0][0]
                    fig_match = go.Figure()
                    fig_match.add_trace(go.Scatter(x=common_x, y=library[sample_target], name=f"Target: {sample_target}"))
                    fig_match.add_trace(go.Scatter(x=common_x, y=library[top_match_name], name=f"Match: {top_match_name}", line=dict(dash='dash')))
                    fig_match.update_layout(
                        xaxis=dict(title="Wavenumber", autorange="reversed", **FTIR_STYLE),
                        yaxis=dict(title="Absorbance", **FTIR_STYLE),
                        template="simple_white", height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_match, use_container_width=True)
    # ---------------------------
    # TAB 7: DATA MATRIX
    # ---------------------------
    with tab7:
        common_wn = np.linspace(4000, 400, 1500)
        export_data = {"Wavenumber": common_wn}
        is_transmittance = display_mode == "Transmittance (%)"
        
        for name, df in spectra.items():
            output_y = 100 * (10 ** -df['Absorbance_Norm'].values) if is_transmittance else df['Absorbance_Norm'].values
            export_data[name] = np.interp(common_wn, df['Wavenumber'].values, output_y)
        
        matrix_df = pd.DataFrame(export_data)
        st.download_button(
            f"📥 Download Standardized Data Matrix ({display_mode})", 
            matrix_df.to_csv(index=False).encode('utf-8'), 
            f"FTIR_Matrix_{display_mode[:3]}.csv"
        )
        st.dataframe(matrix_df.head(15))

else:
    st.info("Upload your raw data files in the sidebar to begin.")
