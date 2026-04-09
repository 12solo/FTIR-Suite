import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from scipy.stats import linregress # <-- Ensure this is here
from scipy import sparse
import os
import re
import io
import base64

# ==========================================
# PAGE CONFIG — must be first Streamlit call
# ==========================================
st.set_page_config(
    page_title="FTIR Pro Suite | Solomon Scientific",
    page_icon="SR.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# GLOBAL CUSTOM CSS — Full Light Theme
# ==========================================
st.markdown("""
<style>
/* ── Google Fonts ─────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── CSS Variables ────────────────────────────── */
:root {
    /* Brand Colors */
    --navy:       #0b1120;
    --navy-mid:   #111827;
    --navy-light: #1a2540;
    --gold:       #c9a84c;
    --gold-light: #e2c97e;
    --gold-dim:   #9c7a32;
    
    /* Light Mode Colors */
    --bg-white:   #ffffff;
    --bg-offwhite:#f8fafc;
    --text-dark:  #000000; /* Pure Dark Black for normal text */
    --text-muted: #111111; /* Almost black for secondary text */
    --border-light:#e2e8f0;
    
    --accent:     #3a7bd5;
    --red:        #e05252;
    --green:      #3db87a;
    
    --font-head:  'Playfair Display', Georgia, serif;
    --font-mono:  'IBM Plex Mono', 'Courier New', monospace;
    --font-body:  'IBM Plex Sans', 'Segoe UI', sans-serif;
}

/* ── Base & Body ──────────────────────────────── */
html, body, [class*="css"] {
    font-family: var(--font-body);
    color: var(--text-dark);
}
.stApp {
    background: var(--bg-white);
}
.stApp::before { display: none; }

/* ── Sidebar (Pure White & User Friendly) ─────── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid var(--border-light);
}

/* Fixed material icon bug and force pure black text */
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
    color: #000000 !important;
    font-family: var(--font-body);
}
.material-symbols-rounded,
[data-testid="stIconMaterial"] {
    font-family: "Material Symbols Rounded" !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--gold-dim) !important;
    font-weight: 700;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}
[data-testid="stSidebar"] hr { border-color: var(--border-light); margin: 1rem 0; }

/* Sidebar Inputs */
[data-testid="stSidebar"] input[type="text"],
[data-testid="stSidebar"] input[type="number"],
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] select {
    background: var(--bg-white) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 4px !important;
    color: #000000 !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
}

/* File Uploader Dropzone */
[data-testid="stFileUploadDropzone"] {
    background-color: var(--bg-white) !important;
    border: 2px dashed #cbd5e1 !important;
    border-radius: 6px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: var(--gold) !important;
    background-color: var(--bg-offwhite) !important;
}

/* ── Main Area Inputs ─────────────────────────── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: var(--bg-white) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 4px !important;
    color: #000000 !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
}
.stSelectbox > div > div:hover,
.stTextInput > div > div > input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 1px var(--gold-dim) !important;
}

/* ── Buttons ──────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--gold-dim), var(--gold)) !important;
    color: var(--navy) !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.45rem 1rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--gold), var(--gold-light)) !important;
    box-shadow: 0 4px 15px rgba(201,168,76,0.3) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #8b1a1a, var(--red)) !important;
    color: white !important;
}

/* Download buttons */
[data-testid="stDownloadButton"] > button {
    background: var(--bg-offwhite) !important;
    color: var(--navy) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 3px !important;
    font-weight: 600 !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #ffffff !important;
    border-color: var(--gold) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
}

/* ── Tabs ─────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    background: var(--bg-offwhite);
    border-bottom: 1px solid var(--border-light);
    gap: 0; padding: 0;
}
[data-testid="stTabs"] [role="tab"] {
    color: #000000 !important;
    font-family: var(--font-body) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.7rem 1.2rem !important;
    border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] [role="tab"]:hover {
    color: var(--navy) !important;
    background: rgba(0,0,0,0.02) !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #000000 !important;
    border-bottom-color: var(--gold) !important;
    background: var(--bg-white) !important;
}

/* ── DataFrames ───────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-light) !important;
    border-radius: 6px !important;
    background: var(--bg-white) !important;
}
[data-testid="stDataFrame"] th {
    background: var(--bg-offwhite) !important;
    color: #000000 !important;
    border-bottom: 1px solid var(--border-light) !important;
}
[data-testid="stDataFrame"] td {
    color: #000000 !important;
}

/* ── Expanders ────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid var(--border-light) !important;
    border-radius: 4px !important;
    background: var(--bg-white) !important;
}
[data-testid="stExpander"] summary {
    color: #000000 !important;
    font-weight: 700 !important;
}

/* ── Text Area & Selectors ────────────────────── */
.stTextArea textarea {
    background: var(--bg-white) !important;
    border: 1px solid var(--border-light) !important;
    color: #000000 !important;
}
[data-baseweb="tag"] {
    background: var(--bg-offwhite) !important;
    border: 1px solid var(--border-light) !important;
}
[data-baseweb="tag"] span { color: #000000 !important; }

/* ── Alerts ───────────────────────────────────── */
[data-testid="stAlert"] { color: #000000 !important; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# HELPER COMPONENTS
# ==========================================
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def render_header():
    logo_path = "SR.png"
    if os.path.exists(logo_path):
        img_b64 = get_base64_of_bin_file(logo_path)
        icon_html = f'<img src="data:image/png;base64,{img_b64}" style="width: 54px; height: 54px; border-radius: 8px; object-fit: contain; box-shadow: 0 4px 20px rgba(0,0,0,0.5); flex-shrink: 0; background: white;">'
    else:
        icon_html = '<div style="width: 54px; height: 54px; background: linear-gradient(135deg, #9c7a32, #c9a84c); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1.6rem; box-shadow: 0 4px 20px rgba(0,0,0,0.3); flex-shrink: 0;">🔬</div>'

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #0b1120 0%, #0f1a2e 100%);
        padding: 1.5rem 2rem;
        border-radius: 8px;
        border: 1px solid rgba(201,168,76,0.3);
        margin-bottom: 1.5rem;
        margin-top: 1rem;
        display: flex;
        align-items: center;
        gap: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    ">
        {icon_html}
        <div>
            <div style="
                font-family: 'Playfair Display', Georgia, serif;
                font-size: 1.75rem;
                font-weight: 700;
                color: #f0f4fb;
                letter-spacing: 0.01em;
                line-height: 1.1;
            ">FTIR Pro Suite <span style="color:#c9a84c;">5.0</span></div>
            <div style="
                font-family: 'IBM Plex Sans', sans-serif;
                font-size: 0.72rem;
                color: #a8b4c8;
                letter-spacing: 0.2em;
                text-transform: uppercase;
                margin-top: 2px;
            ">Spectroscopy Analysis Suite &nbsp;·&nbsp; Solomon Scientific &nbsp;·&nbsp; © 2026</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def metric_card(label, value, unit="", delta=None):
    delta_html = ""
    if delta is not None:
        color = "#3db87a" if delta >= 0 else "#e05252"
        arrow = "▲" if delta >= 0 else "▼"
        delta_html = f'<div style="color:{color};font-size:0.7rem;margin-top:2px;">{arrow} {abs(delta):.3f}</div>'
    return f"""
    <div style="
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        border-top: 3px solid #c9a84c;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    ">
        <div style="font-family:'IBM Plex Sans',sans-serif;font-size:0.68rem;color:#000000;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:4px;font-weight:700;">{label}</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:1.35rem;color:#000000;font-weight:700;">{value}<span style="font-size:0.7rem;color:#000000;margin-left:4px;">{unit}</span></div>
        {delta_html}
    </div>
    """

def section_title(text, icon=""):
    st.markdown(f"""
    <div style="
        display:flex; align-items:center; gap:0.6rem;
        background: linear-gradient(90deg, #0b1120 0%, #1a2540 100%);
        padding: 0.6rem 1.25rem;
        border-radius: 6px;
        border-left: 4px solid #c9a84c;
        margin: 1.5rem 0 1rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    ">
        <span style="font-size:1.1rem; color:#f0f4fb;">{icon}</span>
        <span style="
            font-family:'IBM Plex Sans',sans-serif;
            font-size:0.8rem;
            font-weight:600;
            color:#f0f4fb;
            letter-spacing:0.15em;
            text-transform:uppercase;
        ">{text}</span>
    </div>
    """, unsafe_allow_html=True)

def info_box(text, kind="info"):
    colors = {
        "info":    ("#3a7bd5", "rgba(58,123,213,0.08)"),
        "success": ("#3db87a", "rgba(61,184,122,0.08)"),
        "warning": ("#c9a84c", "rgba(201,168,76,0.08)"),
        "error":   ("#e05252", "rgba(224,82,82,0.08)"),
    }
    border, bg = colors.get(kind, colors["info"])
    icon = {"info": "ℹ", "success": "✓", "warning": "⚠", "error": "✕"}.get(kind, "ℹ")
    st.markdown(f"""
    <div style="
        background:{bg}; border-left:4px solid {border};
        border-radius:4px; padding:0.75rem 1rem;
        font-family:'IBM Plex Sans',sans-serif; font-size:0.85rem; color:#000000;
        margin:0.5rem 0; font-weight:500;
    "><span style="color:{border};margin-right:0.5rem;font-weight:bold;">{icon}</span>{text}</div>
    """, unsafe_allow_html=True)

def render_sidebar_brand():
    logo_path = "SR.png"
    if os.path.exists(logo_path):
        img_b64 = get_base64_of_bin_file(logo_path)
        icon_html = f'<img src="data:image/png;base64,{img_b64}" style="width: 52px; height: 52px; margin: 0 auto 0.75rem auto; border-radius: 10px; display: block; box-shadow: 0 4px 12px rgba(0,0,0,0.1); object-fit: contain; background: white;">'
    else:
        icon_html = '<div style="width:52px; height:52px; margin:0 auto 0.75rem auto; background:linear-gradient(135deg,#9c7a32,#c9a84c); border-radius:10px; display:flex;align-items:center;justify-content:center; font-size:1.5rem; box-shadow:0 4px 12px rgba(0,0,0,0.1);">🔬</div>'

    st.markdown(f"""
    <div style="padding: 1.25rem 0 0.5rem 0; text-align:center;">
        {icon_html}
        <div style="
            font-family:'IBM Plex Sans',sans-serif;
            font-size:0.65rem;
            color:#9c7a32;
            letter-spacing:0.2em;
            text-transform:uppercase;
            margin-bottom:4px;
        ">Solomon Scientific</div>
        <div style="
            font-family:'Playfair Display',Georgia,serif;
            font-size:1.1rem;
            font-weight:700;
            color:#000000;
        ">FTIR Pro Suite <span style="color:#c9a84c;">5.0</span></div>
        <div style="
            margin-top:0.75rem;
            padding-top:0.75rem;
            border-top:1px solid #e2e8f0;
            font-family:'IBM Plex Sans',sans-serif;
            font-size:0.68rem;
            color:#000000;
            font-weight:500;
        ">Advanced Spectroscopy Tools<br>
        <a href='mailto:your.solomon.duf@gmail.com'
           style='color:#9c7a32;text-decoration:none;'>
            ✉ Contact Developer
        </a>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# ADVANCED MATH & UTILS
# ==========================================
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

# ==========================================
# PLOTLY THEME (STRICT JOURNAL QUALITY)
# ==========================================
PLOT_BG    = "#ffffff"
PAPER_BG   = "#ffffff"
BLACK      = "#000000"
GOLD       = "#c9a84c"
WHITE      = "#ffffff"
WHITE_TXT  = "#000000" # Set text to pure black

FTIR_STYLE = dict(
    mirror=True, 
    ticks='inside', 
    showline=True,
    linecolor=BLACK, 
    linewidth=2,
    showgrid=False,   # Strict NO GRIDLINES
    zeroline=False,
    title_font=dict(family="Arial", size=18, color=BLACK),
    tickfont=dict(family="Arial", size=14, color=BLACK),
    tickwidth=2, 
    ticklen=6, 
    tickcolor=BLACK,
)

JOURNAL_CONFIG = {
    # High resolution export scale (approx 500-600 DPI)
    'toImageButtonOptions': {'format': 'png', 'filename': 'FTIR_Journal_Plot', 'scale': 5},
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
}

PALETTE = [
    "#000000", "#d62728", "#1f77b4", "#2ca02c", 
    "#ff7f0e", "#9467bd", "#8c564b", "#e377c2",
    "#7f7f7f", "#bcbd22", "#17becf"
]

POLYMER_DB = {
    "TPV (PP/EPDM)": {
        2950: "CH3 asym (PP)", 2918: "CH2 asym (Backbone)", 2849: "CH2 sym (Backbone)",
        1455: "CH2 bend", 1376: "CH3 sym (PP)", 1167: "C-C/CH wag (PP cryst)",
        997: "CH3 rock (Isotactic)", 973: "CH3 rock (Helical)", 841: "CH2 rock (PP cryst)",
        720: "CH2 rock (EPDM)", 1640: "C=C (EPDM unsat)"
    },
    "Peroxide Cured EPDM": {
        2918: "CH2 asym", 2849: "CH2 sym", 1460: "CH2 bend", 1375: "CH3 bend",
        720: "CH2 rock", 1100: "C-O-C (Ether cross)", 1060: "C-O str (Oxid)",
        1735: "C=O (Ketone/Ald)", 1715: "C=O (Acid/Ester)", 1640: "C=C", 3400: "O-H (Oxid)"
    },
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

# ==========================================
# SESSION STATE
# ==========================================
if 'ftir_master_df' not in st.session_state:
    st.session_state['ftir_master_df'] = pd.DataFrame()
if 'spectra_storage' not in st.session_state:
    st.session_state['spectra_storage'] = {}

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    render_sidebar_brand()

    st.markdown("### 1 · Data Formatting")
    raw_data_format = st.radio("Uploaded Data Format", ["Absorbance", "Transmittance (%)"])
    display_mode = st.radio("Output Display Mode", ["Absorbance", "Transmittance (%)"])
    
    with st.expander("🔬 Advanced Math Settings", expanded=False):
        apply_atr = st.checkbox("Apply ATR Correction")
        smooth_val = st.slider("Savitzky-Golay Window", 5, 101, 15, step=2)
        apply_baseline = st.checkbox("Apply ALS Baseline Correction", True)
        if apply_baseline:
            als_lam = st.selectbox("Baseline Stiffness (Lambda)", [1e3, 1e4, 1e5, 1e6], index=2)
            als_p = st.selectbox("Baseline Asymmetry (p)", [0.001, 0.01, 0.05, 0.1], index=1)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 2 · Plot Formatting")
    stack_offset = st.slider("Vertical Offset Cushion", 0.0, 1.0, 0.2, step=0.05)
    line_w = st.slider("Line Weight", 1.0, 4.0, 2.0)
    selected_ref = st.multiselect("Label Peaks (DB)", list(POLYMER_DB.keys()), default=["General / Unknown"])
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 3 · Data Input")
    with st.form("ftir_upload_form", clear_on_submit=True):
        group_id = st.text_input("Sample Group ID", "Experimental_Batch")
        files = st.file_uploader("Upload FTIR Data (.csv, .txt, .xlsx)", accept_multiple_files=True)
        submit = st.form_submit_button("⚙️ Process Batch Data", use_container_width=True)

    if submit and files:
        for f in files:
            name = clean_name(f.name)
            if name not in st.session_state['spectra_storage']:
                with st.spinner(f"Processing {name}..."):
                    try:
                        df = None
                        if f.name.lower().endswith(('.xls', '.xlsx')):
                            try:
                                df = pd.read_excel(f, header=None)
                            except Exception: pass 
                        
                        if df is None:
                            f.seek(0)
                            content = f.getvalue().decode('utf-8', errors='ignore').split('\n')
                            parsed_data = []
                            for line in content:
                                line = line.strip()
                                if not line: continue
                                parts = re.split(r'[,\t;|\s]+', line)
                                if len(parts) >= 2:
                                    try:
                                        val_x = float(parts[0])
                                        val_y = float(parts[1])
                                        parsed_data.append([val_x, val_y])
                                    except ValueError:
                                        pass 
                            
                            if not parsed_data:
                                raise ValueError("Could not extract any numerical data from file.")
                                
                            df = pd.DataFrame(parsed_data)

                        df = df.iloc[:, :2] 
                        df.columns = ['Wavenumber', 'Raw_Intensity']
                        df['Wavenumber'] = pd.to_numeric(df['Wavenumber'], errors='coerce')
                        df['Raw_Intensity'] = pd.to_numeric(df['Raw_Intensity'], errors='coerce')
                        df = df.dropna().sort_values('Wavenumber', ascending=True).reset_index(drop=True)

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
                        st.success(f"✓ Loaded: {name}")

                    except Exception as e:
                        st.error(f"Error processing {f.name}: {e}")

    # --- Manage Data / Reset ---
    if not st.session_state['ftir_master_df'].empty:
        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("🔄 Reset Entire Workspace", type="primary", use_container_width=True):
            st.session_state['ftir_master_df'] = pd.DataFrame()
            st.session_state['spectra_storage'] = {}
            st.rerun()

    st.markdown("""
    <div style="padding:1rem 0 0.5rem;text-align:center;font-family:'IBM Plex Sans',sans-serif;
                font-size:0.65rem;color:#000000;letter-spacing:0.1em;font-weight:500;">
        For Research & Academic Use Only<br>Version 5.0 Pro
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# MAIN CONTENT
# ==========================================
master = st.session_state['ftir_master_df']
spectra = st.session_state['spectra_storage']

render_header()

if not master.empty:
    n_files = len(master)
    n_groups = master['Group'].nunique()
    
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(metric_card("Spectra Loaded", f"{n_files}", ""), unsafe_allow_html=True)
    k2.markdown(metric_card("Sample Groups", f"{n_groups}", ""), unsafe_allow_html=True)
    k3.markdown(metric_card("Processing Mode", "Active", ""), unsafe_allow_html=True)
    k4.markdown(metric_card("Baseline/Smooth", "ON" if apply_baseline else "OFF", ""), unsafe_allow_html=True)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📉 Primary Spectra", 
        "🔬 2nd Derivative", 
        "📋 Peak Assignments", 
        "📈 Deconvolution", 
        "🧠 PCA Clustering", 
        "🧪 Spectral Match", 
        "📊 Data Export",
        "📚 Method",
        "⏱️ EPDM KOH Aging" # NEW TAB
    ])

    # ---------------------------
    # TAB 1: PRIMARY SPECTRA 
    # ---------------------------
    with tab1:
        section_title("Stacked Primary Spectra", "📉")
        fig = go.Figure()
        current_baseline = 0.0 
        is_transmittance = display_mode == "Transmittance (%)"
        scaled_offset = stack_offset * 100 if is_transmittance else stack_offset
        
        arrow_direction = 60 if is_transmittance else -60
        
        for i, name in enumerate(master['File'].unique()):
            df = spectra[name]
            wavenumbers = df['Wavenumber'].values
            plot_y = 100 * (10 ** -df['Absorbance_Norm'].values) if is_transmittance else df['Absorbance_Norm'].values
            color = PALETTE[i % len(PALETTE)]

            fig.add_trace(go.Scatter(
                x=wavenumbers, y=plot_y + current_baseline,
                mode='lines', line=dict(width=line_w, color=color), 
                name=name, hoverinfo="name+x+y", showlegend=False
            ))

            anchor_x = 3900 if 3900 <= wavenumbers.max() else wavenumbers.max()
            idx_anchor = np.argmin(np.abs(wavenumbers - anchor_x))
            local_y = plot_y[idx_anchor] 

            nudge = (plot_y.max() - plot_y.min()) * 0.08
            label_y_pos = current_baseline + local_y + (nudge if not is_transmittance else -nudge)

            # Trace Name Label (Journal Style)
            fig.add_annotation(
                x=anchor_x, y=label_y_pos, 
                text=f"{name}", showarrow=False,
                xanchor='left', yanchor='bottom' if not is_transmittance else 'top',
                font=dict(family="Arial", size=14, color=color)
            )

            valid_peaks = []
            for poly in selected_ref:
                for wn, label in POLYMER_DB[poly].items():
                    if wavenumbers.min() <= wn <= wavenumbers.max():
                        idx = np.argmin(np.abs(wavenumbers - wn))
                        valid_peaks.append({"wn": wn, "py": plot_y[idx] + current_baseline, "label": label})
            
            valid_peaks = sorted(valid_peaks, key=lambda d: d["wn"])

            for p_idx, p_data in enumerate(valid_peaks):
                stagger_dist = 25 if p_idx % 2 != 0 else 0
                current_ay = (arrow_direction + stagger_dist) if is_transmittance else (arrow_direction - stagger_dist)

                # Peak Annotation (Journal Style) - NO BORDER
                fig.add_annotation(
                    x=p_data["wn"], y=p_data["py"], 
                    text=f"{p_data['label']}", 
                    showarrow=True, 
                    arrowhead=1, arrowsize=1, arrowwidth=1.5, arrowcolor=BLACK,
                    ay=current_ay, ax=0, 
                    standoff=5, 
                    font=dict(family="Arial", size=12, color=BLACK),
                    bgcolor=PAPER_BG, borderpad=3  # Removed bordercolor and borderwidth
                )
            
            spectrum_height = plot_y.max() - plot_y.min() if not is_transmittance else 100
            text_buffer = spectrum_height * 0.35 
            current_baseline += spectrum_height + scaled_offset + text_buffer

        dynamic_plot_height = max(600, 200 + (len(master['File'].unique()) * 120))
        fig.update_layout(
            height=dynamic_plot_height,
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
            xaxis=dict(title="<b>Wavenumber (cm⁻¹)</b>", range=[4000, 400], **FTIR_STYLE),
            yaxis=dict(title=f"<b>{display_mode} (Stacked)</b>", showticklabels=False, **FTIR_STYLE),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        st.plotly_chart(fig, use_container_width=True, config=JOURNAL_CONFIG)

    # ---------------------------
    # TAB 2: 2ND DERIVATIVE
    # ---------------------------
    with tab2:
        section_title("Second Derivative Resolution", "🔬")
        with st.expander("ℹ️ How to interpret the 2nd Derivative"):
            st.markdown("""
            **Second Derivative Spectroscopy** enhances spectral resolution mathematically.
            
            * **Revealing Hidden Peaks:** Broad, overlapping absorption bands hide smaller peaks. The 2nd derivative sharpens these.
            * **Removing Baselines:** Constant (flat) and linear (slanted) baseline errors are cancelled out.
            * **How to read the plot (⚠️ Important):** A peak *maximum* in original absorbance becomes a **sharp minimum (pointing downwards)** in the 2nd derivative. Look for the lowest "valleys"!
            """)

        fig_deriv = go.Figure()
        current_deriv_baseline = 0.0
        
        for i, name in enumerate(master['File'].unique()):
            df = spectra[name]
            if '2nd_Deriv' in df.columns:
                wavenumbers = df['Wavenumber'].values
                deriv_y = df['2nd_Deriv'].values
                color = PALETTE[i % len(PALETTE)]
                
                fig_deriv.add_trace(go.Scatter(
                    x=wavenumbers, y=deriv_y + current_deriv_baseline,
                    mode='lines', line=dict(width=1.5, color=color), name=name, showlegend=False
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
                    text=f"{name}", showarrow=False,
                    xanchor='left', yanchor='bottom',
                    font=dict(family="Arial", size=14, color=color)
                )

                total_height = (amp_max - amp_min)
                text_buffer_deriv = total_height * 0.20
                current_deriv_baseline += total_height + (stack_offset * 0.1) + text_buffer_deriv
        
        fig_deriv.update_layout(
            height=max(600, 200 + (len(master['File'].unique()) * 100)),
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
            xaxis=dict(title="<b>Wavenumber (cm⁻¹)</b>", range=[2000, 600], **FTIR_STYLE), 
            yaxis=dict(title="<b>d²A/dν² (Stacked)</b>", showticklabels=False, **FTIR_STYLE),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        st.plotly_chart(fig_deriv, use_container_width=True, config=JOURNAL_CONFIG)

    # ---------------------------
    # TAB 3: PEAK SUMMARY TABLE
    # ---------------------------
    with tab3:
        section_title("Tabular Peak Assignments", "📋")
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
            st.dataframe(summary_df, use_container_width=True, height=400)
            
            csv = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Peak Summary (CSV)", csv, "Peak_Summary.csv")
        else:
            info_box("Select a Reference Library in the sidebar to generate assignments.", "warning")

    # ---------------------------
    # TAB 4: GAUSSIAN DECONVOLUTION
    # ---------------------------
    with tab4:
        section_title("Peak Deconvolution & Gaussian Fitting", "📈")
        
        with st.expander("ℹ️ What is Peak Deconvolution?"):
            st.markdown("""
            **Peak Deconvolution** mathematically separates broad, overlapping spectral bands into individual, underlying peaks.
            
            * **Why Gaussian Fitting?** Infrared absorption bands form bell-shaped curves. Calculating the perfect curve isolates overlapping bonds.
            * **How to read the plot:** The **red 'x' marks** show detected crests. **Dashed lines** are calculated Gaussian curves.
            """)

        colA, colB = st.columns([2, 1])
        with colA: target_fit = st.selectbox("Select Spectrum to Fit", list(spectra.keys()))
        with colB: fit_count = st.slider("Number of Major Peaks to Auto-Fit", 1, 10, 3)
        
        df = spectra[target_fit]
        x = df["Wavenumber"].values
        y = df["Absorbance_Norm"].values
        
        fig_fit = go.Figure()
        fig_fit.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Processed Signal", line=dict(color=BLACK, width=2)))
        
        peaks = detect_peaks(y, prom=0.03, dist=20)
        if len(peaks) > 0:
            fig_fit.add_trace(go.Scatter(
                x=x[peaks], y=y[peaks], mode="markers",
                marker=dict(size=8, color="#e05252", symbol="x", line=dict(width=1.5, color="#e05252")), name="Detected Peaks"
            ))
            
            peaks_to_fit = peaks[:fit_count]
            for p in peaks_to_fit:
                idx_min = max(0, p - 40)
                idx_max = min(len(x), p + 40)
                
                try:
                    popt, _ = curve_fit(gaussian, x[idx_min:idx_max], y[idx_min:idx_max], p0=[y[p], x[p], 10])
                    curve = gaussian(x, *popt)
                    fig_fit.add_trace(go.Scatter(x=x, y=curve, mode="lines", line=dict(dash="dash", color=GOLD, width=2), name=f"Fit {round(x[p],1)} cm⁻¹"))
                except:
                    pass

        fig_fit.update_layout(
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
            xaxis=dict(title="<b>Wavenumber (cm⁻¹)</b>", range=[4000, 400], autorange="reversed", **FTIR_STYLE),
            yaxis=dict(title="<b>Absorbance</b>", **FTIR_STYLE),
            height=550, margin=dict(l=60, r=40, t=40, b=60),
            legend=dict(bgcolor=WHITE, bordercolor=BLACK, borderwidth=1.5, font=dict(family="Arial", size=12, color=BLACK))
        )
        st.plotly_chart(fig_fit, use_container_width=True, config=JOURNAL_CONFIG)

    # ---------------------------
    # TAB 5: PCA CLUSTERING
    # ---------------------------
    with tab5:
        section_title("PCA", "🧠")
        
        with st.expander("ℹ️ How to interpret this PCA Plot"):
            st.markdown("""
            **PCA** compresses thousands of spectral data points into the most important underlying trends.
            
            * **Spatial Clustering:** Chemically similar spectra cluster together; distinct spectra are pushed apart.
            * **What are PC 1 and PC 2?** **PC 1** represents the direction of *greatest variance* among samples. **PC 2** is the second greatest.
            """)

        if len(spectra) < 3:
            info_box("Upload at least 3 spectra to run PCA clustering.", "warning")
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
                labels={"x": f"<b>PC 1 ({var_ratio[0]:.1f}%)</b>", "y": f"<b>PC 2 ({var_ratio[1]:.1f}%)</b>"}
            )
            fig_pca.update_traces(
                textposition='top center', 
                marker=dict(size=12, color=BLACK, line=dict(color=BLACK, width=1)),
                textfont=dict(family="Arial", size=14, color=BLACK)
            )
            fig_pca.update_layout(
                plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG, height=600,
                margin=dict(l=60, r=60, t=40, b=60),
                xaxis=FTIR_STYLE, yaxis=FTIR_STYLE
            )
            st.plotly_chart(fig_pca, use_container_width=True, config=JOURNAL_CONFIG)

    # ---------------------------
    # TAB 6: SPECTRAL MATCHING
    # ---------------------------
    with tab6:
        section_title("Library Cosine Similarity", "🧪")
        
        with st.expander("ℹ️ How does Spectral Matching work?"):
            st.markdown("""
            **Cosine Similarity** compares two spectra by treating their data points as multi-dimensional vectors and measuring the angle between them. 
            
            * **Why it's the standard:** It evaluates the *overall shape* and *peak alignment*, making it highly robust against baseline shifts or concentration changes.
            * **Interpretation:** A **100%** match indicates structurally identical spectral profiles.
            """)
            
        if len(spectra) < 2:
            info_box("Upload at least 2 spectra to run comparison matching.", "warning")
        else:
            common_x = np.linspace(4000, 400, 1000)
            library = {name: np.interp(common_x, df["Wavenumber"].values, df["Absorbance_Norm"].values) for name, df in spectra.items()}

            col1, col2 = st.columns([1, 2])
            with col1:
                sample_target = st.selectbox("Select Target Spectrum", list(library.keys()))
                results = match_spectrum(library[sample_target], library)
                matches = [r for r in results if r[0] != sample_target]
                
                st.markdown("<br><h4 style='color:#000000;font-family:Arial;'>Top Matches</h4>", unsafe_allow_html=True)
                for r in matches:
                    sim_score = float(r[1])
                    sim_pct = sim_score * 100
                    safe_bar_val = max(0.0, min(1.0, sim_score))
                    st.progress(safe_bar_val, text=f"{r[0]} ({sim_pct:.1f}%)")
                    
            with col2:
                if matches:
                    top_match_name = matches[0][0]
                    fig_match = go.Figure()
                    fig_match.add_trace(go.Scatter(x=common_x, y=library[sample_target], name=f"Target: {sample_target}", line=dict(color=BLACK, width=2)))
                    fig_match.add_trace(go.Scatter(x=common_x, y=library[top_match_name], name=f"Match: {top_match_name}", line=dict(dash='dash', color=GOLD, width=2)))
                    fig_match.update_layout(
                        xaxis=dict(title="<b>Wavenumber (cm⁻¹)</b>", autorange="reversed", **FTIR_STYLE),
                        yaxis=dict(title="<b>Absorbance</b>", **FTIR_STYLE),
                        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG, height=500,
                        margin=dict(l=60, r=40, t=40, b=60),
                        legend=dict(
                            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, 
                            bgcolor=WHITE, bordercolor=BLACK, borderwidth=1.5,
                            font=dict(family="Arial", size=12, color=BLACK)
                        )
                    )
                    st.plotly_chart(fig_match, use_container_width=True, config=JOURNAL_CONFIG)

    # ---------------------------
    # TAB 7: DATA MATRIX
    # ---------------------------
    with tab7:
        section_title("Data Export", "📊")
        st.markdown("<p style='color:#000000;'>Interpolates all processed spectra onto a unified wavenumber axis for external machine learning or plotting.</p>", unsafe_allow_html=True)
        
        common_wn = np.linspace(4000, 400, 1500)
        export_data = {"Wavenumber": common_wn}
        is_transmittance = display_mode == "Transmittance (%)"
        
        for name, df in spectra.items():
            output_y = 100 * (10 ** -df['Absorbance_Norm'].values) if is_transmittance else df['Absorbance_Norm'].values
            export_data[name] = np.interp(common_wn, df['Wavenumber'].values, output_y)
        
        matrix_df = pd.DataFrame(export_data)
        
        csv = matrix_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            f"📥 Download Standardized Data Matrix ({display_mode})", 
            csv, 
            f"FTIR_Matrix_{display_mode[:3]}.csv"
        )
        st.dataframe(matrix_df.head(15), use_container_width=True)
# ---------------------------
    # TAB 8: METHODOLOGIES
    # ---------------------------
    with tab8:
        section_title("Analytical Methodologies & Algorithms", "📚")
        
        st.markdown("""
        <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.9rem; color:#000000; line-height:1.6;">
        This suite implements standard chemometric and spectral processing algorithms. The following methodologies dictate the data pipeline:

        <br><h4 style="font-family:'Playfair Display',serif; color:#c9a84c; margin-bottom:0.2rem;">1. Spectral Preprocessing</h4>
        <b>Transmittance to Absorbance Conversion</b><br>
        For transmittance data, the conversion to absorbance (A) is calculated as:
        <br><i>A = 2 - log₁₀(%T)</i><br><br>

        <b>Attenuated Total Reflectance (ATR) Correction</b><br>
        ATR spectra exhibit a wavenumber-dependent penetration depth. The correction scales the absorbance by the wavenumber (ν) to simulate a transmission spectrum:
        <br><i>A_corrected = A_raw × (ν / 1000)</i><br><br>

        <b>Savitzky-Golay Smoothing</b><br>
        A local polynomial regression is fitted to a sliding window of consecutive data points. This increases the signal-to-noise ratio without significantly distorting the signal morphology.<br><br>

        <b>Asymmetric Least Squares (ALS) Baseline Correction</b><br>
        The ALS algorithm estimates the baseline <i>z</i> for a given spectrum <i>y</i> by minimizing the penalized least squares objective function:
        <br><i>S = Σ w_i(y_i - z_i)² + λ Σ(Δ²z_i)²</i><br>
        where <i>w_i</i> is an asymmetric weighting function (p if y_i > z_i, and 1-p otherwise), λ controls the baseline stiffness, and Δ² is the second difference operator.

        <br><br><h4 style="font-family:'Playfair Display',serif; color:#c9a84c; margin-bottom:0.2rem;">2. Spectral Resolution & Deconvolution</h4>
        <b>Second Derivative Spectroscopy</b><br>
        Calculated using the Savitzky-Golay method, the second derivative (d²A/dν²) isolates overlapping bands and eliminates constant and linear baseline offsets. Peak maxima in original absorbance spectra appear as sharp minima in the second derivative.<br><br>

        <b>Gaussian Deconvolution</b><br>
        Complex overlapping bands are modeled as a linear combination of Gaussian functions:
        <br><i>f(x) = a · exp(-(x - x₀)² / 2σ²)</i><br>
        where <i>a</i> is the amplitude, <i>x₀</i> is the peak center, and <i>σ</i> is the standard deviation. A non-linear least squares solver optimizes these parameters to reconstruct the underlying bands.

        <br><br><h4 style="font-family:'Playfair Display',serif; color:#c9a84c; margin-bottom:0.2rem;">3. Multivariate Analysis & Identification</h4>
        <b>Principal Component Analysis (PCA)</b><br>
        PCA applies an orthogonal transformation to convert correlated spectral variables into a set of linearly uncorrelated principal components. It solves the eigenvalue problem for the data's covariance matrix to maximize variance capture in lower dimensions.<br><br>

        <b>Library Matching (Cosine Similarity)</b><br>
        Spectral matching is quantified by treating spectra as n-dimensional vectors and measuring the cosine of the angle between the unknown spectrum vector (A) and the reference vector (B):
        <br><i>Similarity = (A · B) / (||A|| ||B||)</i><br>
        This normalized dot product provides a robust structural metric that is highly invariant to scalar multiplication, such as changes in sample thickness or concentration.
        </div>
        """, unsafe_allow_html=True)
# ---------------------------
    # ---------------------------
    # TAB 9: EPDM KOH AGING TRACKER - ADVANCED RESEARCH VERSION
    # ---------------------------
    with tab9:
        section_title("EPDM Alkaline Aging Kinetics (AEM Electrolyzer)", "⏱️")
        
        st.markdown("""
        <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.95rem; color:#1a1a1a; margin-bottom: 1.5rem; line-height: 1.6;">
        Advanced thermo-chemical degradation analysis of peroxide-cured EPDM in KOH electrolyte. This module calculates multiple degradation indices, 
        performs Arrhenius kinetic modeling, tracks mechanistic pathways, and provides publication-ready statistical analysis.
        </div>
        """, unsafe_allow_html=True)

        if len(master) == 0:
            info_box("Upload spectra to begin aging analysis.", "warning")
        else:
            # ============================================================
            # SECTION 1: EXPERIMENTAL METADATA & INTEGRATION PARAMETERS
            # ============================================================
            st.markdown("<h3 style='font-family:Arial; font-size:1.3rem; font-weight:700; margin-top:1rem;'>⚙️ Experimental Configuration</h3>", unsafe_allow_html=True)
            
            col_meta, col_params = st.columns([1.2, 1], gap="large")
            
            with col_meta:
                st.markdown("<h4 style='font-family:Arial; font-size:1.05rem; font-weight:600;'>Aging Conditions</h4>", unsafe_allow_html=True)
                st.markdown("<p style='font-size:0.88rem; color:#475569;'>Assign precise aging conditions. Use replicates for statistical rigor.</p>", unsafe_allow_html=True)
                
                # Initialize metadata with replicate tracking
                if 'epdm_metadata' not in st.session_state or len(st.session_state['epdm_metadata']) != len(master):
                    meta_df = master[['File']].copy()
                    meta_df['Replicate_ID'] = ['A'] * len(meta_df)
                    meta_df['Aging_Days'] = 0.0
                    meta_df['Temp_C'] = 65
                    meta_df['KOH_Molar'] = 1.0
                    meta_df['Sample_Type'] = 'Aged'
                    st.session_state['epdm_metadata'] = meta_df
                
                edited_meta = st.data_editor(
                    st.session_state['epdm_metadata'],
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "File": st.column_config.TextColumn("Spectrum File", disabled=True),
                        "Replicate_ID": st.column_config.SelectboxColumn("Rep.", options=['A', 'B', 'C', 'D', 'E'], required=True),
                        "Aging_Days": st.column_config.NumberColumn("Days", min_value=0.0, format="%.2f"),
                        "Temp_C": st.column_config.SelectboxColumn("Temp (°C)", options=[23, 40, 50, 65, 80, 95], required=True),
                        "KOH_Molar": st.column_config.SelectboxColumn("KOH (M)", options=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0], required=True),
                        "Sample_Type": st.column_config.SelectboxColumn("Type", options=['Pristine', 'Aged'], required=True)
                    }
                )
                st.session_state['epdm_metadata'] = edited_meta

            with col_params:
                st.markdown("<h4 style='font-family:Arial; font-size:1.05rem; font-weight:600;'>Peak Assignment (cm⁻¹)</h4>", unsafe_allow_html=True)
                st.markdown("<p style='font-size:0.88rem; color:#475569;'>Define characteristic bands for degradation tracking.</p>", unsafe_allow_html=True)
                
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    ref_peak = st.number_input("Reference (CH₂)", value=1460, step=5, help="Backbone methylene (~1460 cm⁻¹)")
                    carb_peak = st.number_input("Carbonyl (C=O)", value=1715, step=5, help="Oxidation product (~1715 cm⁻¹)")
                    hydr_peak = st.number_input("Hydroxyl (O-H)", value=3400, step=5, help="Hydroxyl/water (~3400 cm⁻¹)")
                
                with col_p2:
                    ester_peak = st.number_input("Ester (C-O-C)", value=1240, step=5, help="Ester/ether (~1240 cm⁻¹)")
                    vinyl_peak = st.number_input("Vinyl (C=C)", value=1640, step=5, help="Residual unsaturation (~1640 cm⁻¹)")
                    ch3_peak = st.number_input("Methyl (CH₃)", value=1380, step=5, help="Chain integrity (~1380 cm⁻¹)")
                
                integration_width = st.slider("Integration Window (±cm⁻¹)", 5, 50, 15, 5, 
                                              help="Spectral range for peak area integration")

            # ============================================================
            # SECTION 2: ADVANCED SPECTRAL ANALYSIS
            # ============================================================
            st.markdown("<h3 style='font-family:Arial; font-size:1.3rem; font-weight:700; margin-top:2rem;'>📊 Degradation Index Calculation</h3>", unsafe_allow_html=True)
            
            analysis_mode = st.radio(
                "Analysis Method:",
                ["Peak Height (Fast)", "Peak Area Integration (Accurate)", "Deconvoluted Peak Area (Advanced)"],
                horizontal=True,
                help="Peak height: quick screening. Area integration: quantitative. Deconvolution: overlapping bands."
            )
            
            # Calculate comprehensive degradation indices
            kinetics_data = []
            
            for idx, row in edited_meta.iterrows():
                file_name = row['File']
                if file_name in spectra:
                    df_spec = spectra[file_name]
                    wavenumbers = df_spec['Wavenumber'].values
                    absorbance = df_spec['Absorbance_Norm'].values
                    
                    def get_peak_value(target_wn, method='height'):
                        """Extract peak value using specified method"""
                        idx_center = np.argmin(np.abs(wavenumbers - target_wn))
                        
                        if method == 'height':
                            return max(absorbance[idx_center], 0.0001)
                        
                        elif method == 'area':
                            # Integrate over window
                            mask = np.abs(wavenumbers - target_wn) <= integration_width
                            if np.sum(mask) > 2:
                                wn_region = wavenumbers[mask]
                                abs_region = absorbance[mask]
                                # Trapezoidal integration
                                area = np.trapz(abs_region, wn_region)
                                return max(abs(area), 0.0001)
                            return max(absorbance[idx_center], 0.0001)
                        
                        elif method == 'deconvolution':
                            # Simple Gaussian fitting for overlapping peaks
                            mask = np.abs(wavenumbers - target_wn) <= integration_width * 2
                            if np.sum(mask) > 5:
                                wn_region = wavenumbers[mask]
                                abs_region = absorbance[mask]
                                
                                def gaussian(x, amp, center, sigma):
                                    return amp * np.exp(-(x - center)**2 / (2 * sigma**2))
                                
                                try:
                                    # Initial guess
                                    amp_init = np.max(abs_region)
                                    popt, _ = curve_fit(gaussian, wn_region, abs_region, 
                                                       p0=[amp_init, target_wn, 10],
                                                       bounds=([0, target_wn-20, 1], [amp_init*2, target_wn+20, 50]))
                                    # Integrate fitted Gaussian
                                    fitted_area = popt[0] * popt[2] * np.sqrt(2 * np.pi)
                                    return max(fitted_area, 0.0001)
                                except:
                                    pass
                            return max(absorbance[idx_center], 0.0001)
                    
                    # Determine method
                    if "Peak Height" in analysis_mode:
                        method = 'height'
                    elif "Peak Area Integration" in analysis_mode:
                        method = 'area'
                    else:
                        method = 'deconvolution'
                    
                    # Extract all peak values
                    val_ref = get_peak_value(ref_peak, method)
                    val_carb = get_peak_value(carb_peak, method)
                    val_hydr = get_peak_value(hydr_peak, method)
                    val_ester = get_peak_value(ester_peak, method)
                    val_vinyl = get_peak_value(vinyl_peak, method)
                    val_ch3 = get_peak_value(ch3_peak, method)
                    
                    # Calculate normalized indices
                    kinetics_data.append({
                        "File": file_name,
                        "Replicate": row['Replicate_ID'],
                        "Days": row['Aging_Days'],
                        "Temp_C": row['Temp_C'],
                        "KOH_M": row['KOH_Molar'],
                        "Sample_Type": row['Sample_Type'],
                        "Condition": f"{row['Temp_C']}°C, {row['KOH_Molar']}M KOH",
                        "Carbonyl_Index": val_carb / val_ref,
                        "Hydroxyl_Index": val_hydr / val_ref,
                        "Ester_Index": val_ester / val_ref,
                        "Vinyl_Index": val_vinyl / val_ref,
                        "Chain_Integrity": val_ch3 / val_ref,
                        "Overall_Degradation": (val_carb + val_hydr) / (2 * val_ref)
                    })
            
            k_df = pd.DataFrame(kinetics_data)
            
            if k_df.empty:
                st.warning("No spectral data available for analysis.")
            else:
                # ============================================================
                # SECTION 3: STATISTICAL AGGREGATION
                # ============================================================
                st.markdown("<h4 style='font-family:Arial; font-size:1.05rem; font-weight:600; margin-top:1.5rem;'>Statistical Summary</h4>", unsafe_allow_html=True)
                
                # Group by condition and calculate statistics
                index_cols = ['Carbonyl_Index', 'Hydroxyl_Index', 'Ester_Index', 'Vinyl_Index', 'Chain_Integrity', 'Overall_Degradation']
                
                stats_data = []
                for (days, temp, koh), group in k_df.groupby(['Days', 'Temp_C', 'KOH_M']):
                    if len(group) > 0:
                        stats_row = {
                            'Days': days,
                            'Temp_C': temp,
                            'KOH_M': koh,
                            'Condition': f"{temp}°C, {koh}M KOH",
                            'N_Replicates': len(group)
                        }
                        for col in index_cols:
                            stats_row[f'{col}_Mean'] = group[col].mean()
                            stats_row[f'{col}_Std'] = group[col].std() if len(group) > 1 else 0
                            stats_row[f'{col}_SE'] = stats_row[f'{col}_Std'] / np.sqrt(len(group)) if len(group) > 1 else 0
                        stats_data.append(stats_row)
                
                stats_df = pd.DataFrame(stats_data)
                
                # ============================================================
                # SECTION 4: VISUALIZATION OPTIONS
                # ============================================================
                st.markdown("<h3 style='font-family:Arial; font-size:1.3rem; font-weight:700; margin-top:2rem;'>📈 Kinetic Analysis & Visualization</h3>", unsafe_allow_html=True)
                
                viz_tabs = st.tabs(["🔍 Degradation Trends", "🌡️ Arrhenius Analysis", "🗺️ Multi-Variable Mapping", "📊 Mechanism Correlation"])
                
                # --------------------------------------------------------
                # TAB 1: DEGRADATION TRENDS
                # --------------------------------------------------------
                with viz_tabs[0]:
                    col_v1, col_v2 = st.columns([1, 2.5])
                    
                    with col_v1:
                        target_metric = st.selectbox(
                            "Degradation Index:",
                            ["Carbonyl_Index", "Hydroxyl_Index", "Ester_Index", "Vinyl_Index", "Chain_Integrity", "Overall_Degradation"],
                            format_func=lambda x: x.replace('_', ' ')
                        )
                        
                        show_individual = st.checkbox("Show Individual Replicates", value=True)
                        show_error_bars = st.checkbox("Show Error Bars (±SE)", value=True)
                        fit_kinetics = st.checkbox("Fit Kinetic Model", value=False)
                        
                        if fit_kinetics:
                            kinetic_model = st.selectbox(
                                "Model Type:",
                                ["Zero Order (Linear)", "First Order (Exponential)", "Second Order", "Power Law"]
                            )
                    
                    with col_v2:
                        # Create scatter plot with error bars
                        fig_trend = go.Figure()
                        
                        # Get unique conditions
                        conditions = stats_df['Condition'].unique() if not stats_df.empty else []
                        colors = px.colors.qualitative.Set2
                        
                        for i, cond in enumerate(conditions):
                            cond_data = stats_df[stats_df['Condition'] == cond].sort_values('Days')
                            color = colors[i % len(colors)]
                            
                            # Mean line with error bars
                            if show_error_bars and not cond_data.empty:
                                fig_trend.add_trace(go.Scatter(
                                    x=cond_data['Days'],
                                    y=cond_data[f'{target_metric}_Mean'],
                                    error_y=dict(
                                        type='data',
                                        array=cond_data[f'{target_metric}_SE'],
                                        visible=True,
                                        thickness=1.5,
                                        width=4
                                    ),
                                    mode='markers+lines',
                                    name=cond,
                                    marker=dict(size=10, color=color, line=dict(width=1.5, color=BLACK)),
                                    line=dict(width=2, color=color)
                                ))
                            
                            # Individual replicates
                            if show_individual:
                                ind_data = k_df[k_df['Condition'] == cond]
                                fig_trend.add_trace(go.Scatter(
                                    x=ind_data['Days'],
                                    y=ind_data[target_metric],
                                    mode='markers',
                                    name=f"{cond} (raw)",
                                    marker=dict(size=6, color=color, opacity=0.4, symbol='circle'),
                                    showlegend=False,
                                    hovertemplate='%{text}<br>Day: %{x}<br>Value: %{y:.4f}<extra></extra>',
                                    text=ind_data['File']
                                ))
                            
                            # Fit kinetic model if requested
                            if fit_kinetics and len(cond_data) > 2:
                                x_data = cond_data['Days'].values
                                y_data = cond_data[f'{target_metric}_Mean'].values
                                
                                # Define models
                                def zero_order(t, k, y0):
                                    return y0 + k * t
                                
                                def first_order(t, k, y0):
                                    return y0 * np.exp(k * t)
                                
                                def second_order(t, k, y0):
                                    return 1 / (1/y0 - k * t) if y0 != 0 else y0
                                
                                def power_law(t, k, y0, n):
                                    return y0 + k * t**n
                                
                                try:
                                    if "Zero Order" in kinetic_model:
                                        popt, _ = curve_fit(zero_order, x_data, y_data, p0=[0.001, y_data[0]])
                                        x_fit = np.linspace(0, x_data.max() * 1.1, 100)
                                        y_fit = zero_order(x_fit, *popt)
                                        model_label = f"k={popt[0]:.2e} day⁻¹"
                                    
                                    elif "First Order" in kinetic_model:
                                        popt, _ = curve_fit(first_order, x_data, y_data, p0=[0.001, y_data[0]])
                                        x_fit = np.linspace(0, x_data.max() * 1.1, 100)
                                        y_fit = first_order(x_fit, *popt)
                                        model_label = f"k={popt[0]:.2e} day⁻¹"
                                    
                                    elif "Power Law" in kinetic_model:
                                        popt, _ = curve_fit(power_law, x_data, y_data, p0=[0.001, y_data[0], 1.0])
                                        x_fit = np.linspace(0, x_data.max() * 1.1, 100)
                                        y_fit = power_law(x_fit, *popt)
                                        model_label = f"k={popt[0]:.2e}, n={popt[2]:.2f}"
                                    
                                    fig_trend.add_trace(go.Scatter(
                                        x=x_fit, y=y_fit,
                                        mode='lines',
                                        name=f"{cond} fit ({model_label})",
                                        line=dict(dash='dash', width=2, color=color)
                                    ))
                                except Exception as e:
                                    st.warning(f"Could not fit {kinetic_model} for {cond}: {str(e)}")
                        
                        fig_trend.update_layout(
                            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
                            title=dict(text=f"<b>{target_metric.replace('_', ' ')} vs. Aging Time</b>", 
                                      font=dict(family="Arial", size=16, color=BLACK)),
                            xaxis=dict(title="<b>Aging Time (Days)</b>", **FTIR_STYLE, showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
                            yaxis=dict(title=f"<b>{target_metric.replace('_', ' ')}</b>", **FTIR_STYLE, showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
                            height=550,
                            margin=dict(l=70, r=40, t=70, b=70),
                            legend=dict(
                                bgcolor=WHITE, bordercolor=BLACK, borderwidth=1,
                                font=dict(family="Arial", size=11, color=BLACK),
                                x=1.02, y=1, xanchor='left'
                            ),
                            hovermode='closest'
                        )
                        st.plotly_chart(fig_trend, use_container_width=True, config=JOURNAL_CONFIG)
                
                # --------------------------------------------------------
                # TAB 2: ARRHENIUS ANALYSIS
                # --------------------------------------------------------
                with viz_tabs[1]:
                    st.markdown("""
                    <p style='font-size:0.9rem; color:#475569; margin-bottom:1rem;'>
                    Arrhenius analysis determines the activation energy (Eₐ) for degradation processes. 
                    The rate constant k follows: k = A·exp(-Eₐ/RT), where ln(k) vs. 1/T yields Eₐ from the slope.
                    </p>
                    """, unsafe_allow_html=True)
                    
                    col_a1, col_a2 = st.columns([1, 2])
                    
                    with col_a1:
                        arrhenius_index = st.selectbox(
                            "Index for Arrhenius:",
                            ["Carbonyl_Index", "Hydroxyl_Index", "Overall_Degradation"],
                            format_func=lambda x: x.replace('_', ' '),
                            key='arrhenius_idx'
                        )
                        
                        arrhenius_koh = st.selectbox(
                            "KOH Concentration (M):",
                            sorted(k_df['KOH_M'].unique()),
                            key='arrhenius_koh'
                        )
                        
                        time_point = st.number_input(
                            "Reference Time Point (Days):",
                            min_value=0.0,
                            value=float(k_df['Days'].max()) if len(k_df) > 0 else 7.0,
                            help="Compare degradation across temperatures at this specific aging time"
                        )
                    
                    with col_a2:
                        # Filter data
                        arr_data = stats_df[
                            (stats_df['KOH_M'] == arrhenius_koh) & 
                            (stats_df['Days'] == time_point)
                        ].copy()
                        
                        if len(arr_data) > 1:
                            # Calculate 1/T and ln(Index)
                            arr_data['Temp_K'] = arr_data['Temp_C'] + 273.15
                            arr_data['InvT'] = 1000 / arr_data['Temp_K']  # 1000/T for better scaling
                            arr_data['ln_Index'] = np.log(arr_data[f'{arrhenius_index}_Mean'])
                            
                            # Linear regression
                            slope, intercept, r_value, p_value, std_err = linregress(
                                arr_data['InvT'], arr_data['ln_Index']
                            )
                            
                            # Calculate activation energy
                            R = 8.314  # J/(mol·K)
                            Ea_kJ = -slope * R  # kJ/mol
                            
                            # Create Arrhenius plot
                            fig_arr = go.Figure()
                            
                            # Data points
                            fig_arr.add_trace(go.Scatter(
                                x=arr_data['InvT'],
                                y=arr_data['ln_Index'],
                                mode='markers',
                                name='Experimental',
                                marker=dict(size=12, color='#E74C3C', line=dict(width=2, color=BLACK)),
                                error_y=dict(
                                    type='data',
                                    array=arr_data[f'{arrhenius_index}_SE'] / arr_data[f'{arrhenius_index}_Mean'],
                                    visible=True
                                ) if f'{arrhenius_index}_SE' in arr_data.columns else None
                            ))
                            
                            # Fit line
                            x_fit = np.linspace(arr_data['InvT'].min() * 0.95, arr_data['InvT'].max() * 1.05, 100)
                            y_fit = slope * x_fit + intercept
                            
                            fig_arr.add_trace(go.Scatter(
                                x=x_fit, y=y_fit,
                                mode='lines',
                                name=f'Linear Fit (R²={r_value**2:.4f})',
                                line=dict(dash='dash', width=2, color='#3498DB')
                            ))
                            
                            fig_arr.update_layout(
                                plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
                                title=dict(
                                    text=f"<b>Arrhenius Plot: {arrhenius_index.replace('_', ' ')}</b><br>" + 
                                         f"<sub>Eₐ = {Ea_kJ:.1f} kJ/mol | {arrhenius_koh}M KOH | Day {time_point}</sub>",
                                    font=dict(family="Arial", size=15, color=BLACK)
                                ),
                                xaxis=dict(title="<b>1000/T (K⁻¹)</b>", **FTIR_STYLE),
                                yaxis=dict(title=f"<b>ln({arrhenius_index.replace('_', ' ')})</b>", **FTIR_STYLE),
                                height=500,
                                margin=dict(l=70, r=40, t=80, b=70),
                                annotations=[
                                    dict(
                                        text=f"Eₐ = {Ea_kJ:.1f} ± {std_err*R:.1f} kJ/mol<br>R² = {r_value**2:.4f}<br>p = {p_value:.3e}",
                                        xref="paper", yref="paper",
                                        x=0.05, y=0.95,
                                        showarrow=False,
                                        bgcolor="rgba(255,255,255,0.9)",
                                        bordercolor=BLACK,
                                        borderwidth=1,
                                        font=dict(size=11, family="Arial")
                                    )
                                ]
                            )
                            st.plotly_chart(fig_arr, use_container_width=True, config=JOURNAL_CONFIG)
                            
                            # Lifetime prediction
                            st.markdown("<h4 style='font-size:1rem; font-weight:600; margin-top:1rem;'>Lifetime Prediction</h4>", unsafe_allow_html=True)
                            pred_temp = st.slider("Operating Temperature (°C):", 20, 80, 50)
                            failure_criterion = st.number_input("Failure Criterion (Index Value):", 
                                                               min_value=0.1, value=2.0, step=0.1)
                            
                            # Predict time to failure
                            T_op = pred_temp + 273.15
                            k_op = np.exp(intercept) * np.exp(-Ea_kJ * 1000 / (R * T_op))
                            
                            # Assuming first-order: Index = Index0 * exp(k*t)
                            if arrhenius_index in arr_data.columns:
                                Index0 = arr_data[f'{arrhenius_index}_Mean'].min()
                                if k_op > 0 and failure_criterion > Index0:
                                    t_failure = np.log(failure_criterion / Index0) / k_op
                                    st.success(f"**Predicted time to failure at {pred_temp}°C: {t_failure:.1f} days ({t_failure/365:.2f} years)**")
                                else:
                                    st.warning("Cannot predict failure time with current model parameters.")
                        else:
                            st.warning(f"Need at least 2 different temperatures at day {time_point} with {arrhenius_koh}M KOH for Arrhenius analysis.")
                
                # --------------------------------------------------------
                # TAB 3: MULTI-VARIABLE MAPPING
                # --------------------------------------------------------
                with viz_tabs[2]:
                    st.markdown("<p style='font-size:0.9rem; color:#475569;'>Visualize degradation as a function of time, temperature, and KOH concentration.</p>", unsafe_allow_html=True)
                    
                    map_index = st.selectbox(
                        "Degradation Index:",
                        ["Carbonyl_Index", "Hydroxyl_Index", "Overall_Degradation"],
                        format_func=lambda x: x.replace('_', ' '),
                        key='map_idx'
                    )
                    
                    plot_type = st.radio("Plot Type:", ["2D Contour (Temp vs Time)", "3D Surface", "Heatmap (Temp vs KOH)"], horizontal=True)
                    
                    if plot_type == "2D Contour (Temp vs Time)":
                        selected_koh = st.selectbox("KOH Concentration (M):", sorted(stats_df['KOH_M'].unique()), key='contour_koh')
                        
                        contour_data = stats_df[stats_df['KOH_M'] == selected_koh].copy()
                        
                        if len(contour_data) > 3:
                            # Create pivot table
                            pivot = contour_data.pivot_table(
                                values=f'{map_index}_Mean',
                                index='Temp_C',
                                columns='Days',
                                aggfunc='mean'
                            )
                            
                            fig_contour = go.Figure(data=go.Contour(
                                z=pivot.values,
                                x=pivot.columns,
                                y=pivot.index,
                                colorscale='RdYlBu_r',
                                colorbar=dict(title=map_index.replace('_', ' ')),
                                contours=dict(showlabels=True, labelfont=dict(size=10, color='white'))
                            ))
                            
                            fig_contour.update_layout(
                                plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
                                title=f"<b>{map_index.replace('_', ' ')} - {selected_koh}M KOH</b>",
                                xaxis=dict(title="<b>Aging Time (Days)</b>", **FTIR_STYLE),
                                yaxis=dict(title="<b>Temperature (°C)</b>", **FTIR_STYLE),
                                height=500
                            )
                            st.plotly_chart(fig_contour, use_container_width=True, config=JOURNAL_CONFIG)
                        else:
                            st.warning("Need more data points for contour plot.")
                    
                    elif plot_type == "3D Surface":
                        selected_koh_3d = st.selectbox("KOH Concentration (M):", sorted(stats_df['KOH_M'].unique()), key='3d_koh')
                        
                        surface_data = stats_df[stats_df['KOH_M'] == selected_koh_3d].copy()
                        
                        if len(surface_data) > 3:
                            pivot = surface_data.pivot_table(
                                values=f'{map_index}_Mean',
                                index='Temp_C',
                                columns='Days',
                                aggfunc='mean'
                            )
                            
                            fig_3d = go.Figure(data=[go.Surface(
                                z=pivot.values,
                                x=pivot.columns,
                                y=pivot.index,
                                colorscale='Viridis',
                                colorbar=dict(title=map_index.replace('_', ' '))
                            )])
                            
                            fig_3d.update_layout(
                                title=f"<b>{map_index.replace('_', ' ')} Surface</b>",
                                scene=dict(
                                    xaxis=dict(title='Days', backgroundcolor=PLOT_BG),
                                    yaxis=dict(title='Temp (°C)', backgroundcolor=PLOT_BG),
                                    zaxis=dict(title=map_index.replace('_', ' '), backgroundcolor=PLOT_BG)
                                ),
                                height=600
                            )
                            st.plotly_chart(fig_3d, use_container_width=True, config=JOURNAL_CONFIG)
                        else:
                            st.warning("Need more data points for 3D surface.")
                    
                    else:  # Heatmap
                        selected_days = st.selectbox("Time Point (Days):", sorted(stats_df['Days'].unique()), key='heatmap_days')
                        
                        heatmap_data = stats_df[stats_df['Days'] == selected_days].copy()
                        
                        if len(heatmap_data) > 0:
                            pivot = heatmap_data.pivot_table(
                                values=f'{map_index}_Mean',
                                index='Temp_C',
                                columns='KOH_M',
                                aggfunc='mean'
                            )
                            
                            fig_heat = go.Figure(data=go.Heatmap(
                                z=pivot.values,
                                x=pivot.columns,
                                y=pivot.index,
                                colorscale='YlOrRd',
                                colorbar=dict(title=map_index.replace('_', ' ')),
                                text=np.round(pivot.values, 3),
                                texttemplate='%{text}',
                                textfont=dict(size=10)
                            ))
                            
                            fig_heat.update_layout(
                                plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
                                title=f"<b>{map_index.replace('_', ' ')} at Day {selected_days}</b>",
                                xaxis=dict(title="<b>KOH Concentration (M)</b>", **FTIR_STYLE),
                                yaxis=dict(title="<b>Temperature (°C)</b>", **FTIR_STYLE),
                                height=500
                            )
                            st.plotly_chart(fig_heat, use_container_width=True, config=JOURNAL_CONFIG)
                        else:
                            st.warning("No data available for selected time point.")
                
                # --------------------------------------------------------
                # TAB 4: MECHANISM CORRELATION
                # --------------------------------------------------------
                with viz_tabs[3]:
                    st.markdown("<p style='font-size:0.9rem; color:#475569;'>Analyze correlations between different degradation mechanisms and pathways.</p>", unsafe_allow_html=True)
                    
                    # Correlation matrix
                    corr_indices = ['Carbonyl_Index', 'Hydroxyl_Index', 'Ester_Index', 'Vinyl_Index', 'Chain_Integrity']
                    available_indices = [idx for idx in corr_indices if idx in k_df.columns]
                    
                    if len(available_indices) > 1:
                        corr_matrix = k_df[available_indices].corr()
                        
                        fig_corr = go.Figure(data=go.Heatmap(
                            z=corr_matrix.values,
                            x=[idx.replace('_', ' ') for idx in corr_matrix.columns],
                            y=[idx.replace('_', ' ') for idx in corr_matrix.index],
                            colorscale='RdBu',
                            zmid=0,
                            colorbar=dict(title='Correlation'),
                            text=np.round(corr_matrix.values, 2),
                            texttemplate='%{text}',
                            textfont=dict(size=11, color='black')
                        ))
                        
                        fig_corr.update_layout(
                            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
                            title="<b>Degradation Mechanism Correlation Matrix</b>",
                            height=500,
                            margin=dict(l=150, r=40, t=80, b=150)
                        )
                        st.plotly_chart(fig_corr, use_container_width=True, config=JOURNAL_CONFIG)
                        
                        # Scatter plot matrix
                        st.markdown("<h4 style='font-size:1rem; font-weight:600; margin-top:1.5rem;'>Pairwise Relationships</h4>", unsafe_allow_html=True)
                        
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            x_mech = st.selectbox("X-axis:", available_indices, index=0, format_func=lambda x: x.replace('_', ' '))
                        with col_m2:
                            y_mech = st.selectbox("Y-axis:", available_indices, index=min(1, len(available_indices)-1), format_func=lambda x: x.replace('_', ' '))
                        
                        if x_mech != y_mech:
                            fig_scatter = px.scatter(
                                k_df,
                                x=x_mech,
                                y=y_mech,
                                color='Condition',
                                size='Days',
                                hover_data=['File', 'Days', 'Temp_C', 'KOH_M'],
                                trendline='ols'
                            )
                            
                            fig_scatter.update_traces(marker=dict(line=dict(width=1, color=BLACK)))
                            fig_scatter.update_layout(
                                plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
                                title=f"<b>{x_mech.replace('_', ' ')} vs {y_mech.replace('_', ' ')}</b>",
                                xaxis=dict(title=f"<b>{x_mech.replace('_', ' ')}</b>", **FTIR_STYLE),
                                yaxis=dict(title=f"<b>{y_mech.replace('_', ' ')}</b>", **FTIR_STYLE),
                                height=500
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True, config=JOURNAL_CONFIG)
                    
                    # Degradation pathway analysis
                    st.markdown("<h4 style='font-size:1rem; font-weight:600; margin-top:1.5rem;'>Degradation Pathway Contribution</h4>", unsafe_allow_html=True)
                    
                    # Normalize all indices to relative contribution
                    pathway_cols = ['Carbonyl_Index', 'Hydroxyl_Index', 'Ester_Index']
                    available_pathways = [col for col in pathway_cols if col in k_df.columns]
                    
                    if len(available_pathways) > 1:
                        latest_time = k_df['Days'].max()
                        pathway_data = k_df[k_df['Days'] == latest_time].copy()
                        
                        if len(pathway_data) > 0:
                            # Calculate relative contributions
                            pathway_data['Total_Oxidation'] = pathway_data[available_pathways].sum(axis=1)
                            for col in available_pathways:
                                pathway_data[f'{col}_Fraction'] = pathway_data[col] / pathway_data['Total_Oxidation']
                            
                            # Stacked bar chart
                            fig_pathway = go.Figure()
                            
                            for col in available_pathways:
                                fig_pathway.add_trace(go.Bar(
                                    name=col.replace('_Index', '').replace('_', ' '),
                                    x=pathway_data['Condition'],
                                    y=pathway_data[f'{col}_Fraction'] * 100,
                                    text=np.round(pathway_data[f'{col}_Fraction'] * 100, 1),
                                    texttemplate='%{text}%',
                                    textposition='inside'
                                ))
                            
                            fig_pathway.update_layout(
                                plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
                                title=f"<b>Degradation Pathway Distribution (Day {latest_time})</b>",
                                xaxis=dict(title="<b>Condition</b>", **FTIR_STYLE),
                                yaxis=dict(title="<b>Contribution (%)</b>", **FTIR_STYLE),
                                barmode='stack',
                                height=450,
                                legend=dict(bgcolor=WHITE, bordercolor=BLACK, borderwidth=1)
                            )
                            st.plotly_chart(fig_pathway, use_container_width=True, config=JOURNAL_CONFIG)
                
                # ============================================================
                # SECTION 5: DATA EXPORT
                # ============================================================
                st.markdown("<h3 style='font-family:Arial; font-size:1.3rem; font-weight:700; margin-top:2rem;'>💾 Export Analysis Results</h3>", unsafe_allow_html=True)
                
                col_e1, col_e2, col_e3 = st.columns(3)
                
                with col_e1:
                    # Raw kinetics data
                    csv_raw = k_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Raw Kinetics Data",
                        data=csv_raw,
                        file_name="EPDM_Kinetics_Raw.csv",
                        mime="text/csv",
                    )
                
                with col_e2:
                    # Statistical summary
                    if not stats_df.empty:
                        csv_stats = stats_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📊 Statistical Summary",
                            data=csv_stats,
                            file_name="EPDM_Kinetics_Statistics.csv",
                            mime="text/csv",
                        )
                
                with col_e3:
                    # Generate comprehensive report
                    if st.button("📄 Generate Full Report"):
                        report_lines = [
                            "# EPDM-KOH AGING KINETICS REPORT",
                            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            "",
                            "## EXPERIMENTAL CONDITIONS",
                            f"Total Spectra: {len(k_df)}",
                            f"Time Range: {k_df['Days'].min():.1f} - {k_df['Days'].max():.1f} days",
                            f"Temperature Range: {k_df['Temp_C'].min()} - {k_df['Temp_C'].max()} °C",
                            f"KOH Range: {k_df['KOH_M'].min()} - {k_df['KOH_M'].max()} M",
                            "",
                            "## DEGRADATION INDICES (Latest Time Point)",
                        ]
                        
                        latest = k_df[k_df['Days'] == k_df['Days'].max()]
                        for idx in ['Carbonyl_Index', 'Hydroxyl_Index', 'Overall_Degradation']:
                            if idx in latest.columns:
                                report_lines.append(f"{idx}: {latest[idx].mean():.4f} ± {latest[idx].std():.4f}")
                        
                        report_lines.extend([
                            "",
                            "## STATISTICAL SUMMARY",
                            stats_df.to_string(),
                            "",
                            "## RAW DATA",
                            k_df.to_string()
                        ])
                        
                        report_text = "\n".join(report_lines)
                        st.download_button(
                            label="💾 Download Report (TXT)",
                            data=report_text.encode('utf-8'),
                            file_name="EPDM_Aging_Report.txt",
                            mime="text/plain"
                        )
else:
    # --- Empty State UI ---
    st.markdown("""
    <div style="
        margin-top:3rem; padding:3rem 2rem; background:#ffffff;
        border:1px solid #e2e8f0; box-shadow: 0 4px 15px rgba(0,0,0,0.03);
        border-radius:8px; text-align:center;
    ">
        <div style="font-size:3rem;margin-bottom:1rem;">📉</div>
        <div style="
            font-family:'Playfair Display',Georgia,serif;
            font-size:1.5rem;color:#000000; margin-bottom:0.5rem; font-weight:700;
        ">Ready for Spectral Analysis</div>
        <div style="
            font-family:'IBM Plex Sans',sans-serif;
            font-size:0.85rem;color:#000000;
            max-width:480px;margin:0 auto;line-height:1.7;
        ">
            Upload your raw FTIR data files via the <b style="color:#c9a84c;">Data Input</b> panel
            in the sidebar. Supports automatic baseline correction (ALS), Savitzky-Golay smoothing, 
            2nd Derivative resolution enhancement, and PCA Clustering.
        </div>
    </div>
    """, unsafe_allow_html=True)
