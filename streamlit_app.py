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

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📉 Primary Spectra", 
        "🔬 2nd Derivative", 
        "📋 Peak Assignments", 
        "📈 Deconvolution", 
        "🧠 PCA Clustering", 
        "🧪 Spectral Match", 
        "📊 Data Matrix Export"
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
        section_title("Principal Component Analysis (PCA)", "🧠")
        
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
        section_title("Data Matrix Export", "📊")
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
