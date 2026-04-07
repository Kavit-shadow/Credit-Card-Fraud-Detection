import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import uuid
import numpy as np
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def chart_key():
    return str(uuid.uuid4())

MODEL_FEATURES = [f'V{i}' for i in range(1, 29)] + ['Time', 'scaled_amount']
STATUS_MAP     = {0: "Legitimate", 1: "Fraudulent"}
STATUS_COLOR   = {"Legitimate": "#00b894", "Fraudulent": "#d63031"}

st.set_page_config(page_title="FraudShield AI", layout="wide", page_icon="🛡️")

for k, v in [("bulk_df", None), ("uploaded_df", None), ("page", "bulk"), ("theme", "dark")]:
    if k not in st.session_state:
        st.session_state[k] = v

DARK  = {"bg":"#0d1b2a","bg2":"#1b2838","card":"rgba(255,255,255,0.07)",
         "text":"#f0f4f8","subtext":"#a0b8d0","border":"rgba(255,255,255,0.12)",
         "accent":"#e94560","accent2":"#7ec8e3","plot_bg":"rgba(13,27,42,0.97)",
         "input":"#1e2d40","sidebar":"#0f1e2e"}
LIGHT = {"bg":"#f0f4f8","bg2":"#e2eaf3","card":"rgba(0,0,0,0.04)",
         "text":"#1a2332","subtext":"#445566","border":"rgba(0,0,0,0.10)",
         "accent":"#e94560","accent2":"#1565c0","plot_bg":"rgba(240,244,248,0.97)",
         "input":"#ffffff","sidebar":"#dde6f0"}

def T():
    return DARK if st.session_state.theme == "dark" else LIGHT

def style_fig(fig, height=420):
    c = T()
    fig.update_layout(
        height=height,
        font=dict(family="Poppins, sans-serif", size=14, color=c["text"]),
        title_font=dict(family="Poppins, sans-serif", size=17, color=c["accent"]),
        paper_bgcolor=c["plot_bg"], plot_bgcolor=c["plot_bg"],
        legend=dict(font=dict(size=13, color=c["text"]), bgcolor="rgba(0,0,0,0.2)"),
        margin=dict(l=20, r=20, t=55, b=20),
    )
    fig.update_xaxes(tickfont=dict(size=13, color=c["subtext"], family="Poppins"),
                     title_font=dict(size=14, color=c["accent2"], family="Poppins"),
                     gridcolor="rgba(128,128,128,0.12)")
    fig.update_yaxes(tickfont=dict(size=13, color=c["subtext"], family="Poppins"),
                     title_font=dict(size=14, color=c["accent2"], family="Poppins"),
                     gridcolor="rgba(128,128,128,0.12)")
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# CSS  — KEY FIX: do NOT use blanket  div/span/p overrides.
#        Target only Streamlit-specific selectors so HTML in markdown renders.
# ─────────────────────────────────────────────────────────────────────────────
def inject_css():
    c = T()
    dark = st.session_state.theme == "dark"
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

/* ── APP BACKGROUND ── */
html, body, .stApp {{
    font-family: 'Poppins', sans-serif !important;
    background: linear-gradient(135deg,{c['bg']},{c['bg2']},{c['bg']}) !important;
}}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {{
    background: {c['sidebar']} !important;
    border-right: 1px solid {c['border']} !important;
}}
/* Fix the sidebar collapse arrow — hide broken text, show SVG */
[data-testid="collapsedControl"] {{
    background: {c['accent']} !important;
    border-radius: 0 8px 8px 0 !important;
}}
[data-testid="collapsedControl"] span {{
    display: none !important;
}}
[data-testid="collapsedControl"] svg {{
    display: block !important;
    fill: #ffffff !important;
}}

/* ── STREAMLIT MARKDOWN TEXT ── */
[data-testid="stMarkdownContainer"] p {{
    font-family: 'Poppins', sans-serif !important;
    color: {c['text']} !important;
    font-size: 15px !important;
}}
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {{
    font-family: 'Poppins', sans-serif !important;
}}

/* ── WIDGET LABELS ── */
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] p {{
    font-family: 'Poppins', sans-serif !important;
    font-size: 14px !important; font-weight: 700 !important;
    color: {c['accent2']} !important;
}}

/* ── HEADINGS ── */
h1 {{ font-size:2.3rem !important; font-weight:800 !important;
      color:{c['accent']} !important; font-family:'Poppins',sans-serif !important; }}
h2 {{ font-size:1.7rem !important; font-weight:700 !important;
      color:{c['accent']} !important; font-family:'Poppins',sans-serif !important; }}
h3 {{ font-size:1.35rem !important; font-weight:700 !important;
      color:{c['accent']} !important; font-family:'Poppins',sans-serif !important; }}
h4 {{ font-size:1.05rem !important; font-weight:700 !important;
      color:{c['accent2']} !important; font-family:'Poppins',sans-serif !important; }}

/* ── BUTTONS ── */
.stButton > button {{
    background: linear-gradient(135deg,#e94560,#c0392b) !important;
    color: #ffffff !important; font-size:15px !important; font-weight:700 !important;
    border-radius: 10px !important; border: none !important; padding: 11px 28px !important;
    font-family: 'Poppins',sans-serif !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 14px rgba(233,69,96,0.35) !important;
}}
.stButton > button:hover {{
    background: linear-gradient(135deg,#c0392b,#96281b) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(233,69,96,0.55) !important;
}}
.stButton > button p,
.stButton > button span {{
    color: #ffffff !important; font-size:15px !important; font-weight:700 !important;
}}

/* ── DOWNLOAD BUTTON ── */
.stDownloadButton > button {{
    background: linear-gradient(135deg,#00b894,#00896e) !important;
    color: #ffffff !important; font-size:14px !important; font-weight:700 !important;
    border-radius: 10px !important; border: none !important;
}}
.stDownloadButton > button p,
.stDownloadButton > button span {{ color: #ffffff !important; }}

/* ── INPUTS / SELECT ── */
div[data-baseweb="input"] input,
input[type="text"], textarea {{
    background-color: {c['input']} !important;
    color: {c['text']} !important;
    border-radius: 8px !important; font-weight:600 !important;
}}
div[data-baseweb="select"] > div {{
    background-color: {c['input']} !important; color:{c['text']} !important;
    border-radius: 8px !important; border: 1.5px solid {c['accent']} !important;
}}
div[data-baseweb="select"] span {{ color:{c['text']} !important; font-weight:600 !important; }}
div[data-baseweb="select"] svg  {{ fill:{c['text']} !important; }}
ul[role="listbox"]    {{ background-color:{c['input']} !important; }}
ul[role="listbox"] li {{ background-color:{c['input']} !important; color:{c['text']} !important; }}
ul[role="listbox"] li:hover {{ background-color:{c['accent']} !important; color:#fff !important; }}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {{
    background:{c['card']} !important;
    border: 2px dashed {c['accent']} !important;
    border-radius: 12px !important;
}}
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploaderDropzoneInstructions"] p,
[data-testid="stFileUploaderDropzoneInstructions"] span {{
    color:{c['text']} !important;
}}
[data-testid="stFileUploader"] button {{
    background: linear-gradient(135deg,#e94560,#c0392b) !important;
    color: #ffffff !important; border-radius:8px !important;
    border: none !important; font-weight:700 !important;
}}

/* ── KPI CARD ── */
.kpi-card {{
    background:{c['card']}; padding:22px 16px; border-radius:16px;
    text-align:center; box-shadow:0 6px 24px rgba(0,0,0,0.25);
    border:1px solid {c['border']}; border-left:5px solid {c['accent']};
    backdrop-filter:blur(8px); transition:transform 0.2s ease; margin-bottom:4px;
}}
.kpi-card:hover {{ transform:translateY(-3px); }}
.kpi-label {{
    font-family:'Poppins',sans-serif !important;
    color:{c['accent2']} !important; font-size:0.68rem !important;
    font-weight:700 !important; text-transform:uppercase !important;
    letter-spacing:0.09em !important; display:block; margin-bottom:8px;
}}
.kpi-value {{
    font-family:'Poppins',sans-serif !important;
    color:{c['text']} !important; font-size:2rem !important;
    font-weight:800 !important; display:block;
}}

/* ── INSIGHT BOX ── */
.insight-box {{
    background:{c['card']}; padding:13px 18px; border-radius:12px;
    box-shadow:0 3px 10px rgba(0,0,0,0.2); margin-bottom:10px;
    border-left:4px solid {c['accent']}; font-size:14px;
    font-weight:600; color:{c['text']}; backdrop-filter:blur(6px);
}}
.insight-box b {{ color:{c['accent']}; }}

/* ── SECTION HEADER ── */
.section-header {{
    background:{c['card']}; border-radius:14px; padding:16px 22px;
    margin-bottom:18px; border:1px solid {c['border']};
    backdrop-filter:blur(8px);
}}

/* ── NAV ── */
.nav-active  > div > button {{
    background: linear-gradient(135deg,#e94560,#c0392b) !important;
    box-shadow: 0 4px 16px rgba(233,69,96,0.5) !important;
    color: #ffffff !important;
}}
.nav-inactive > div > button {{
    background: {c['card']} !important;
    border: 1px solid {c['border']} !important;
    color: {c['text']} !important; box-shadow:none !important;
}}
.nav-inactive > div > button:hover {{
    background: rgba(233,69,96,0.15) !important;
    border-color: {c['accent']} !important;
}}

/* ── THEME BTN ── */
.theme-btn > div > button {{
    background: {'rgba(255,255,255,0.08)' if dark else 'rgba(0,0,0,0.06)'} !important;
    border: 1px solid {c['border']} !important;
    color: {c['text']} !important;
    font-size:13px !important; padding:6px 14px !important;
    border-radius:20px !important; box-shadow:none !important;
}}

/* ── ALERTS ── */
[data-testid="stAlert"] p {{ font-size:14px !important; font-weight:600 !important; }}
[data-testid="stInfo"] p, [data-testid="stInfo"] span {{ color:#1a1a2e !important; }}

/* ── SCROLLBAR ── */
::-webkit-scrollbar {{ width:6px; }}
::-webkit-scrollbar-track {{ background:{c['bg']}; }}
::-webkit-scrollbar-thumb {{ background:{c['accent']}; border-radius:3px; }}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {{ border-radius:12px !important; overflow:hidden !important; }}
</style>
""", unsafe_allow_html=True)

inject_css()

# ── MODEL ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open("creditcard.pkl", "rb") as f:
            model = pickle.load(f)

        if hasattr(model, "feature_names_in_"):
            features = list(model.feature_names_in_)
        else:
            features = MODEL_FEATURES

        return model, features

    except Exception:
        st.error("❌ creditcard.pkl not found!")
        st.stop()

model, TRAINED_FEATURES = load_model()
model_source = "trained"

@st.cache_resource
def load_scaler():
    try:
        with open("scaler.pkl", "rb") as f:
            return pickle.load(f)
    except:
        st.error("❌ scaler.pkl not found!")
        st.stop()

scaler = load_scaler()

# ── PREPROCESSING ─────────────────────────────────────────────────────────────
import re as _re

def _coerce(val):
    if pd.isna(val): return np.nan
    s = str(val).strip()
    if s.lower() in ('','na','n/a','nan','null','none','?','-','--','error','#n/a','#value!'):
        return np.nan
    try:    return float(_re.sub(r'[^\d.\-eE]', '', s))
    except: return np.nan

def clean_dataset(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    # ✅ Use TRAINED scaler
    if 'Amount' in df.columns and 'scaled_amount' not in df.columns:
        df['scaled_amount'] = scaler.transform(df[['Amount']])
        df = df.drop(columns=['Amount'], errors='ignore')

    df = df.drop(columns=['Class'], errors='ignore')

    missing = [f for f in TRAINED_FEATURES if f not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    df = df[TRAINED_FEATURES]

    # Convert safely
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(df.median())

    return df

# ── KPI helper — uses custom CSS classes, NOT h2/h4 tags ─────────────────────
def kpi(label, value, color=None):
    val_style = f"color:{color}!important;" if color else ""
    st.markdown(
        f'<div class="kpi-card">'
        f'<span class="kpi-label">{label}</span>'
        f'<span class="kpi-value" style="{val_style}">{value}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
c = T()
with st.sidebar:
    st.markdown(
        f'<div style="text-align:center;padding:22px 0 10px;">'
        f'<div style="font-size:2.6rem;line-height:1;">🛡️</div>'
        f'<div style="font-size:1.2rem;font-weight:800;color:{c["accent"]};'
        f'font-family:Poppins,sans-serif;margin-top:8px;">FraudShield AI</div>'
        f'<div style="font-size:0.7rem;color:{c["subtext"]};letter-spacing:0.07em;'
        f'text-transform:uppercase;margin-top:3px;font-family:Poppins,sans-serif;">'
        f'Credit Card Fraud Detection</div></div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    for pid, label in [("bulk","📂  Bulk Scanner"),
                        ("dashboard","📊  Analytics Dashboard"),
                        ("contact","📞  Contact")]:
        active = st.session_state.page == pid
        st.markdown(f'<div class="{"nav-active" if active else "nav-inactive"}">', unsafe_allow_html=True)
        if st.button(label, key=f"nav_{pid}", use_container_width=True):
            st.session_state.page = pid
            st.rerun()
        st.markdown('</div><div style="height:6px"></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        f'<div style="font-size:11px;color:{c["subtext"]};text-transform:uppercase;'
        f'letter-spacing:0.08em;margin-bottom:6px;font-family:Poppins,sans-serif;">Appearance</div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="theme-btn">', unsafe_allow_html=True)
    if st.button("☀️  Light Mode" if st.session_state.theme == "dark" else "🌙  Dark Mode",
                 key="theme_toggle", use_container_width=True):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    if model_source == "ensemble":
        st.markdown(
            f'<div style="background:rgba(233,69,96,0.12);border:1px solid rgba(233,69,96,0.3);'
            f'border-radius:10px;padding:12px;font-size:12px;color:{c["subtext"]};line-height:1.6;'
            f'font-family:Poppins,sans-serif;">'
            f'&#8505;&#65039; Built-in Ensemble active.<br>'
            f'Place <b style="color:{c["accent"]};">creditcard.pkl</b> here to load your trained model.'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.success("✅ Trained model loaded")

# ── HEADER ────────────────────────────────────────────────────────────────────
c = T()
badge_bg  = 'rgba(0,184,148,0.15)' if model_source != 'ensemble' else 'rgba(233,69,96,0.15)'
badge_col = '#00b894'               if model_source != 'ensemble' else c['accent']
badge_txt = '&#x2705; Trained Model' if model_source != 'ensemble' else '&#x2699;&#xFE0F; Ensemble Model'

st.markdown(
    f'<div class="section-header" style="display:flex;align-items:center;'
    f'justify-content:space-between;flex-wrap:wrap;gap:10px;">'
    f'<div>'
    f'<div style="font-size:1.85rem;font-weight:800;color:{c["accent"]};'
    f'font-family:Poppins,sans-serif;">&#x1F6E1;&#xFE0F; FraudShield AI Dashboard</div>'
    f'<div style="font-size:14px;color:{c["subtext"]};margin-top:4px;'
    f'font-family:Poppins,sans-serif;">Intelligent Credit Card Fraud Detection System</div>'
    f'</div>'
    f'<span style="background:{badge_bg};border:1px solid {badge_col};color:{badge_col};'
    f'border-radius:20px;padding:4px 14px;font-size:12px;font-weight:700;'
    f'font-family:Poppins,sans-serif;">{badge_txt}</span>'
    f'</div>',
    unsafe_allow_html=True
)
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# BULK SCANNER
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "bulk":
    st.markdown(
        f'<div class="section-header">'
        f'<div style="font-size:1.2rem;font-weight:700;color:{c["accent"]};'
        f'font-family:Poppins,sans-serif;">&#x1F4C2; Bulk Transaction Fraud Scanner</div>'
        f'<div style="font-size:13px;color:{c["subtext"]};margin-top:4px;'
        f'font-family:Poppins,sans-serif;">Upload or import your dataset to run batch fraud detection.</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    sample_df = pd.DataFrame(np.random.normal(0, 1, (5, 28)),
                             columns=[f'V{i}' for i in range(1, 29)])
    sample_df['Time']          = [40000, 60000, 80000, 100000, 120000]
    sample_df['scaled_amount'] = np.random.normal(0, 1, 5)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f'<div style="font-size:1rem;font-weight:700;color:{c["accent2"]};'
                    f'font-family:Poppins,sans-serif;margin-bottom:8px;">&#x1F4E5; Sample File</div>',
                    unsafe_allow_html=True)
        fmt = st.selectbox("Format", ["CSV","Excel","JSON","SQLite"], key="sample_format")
        if st.button("Download", use_container_width=True):
            if fmt == "CSV":
                st.download_button("Download CSV", sample_df.to_csv(index=False).encode(),
                                   "sample_transactions.csv", use_container_width=True)
            elif fmt == "Excel":
                buf = BytesIO(); sample_df.to_excel(buf, index=False)
                st.download_button("Download Excel", buf.getvalue(),
                                   "sample_transactions.xlsx", use_container_width=True)
            elif fmt == "JSON":
                st.download_button("Download JSON", sample_df.to_json(orient="records"),
                                   "sample_transactions.json", use_container_width=True)
            elif fmt == "SQLite":
                conn = sqlite3.connect("sample_transactions.db")
                sample_df.to_sql("transactions", conn, if_exists="replace", index=False); conn.close()
                with open("sample_transactions.db","rb") as f:
                    st.download_button("Download DB", f, "sample_transactions.db",
                                       use_container_width=True)

    with col2:
        st.markdown(f'<div style="font-size:1rem;font-weight:700;color:{c["accent2"]};'
                    f'font-family:Poppins,sans-serif;margin-bottom:8px;">&#x1F517; Google Drive</div>',
                    unsafe_allow_html=True)
        import gdown, tempfile
        drive_link = st.text_input("Paste Link", key="drive")
        if st.button("Fetch", use_container_width=True):
            try:
                fid = drive_link.split("/d/")[1].split("/")[0]
                tmp = tempfile.NamedTemporaryFile(delete=False)
                gdown.download(f"https://drive.google.com/uc?id={fid}", tmp.name, quiet=False)
                st.session_state.uploaded_df = pd.read_csv(tmp.name)
                st.success("✅ Loaded from Google Drive!")
            except:
                st.error("❌ Invalid or inaccessible link.")

    with col3:
        st.markdown(f'<div style="font-size:1rem;font-weight:700;color:{c["accent2"]};'
                    f'font-family:Poppins,sans-serif;margin-bottom:8px;">&#x1F4E4; Upload File</div>',
                    unsafe_allow_html=True)
        file = st.file_uploader("CSV / Excel / JSON / DB", type=["csv","xlsx","json","db"],
                                label_visibility="collapsed")
        if file:
            try:
                if   file.name.endswith(".csv"):  df_up = pd.read_csv(file)
                elif file.name.endswith(".xlsx"): df_up = pd.read_excel(file)
                elif file.name.endswith(".json"): df_up = pd.read_json(file)
                elif file.name.endswith(".db"):
                    with open("temp.db","wb") as f2: f2.write(file.getbuffer())
                    conn  = sqlite3.connect("temp.db")
                    df_up = pd.read_sql("SELECT * FROM transactions", conn)
                st.session_state.uploaded_df = df_up
                st.success(f"✅ Uploaded: **{file.name}** ({len(df_up):,} rows)")
            except:
                st.error("❌ File error — check format.")

    st.markdown("---")

    if st.session_state.uploaded_df is not None:
        with st.expander(
            f"📋 Preview — {len(st.session_state.uploaded_df):,} rows "
            f"× {len(st.session_state.uploaded_df.columns)} columns", expanded=False
        ):
            st.dataframe(st.session_state.uploaded_df.head(10), use_container_width=True)

        if st.button("🚀 Run Fraud Detection", use_container_width=True):
            with st.spinner("Running detection model..."):
                X     = clean_dataset(st.session_state.uploaded_df)
                probs = model.predict_proba(X)[:, 1]
                threshold = np.percentile(probs, 70)
                preds = (probs >= threshold).astype(int)
                res = pd.DataFrame({
                    'Time':                  X['Time'].values,
                    'Scaled Amount':         X['scaled_amount'].values.round(4),
                    'Fraud Probability (%)': (probs * 100).round(2),
                    'Prediction':            [STATUS_MAP.get(int(p), str(p)) for p in preds],
                })
                res['Risk Level'] = res['Fraud Probability (%)'].apply(
                    lambda x: "🔴 High" if x >= 70 else ("🟡 Medium" if x >= 40 else "🟢 Low"))
                st.session_state.bulk_df = res

            n_fraud = int(preds.sum())
            ka, kb, kc = st.columns(3)
            with ka: kpi("Total Scanned",  f"{len(preds):,}")
            with kb: kpi("Fraud Flagged",  f"{n_fraud:,}", "#e94560")
            with kc: kpi("Fraud Rate",     f"{n_fraud/len(preds)*100:.2f}%")
            st.markdown("<br>", unsafe_allow_html=True)
            st.success(f"✅ Done — **{n_fraud}** fraudulent out of **{len(preds):,}** transactions.")
            st.markdown("### 📋 Detection Output Preview")
            st.dataframe(res.head(20), use_container_width=True)
            fig = px.pie(res, names="Prediction", hole=0.4, color="Prediction",
                         color_discrete_map=STATUS_COLOR, title="Transaction Classification")
            st.plotly_chart(style_fig(fig), use_container_width=True, key=chart_key())
            st.download_button("⬇️ Download Result CSV",
                               res.to_csv(index=False).encode(),
                               "fraud_detection_results.csv", use_container_width=True)
            st.info("💡 Switch to 📊 Analytics Dashboard in the sidebar for full insights.")
    else:
        st.markdown(
            f'<div style="background:{c["card"]};border:2px dashed {c["border"]};'
            f'border-radius:16px;padding:40px;text-align:center;">'
            f'<div style="font-size:3rem;margin-bottom:12px;">📁</div>'
            f'<div style="font-size:1.1rem;font-weight:600;color:{c["text"]};'
            f'font-family:Poppins,sans-serif;">No dataset loaded yet</div>'
            f'<div style="font-size:13px;color:{c["subtext"]};margin-top:6px;'
            f'font-family:Poppins,sans-serif;">Upload a file above or fetch from Google Drive.</div>'
            f'</div>',
            unsafe_allow_html=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "dashboard":
    st.markdown(
        f'<div class="section-header">'
        f'<div style="font-size:1.2rem;font-weight:700;color:{c["accent"]};'
        f'font-family:Poppins,sans-serif;">&#x1F4CA; Fraud Analytics Dashboard</div>'
        f'<div style="font-size:13px;color:{c["subtext"]};margin-top:4px;'
        f'font-family:Poppins,sans-serif;">Visual breakdown of detection results.</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    if st.session_state.bulk_df is not None:
        df       = st.session_state.bulk_df.copy()
        total    = len(df)
        n_fraud  = int((df['Prediction'] == 'Fraudulent').sum())
        n_legit  = total - n_fraud
        rate     = (n_fraud / total * 100) if total else 0
        avg_prob = df['Fraud Probability (%)'].mean()
        high_r   = int((df['Risk Level'] == '🔴 High').sum())
        med_r    = int((df['Risk Level'] == '🟡 Medium').sum())

        k1,k2,k3,k4 = st.columns(4)
        with k1: kpi("Total Transactions", f"{total:,}")
        with k2: kpi("Fraudulent Flagged",  f"{n_fraud:,}", "#e94560")
        with k3: kpi("Legitimate",          f"{n_legit:,}", "#00b894")
        with k4: kpi("Fraud Rate",          f"{rate:.2f}%")
        st.markdown("<br>", unsafe_allow_html=True)
        k5,k6,k7 = st.columns(3)
        with k5: kpi("🔴 High Risk",    f"{high_r:,}", "#d63031")
        with k6: kpi("🟡 Medium Risk",  f"{med_r:,}",  "#f39c12")
        with k7: kpi("Avg Fraud Score", f"{avg_prob:.1f}%")
        st.markdown("---")

        r1a,r1b = st.columns(2)
        with r1a:
            st.markdown('<div class="insight-box"><b>Insight:</b> Overall distribution of legitimate vs fraudulent transactions.</div>', unsafe_allow_html=True)
            f1 = px.pie(df, names="Prediction", title="Transaction Classification Overview",
                        color="Prediction", color_discrete_map=STATUS_COLOR, hole=0.4)
            f1.update_traces(textfont_size=14, textfont_family="Poppins")
            st.plotly_chart(style_fig(f1), use_container_width=True, key=chart_key())
        with r1b:
            rdf = df['Risk Level'].value_counts().reset_index()
            rdf.columns = ['Risk Level','Count']
            rdf['Color'] = rdf['Risk Level'].map({'🔴 High':'#d63031','🟡 Medium':'#f39c12','🟢 Low':'#00b894'})
            st.markdown('<div class="insight-box"><b>Insight:</b> Breakdown of transactions by risk severity.</div>', unsafe_allow_html=True)
            f2 = go.Figure(go.Bar(x=rdf['Risk Level'], y=rdf['Count'], marker_color=rdf['Color'],
                                   text=rdf['Count'], textposition="outside",
                                   textfont=dict(size=14, color=c['text'], family="Poppins")))
            f2.update_layout(title="Risk Level Distribution", yaxis_title="Transaction Count",
                             yaxis=dict(range=[0, rdf['Count'].max()*1.2]))
            st.plotly_chart(style_fig(f2), use_container_width=True, key=chart_key())

        r2a,r2b = st.columns(2)
        with r2a:
            st.markdown('<div class="insight-box"><b>Insight:</b> Distribution of fraud confidence scores.</div>', unsafe_allow_html=True)
            f3 = px.histogram(df, x='Fraud Probability (%)', nbins=30,
                              title="Fraud Probability Score Distribution",
                              color_discrete_sequence=['#e94560'])
            f3.update_traces(marker_line_width=1.5, marker_line_color="white")
            st.plotly_chart(style_fig(f3), use_container_width=True, key=chart_key())
        with r2b:
            st.markdown('<div class="insight-box"><b>Insight:</b> Fraud scores grouped by predicted class.</div>', unsafe_allow_html=True)
            f4 = px.box(df, x='Prediction', y='Fraud Probability (%)',
                        color='Prediction', color_discrete_map=STATUS_COLOR,
                        title="Fraud Score by Prediction Class")
            st.plotly_chart(style_fig(f4), use_container_width=True, key=chart_key())

        r3a,r3b = st.columns(2)
        with r3a:
            st.markdown('<div class="insight-box"><b>Insight:</b> When do fraudulent cases appear most?</div>', unsafe_allow_html=True)
            f5 = px.scatter(df, x='Time', y='Fraud Probability (%)',
                            color='Prediction', color_discrete_map=STATUS_COLOR,
                            title="Transaction Time vs Fraud Probability", opacity=0.7)
            f5.update_traces(marker=dict(size=6, line=dict(width=0.5, color="white")))
            st.plotly_chart(style_fig(f5), use_container_width=True, key=chart_key())
        with r3b:
            cdf = df['Prediction'].value_counts().reset_index()
            cdf.columns = ['Prediction','Count']
            cdf['Color'] = cdf['Prediction'].map(STATUS_COLOR)
            st.markdown('<div class="insight-box"><b>Insight:</b> Total legitimate vs fraudulent count.</div>', unsafe_allow_html=True)
            f6 = go.Figure(go.Bar(x=cdf['Prediction'], y=cdf['Count'], marker_color=cdf['Color'],
                                   text=cdf['Count'], textposition="outside",
                                   textfont=dict(size=15, color=c['text'], family="Poppins")))
            f6.update_layout(title="Legitimate vs Fraudulent Count", yaxis_title="Number of Transactions",
                             yaxis=dict(range=[0, cdf['Count'].max()*1.2]))
            st.plotly_chart(style_fig(f6), use_container_width=True, key=chart_key())
    else:
        st.markdown(
            f'<div style="background:{c["card"]};border:1px solid {c["border"]};'
            f'border-radius:16px;padding:40px;text-align:center;margin-top:20px;">'
            f'<div style="font-size:3rem;margin-bottom:12px;">📊</div>'
            f'<div style="font-size:1.1rem;font-weight:700;color:{c["accent"]};'
            f'font-family:Poppins,sans-serif;">No data to display yet</div>'
            f'<div style="font-size:14px;color:{c["subtext"]};margin-top:8px;'
            f'font-family:Poppins,sans-serif;">Go to Bulk Scanner, upload a dataset and run detection first.</div>'
            f'</div>',
            unsafe_allow_html=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# CONTACT PAGE
# KEY FIX: every st.markdown block is a plain string literal (no f-string),
# uses HTML numeric entities instead of emoji chars, and is kept short.
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "contact":

    st.markdown(
        '<div class="section-header">'
        '<div style="font-size:1.2rem;font-weight:700;color:#e94560;font-family:Poppins,sans-serif;">'
        '&#x1F4DE; Contact &amp; Connect</div>'
        '<div style="font-size:13px;color:#a0b8d0;margin-top:4px;font-family:Poppins,sans-serif;">'
        'Get in touch or explore the project.</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    _, mid, _ = st.columns([1, 2, 1])
    with mid:

        # --- avatar block ---
        st.markdown(
            '<div style="text-align:center;padding:28px 20px 16px;'
            'background:rgba(255,255,255,0.06);'
            'border:1px solid rgba(255,255,255,0.12);'
            'border-radius:18px 18px 0 0;border-bottom:none;">'
            '<div style="width:80px;height:80px;border-radius:50%;'
            'background:linear-gradient(135deg,#e94560,#c0392b);'
            'display:flex;align-items:center;justify-content:center;'
            'font-size:2rem;margin:0 auto 10px;">'
            '&#x1F6E1;&#xFE0F;</div>'
            '<div style="font-size:1.35rem;font-weight:800;color:#f0f4f8;'
            'font-family:Poppins,sans-serif;">Kavit Patel</div>'
            '<div style="font-size:0.7rem;color:#7ec8e3;letter-spacing:0.1em;'
            'text-transform:uppercase;margin-top:4px;font-family:Poppins,sans-serif;">'
            'Data Analyst &nbsp;&bull;&nbsp; AI/ML Developer</div>'
            '</div>',
            unsafe_allow_html=True
        )

        # --- three icon-link columns ---
        ia, ib, ic_ = st.columns(3)
        with ia:
            st.markdown(
                '<a href="mailto:kavitpatel1574.kp@gmail.com" '
                'style="display:block;text-align:center;padding:14px 4px;'
                'background:linear-gradient(135deg,#ea4335,#c23321);'
                'border-radius:12px;text-decoration:none;">'
                '<div style="font-size:1.4rem;line-height:1;">&#x1F4E7;</div>'
                '<div style="font-size:10.5px;font-weight:700;color:#fff;'
                'letter-spacing:0.06em;margin-top:5px;font-family:Poppins,sans-serif;">'
                'EMAIL</div></a>',
                unsafe_allow_html=True
            )
        with ib:
            st.markdown(
                '<a href="https://github.com/Kavit-shadow" target="_blank" '
                'style="display:block;text-align:center;padding:14px 4px;'
                'background:linear-gradient(135deg,#24292e,#555d66);'
                'border-radius:12px;text-decoration:none;">'
                '<div style="font-size:1.4rem;line-height:1;">&#x1F4BB;</div>'
                '<div style="font-size:10.5px;font-weight:700;color:#fff;'
                'letter-spacing:0.06em;margin-top:5px;font-family:Poppins,sans-serif;">'
                'GITHUB</div></a>',
                unsafe_allow_html=True
            )
        with ic_:
            st.markdown(
                '<a href="https://www.linkedin.com/in/kavit-patel-84597a26b" target="_blank" '
                'style="display:block;text-align:center;padding:14px 4px;'
                'background:linear-gradient(135deg,#0077b5,#005983);'
                'border-radius:12px;text-decoration:none;">'
                '<div style="font-size:1.4rem;line-height:1;">&#x1F517;</div>'
                '<div style="font-size:10.5px;font-weight:700;color:#fff;'
                'letter-spacing:0.06em;margin-top:5px;font-family:Poppins,sans-serif;">'
                'LINKEDIN</div></a>',
                unsafe_allow_html=True
            )

        # --- detail rows ---
        st.markdown(
            '<div style="background:rgba(255,255,255,0.06);'
            'border:1px solid rgba(255,255,255,0.12);'
            'border-radius:0 0 18px 18px;border-top:none;'
            'padding:18px 22px 22px;">'

            '<div style="margin-bottom:12px;">'
            '<div style="font-size:10px;font-weight:700;color:#7ec8e3;'
            'text-transform:uppercase;letter-spacing:0.09em;'
            'font-family:Poppins,sans-serif;margin-bottom:3px;">'
            '&#x1F4E7; Email</div>'
            '<a href="mailto:kavitpatel1574.kp@gmail.com" '
            'style="font-size:13px;color:#f0f4f8;text-decoration:none;'
            'font-family:Poppins,sans-serif;">'
            'kavitpatel1574.kp&#64;gmail.com</a>'
            '</div>'

            '<div style="margin-bottom:12px;">'
            '<div style="font-size:10px;font-weight:700;color:#7ec8e3;'
            'text-transform:uppercase;letter-spacing:0.09em;'
            'font-family:Poppins,sans-serif;margin-bottom:3px;">'
            '&#x1F4BB; GitHub</div>'
            '<a href="https://github.com/Kavit-shadow" target="_blank" '
            'style="font-size:13px;color:#f0f4f8;text-decoration:none;'
            'font-family:Poppins,sans-serif;">'
            'github.com/Kavit-shadow</a>'
            '</div>'

            '<div>'
            '<div style="font-size:10px;font-weight:700;color:#7ec8e3;'
            'text-transform:uppercase;letter-spacing:0.09em;'
            'font-family:Poppins,sans-serif;margin-bottom:3px;">'
            '&#x1F517; LinkedIn</div>'
            '<a href="https://www.linkedin.com/in/kavit-patel-84597a26b" target="_blank" '
            'style="font-size:13px;color:#f0f4f8;text-decoration:none;'
            'font-family:Poppins,sans-serif;">'
            'linkedin.com/in/kavit-patel-84597a26b</a>'
            '</div>'

            '</div>',
            unsafe_allow_html=True
        )

    st.write("")

    # --- about card ---
    st.markdown(
        '<div style="background:rgba(255,255,255,0.06);'
        'border:1px solid rgba(255,255,255,0.12);'
        'border-radius:16px;padding:24px 28px;">'
        '<div style="font-size:1rem;font-weight:700;color:#e94560;'
        'font-family:Poppins,sans-serif;margin-bottom:10px;">'
        '&#x1F9E0; About FraudShield AI</div>'
        '<div style="font-size:13.5px;color:#c0d0e0;line-height:1.8;'
        'font-family:Poppins,sans-serif;">'
        'FraudShield AI is an intelligent credit card fraud detection system that uses '
        '<b style="color:#f0f4f8;">Random Forest</b> and '
        '<b style="color:#f0f4f8;">Gradient Boosting</b> classifiers to identify '
        'fraudulent transactions in real time. It processes V1&#x2013;V28 PCA features '
        'along with transaction time and scaled amount.</div>'
        '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:14px;">'
        '<span style="background:rgba(233,69,96,0.15);border:1px solid rgba(233,69,96,0.35);'
        'color:#e94560;border-radius:20px;padding:4px 13px;font-size:11px;font-weight:700;'
        'font-family:Poppins,sans-serif;">&#x1F916; Machine Learning</span>'
        '<span style="background:rgba(0,184,148,0.15);border:1px solid rgba(0,184,148,0.35);'
        'color:#00b894;border-radius:20px;padding:4px 13px;font-size:11px;font-weight:700;'
        'font-family:Poppins,sans-serif;">&#x1F4CA; Analytics</span>'
        '<span style="background:rgba(126,200,227,0.15);border:1px solid rgba(126,200,227,0.35);'
        'color:#7ec8e3;border-radius:20px;padding:4px 13px;font-size:11px;font-weight:700;'
        'font-family:Poppins,sans-serif;">&#x1F512; Fraud Detection</span>'
        '<span style="background:rgba(243,156,18,0.15);border:1px solid rgba(243,156,18,0.35);'
        'color:#f39c12;border-radius:20px;padding:4px 13px;font-size:11px;font-weight:700;'
        'font-family:Poppins,sans-serif;">&#x26A1; Real-time</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(
    f'<hr style="border-color:{c["border"]};margin-top:28px;">'
    f'<div style="text-align:center;padding:12px 0;">'
    f'<div style="color:{c["accent2"]};font-size:14px;font-weight:700;'
    f'font-family:Poppins,sans-serif;">'
    f'&#x1F6E1;&#xFE0F; FraudShield AI &bull; Intelligent Credit Card Fraud Detection System</div>'
    f'<div style="color:{c["accent"]};font-size:13px;font-weight:700;'
    f'font-family:Poppins,sans-serif;margin-top:3px;">Created by Kavit</div>'
    f'</div>',
    unsafe_allow_html=True
)
