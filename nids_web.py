# -*- coding: utf-8 -*-
"""
nids_web.py — NIDS Detector v4 (UI Redesign)
Hybrid model: XGBoost + PyTorch NN · CIC-IDS2017
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import nids_core as core

# ─── Page config ────────────────────────────────────
st.set_page_config(
    page_title="NIDS Detector v4",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────
st.markdown("""
<style>
/* ──── Google Fonts ──── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Noto+Sans+Thai:wght@300;400;500;600;700&display=swap');

/* ──── keyframes ──── */
@keyframes pulse-dot { 0%,100%{opacity:1} 50%{opacity:.3} }
@keyframes float-y { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-6px)} }
@keyframes scan { 0%{left:-30%} 100%{left:130%} }
@keyframes gradient-x { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }

/* ──── global ──── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', 'Noto Sans Thai', sans-serif !important;
}
code, pre, .stCodeBlock, .stCodeBlock code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}
.main .block-container { max-width: 1100px; }

/* ──── sidebar ──── */
section[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #020617 0%, #0f172a 40%, #0c1425 100%);
}
section[data-testid="stSidebar"]::before {
    content: "";
    position: absolute; inset: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(6,182,212,.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(6,182,212,.04) 1px, transparent 1px);
    background-size: 24px 24px;
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
section[data-testid="stSidebar"] hr { border-color: #1e293b; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label { font-size: .82rem !important; }

/* ──── hero banner ──── */
.hero-box {
    background: linear-gradient(135deg, #020617 0%, #0c1e3a 50%, #020617 100%);
    border-radius: 20px;
    padding: 2.5rem 2.8rem 2rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(6,182,212,.15);
    position: relative;
    overflow: hidden;
}
.hero-box::before {
    content: "";
    position: absolute; top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 65% 25%, rgba(6,182,212,.12) 0%, transparent 55%);
}
.hero-box::after {
    content: "";
    position: absolute; top: 0; left: -30%;
    width: 30%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(6,182,212,.06), transparent);
    animation: scan 4s ease-in-out infinite;
}
.hero-box h1 {
    color: #f1f5f9;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 .5rem 0;
    position: relative;
    letter-spacing: -.02em;
}
.hero-box h1 .accent {
    background: linear-gradient(135deg, #06b6d4, #3b82f6, #8b5cf6);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradient-x 4s ease infinite;
}
.hero-box p {
    color: #94a3b8;
    font-size: .9rem;
    margin: 0;
    position: relative;
}
.hero-badge {
    display: inline-block;
    background: rgba(6,182,212,.1);
    border: 1px solid rgba(6,182,212,.25);
    color: #22d3ee !important;
    font-size: .72rem;
    font-weight: 500;
    padding: .2rem .65rem;
    border-radius: 20px;
    margin-right: .4rem;
    position: relative;
    font-family: 'JetBrains Mono', monospace;
}

/* ──── telecom wave decoration ──── */
.tele-wave {
    position: absolute;
    bottom: 0; right: 0;
    width: 280px; height: 80px;
    opacity: .25;
}

/* ──── metric cards ──── */
div[data-testid="stMetric"] {
    background: rgba(15,23,42,.6);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(51,65,85,.5);
    border-radius: 14px;
    padding: 1rem 1.2rem;
}
div[data-testid="stMetric"] label {
    color: #64748b !important;
    font-weight: 500;
    font-size: .78rem !important;
    text-transform: uppercase;
    letter-spacing: .06em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-weight: 700;
    font-size: 1.5rem !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ──── metric per attack color ──── */
.metric-bot div[data-testid="stMetric"] { border-top: 3px solid #ef4444; }
.metric-portscan div[data-testid="stMetric"] { border-top: 3px solid #3b82f6; }
.metric-ddos div[data-testid="stMetric"] { border-top: 3px solid #10b981; }
.metric-webattack div[data-testid="stMetric"] { border-top: 3px solid #f59e0b; }

/* ──── tabs ──── */
button[data-baseweb="tab"] {
    font-family: 'Space Grotesk', 'Noto Sans Thai', sans-serif !important;
    font-weight: 600 !important;
    font-size: .9rem !important;
    padding: .65rem 1.3rem !important;
    border-radius: 10px 10px 0 0 !important;
}

/* ──── attack result header ──── */
.attack-header {
    background: linear-gradient(90deg, #0f172a 0%, #1e293b 70%, #0f172a 100%);
    border-radius: 12px;
    padding: .85rem 1.5rem;
    margin: 1.5rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: .8rem;
    border: 1px solid #1e293b;
    position: relative;
    overflow: hidden;
}
.attack-header::before {
    content: "";
    position: absolute; left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, #06b6d4, #3b82f6);
    border-radius: 3px 0 0 3px;
}
.attack-header h3 {
    color: #f1f5f9;
    margin: 0;
    font-size: 1.15rem;
    font-weight: 600;
}
.attack-tag {
    background: rgba(6,182,212,.12);
    color: #22d3ee;
    font-size: .7rem;
    font-weight: 600;
    padding: .15rem .6rem;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    border: 1px solid rgba(6,182,212,.2);
}

/* ──── info box ──── */
.info-card {
    background: rgba(6,182,212,.06);
    border-left: 3px solid #06b6d4;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.3rem;
    margin: .5rem 0 1rem 0;
}
.info-card p { color: #94a3b8; margin: 0; font-size: .88rem; }

/* ──── status pill ──── */
.pill-ok {
    display: inline-block;
    background: rgba(34,197,94,.1); color: #4ade80;
    font-size: .78rem; font-weight: 500;
    padding: .15rem .7rem; border-radius: 20px;
    border: 1px solid rgba(34,197,94,.2);
}
.pill-warn {
    display: inline-block;
    background: rgba(245,158,11,.1); color: #fbbf24;
    font-size: .78rem; font-weight: 500;
    padding: .15rem .7rem; border-radius: 20px;
    border: 1px solid rgba(245,158,11,.2);
}

/* ──── separator ──── */
.sep { border: none; border-top: 1px solid #1e293b; margin: 1.8rem 0; }

/* ──── dashboard score cards ──── */
.score-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}
.score-card {
    background: rgba(15,23,42,.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(51,65,85,.5);
    border-radius: 16px;
    padding: 1.3rem;
    text-align: center;
    transition: transform .2s, border-color .2s;
    position: relative;
    overflow: hidden;
}
.score-card::before {
    content: "";
    position: absolute; inset: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(6,182,212,.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(6,182,212,.03) 1px, transparent 1px);
    background-size: 16px 16px;
}
.score-card:hover { transform: translateY(-3px); border-color: rgba(6,182,212,.3); }
.score-card .atk-name {
    font-size: .78rem; color: #64748b;
    font-weight: 600; text-transform: uppercase;
    letter-spacing: .08em;
    position: relative;
}
.score-card .atk-f1 {
    font-size: 2.4rem; font-weight: 700;
    margin: .3rem 0;
    font-family: 'JetBrains Mono', monospace;
    position: relative;
}
.score-card .atk-model {
    font-size: .68rem; color: #475569;
    font-family: 'JetBrains Mono', monospace;
    position: relative;
}

/* ──── upload area ──── */
.upload-placeholder {
    text-align: center;
    padding: 3.5rem 1.5rem;
    border: 2px dashed #1e293b;
    border-radius: 16px;
    background: rgba(15,23,42,.3);
}
.upload-placeholder .icon { font-size: 2.8rem; margin-bottom: .6rem; }
.upload-placeholder .title {
    font-size: 1.05rem; font-weight: 500; color: #64748b;
}
.upload-placeholder .sub {
    font-size: .82rem; color: #475569; margin-top: .3rem;
}

/* ──── buttons ──── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #0f172a, #1e293b) !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: .6rem 1.5rem !important;
}
.stDownloadButton > button:hover {
    border-color: #06b6d4 !important;
}
.stButton > button[kind="primary"] {
    border-radius: 12px !important;
    font-weight: 600 !important;
    background: linear-gradient(135deg, #0891b2, #0284c7) !important;
    border: none !important;
}

/* ──── dataframe ──── */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* ──── main bg ──── */
.stApp {
    background: linear-gradient(175deg, #020617 0%, #0f172a 100%);
}
.stApp > header { background: transparent !important; }

/* ──── override text colors for dark bg ──── */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stCaption, .stMarkdown h4,
.stMarkdown h3, .stMarkdown h5, label, .stSelectbox label {
    color: #cbd5e1 !important;
}
.stMarkdown h4, .stMarkdown h3 { color: #e2e8f0 !important; }
.stMarkdown strong { color: #f1f5f9 !important; }
.stMarkdown a { color: #22d3ee !important; }

/* ──── table dark override ──── */
.stMarkdown table { border-collapse: collapse; }
.stMarkdown th {
    background: rgba(15,23,42,.8) !important;
    color: #94a3b8 !important;
    border: 1px solid #1e293b !important;
    padding: .5rem .8rem !important;
    font-size: .82rem !important;
}
.stMarkdown td {
    background: rgba(15,23,42,.4) !important;
    color: #cbd5e1 !important;
    border: 1px solid #1e293b !important;
    padding: .5rem .8rem !important;
    font-size: .82rem !important;
}

/* ──── expander dark ──── */
.streamlit-expanderHeader { color: #94a3b8 !important; }

/* ──── code block dark ──── */
.stCodeBlock { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ─── Matplotlib style ───────────────────────────────
CHART_BG = "#0a0f1a"
CHART_COLORS = {"Bot": "#ef4444", "WebAttack": "#f59e0b",
                "PortScan": "#3b82f6", "DDoS": "#10b981",
                "Web Attack": "#f59e0b"}

def styled_fig(w=8, h=4):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=CHART_BG)
    ax.set_facecolor(CHART_BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#1e293b")
    ax.spines["bottom"].set_color("#1e293b")
    ax.tick_params(colors="#64748b", labelsize=9)
    ax.xaxis.label.set_color("#94a3b8")
    ax.yaxis.label.set_color("#94a3b8")
    ax.title.set_color("#e2e8f0")
    return fig, ax


@st.cache_data(show_spinner=False)
def _load(file_bytes, name):
    return core.load_dataframe(file_bytes, name)


# ─── Sidebar ────────────────────────────────────────
bundles = core.list_bundles()

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.5rem 0 .5rem 0;">
        <div style="margin:0 auto; width:56px; height:56px; border-radius:14px;
                    background:linear-gradient(135deg,#06b6d4,#3b82f6);
                    display:flex; align-items:center; justify-content:center;
                    box-shadow: 0 4px 20px rgba(6,182,212,.25);">
            <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="#fff" stroke-width="2">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
        </div>
        <div style="font-size:1.2rem; font-weight:700; letter-spacing:.03em; margin-top:.6rem;
                    font-family:'Space Grotesk',sans-serif;">
            NIDS<span style="color:#22d3ee !important;">.detect</span>
        </div>
        <div style="font-size:.68rem; color:#475569 !important; margin-top:.15rem;
                    font-family:'JetBrains Mono',monospace; letter-spacing:.05em;">
            HYBRID ML PIPELINE v4
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if bundles:
        st.markdown(f"""
        <div style="background:rgba(6,182,212,.08); border:1px solid rgba(6,182,212,.2);
                    border-radius:10px; padding:.6rem 1rem;
                    font-size:.82rem; text-align:center; margin-bottom:.8rem;
                    display:flex; align-items:center; justify-content:center; gap:.5rem;">
            <span style="width:6px;height:6px;border-radius:50%;background:#22d3ee;
                        display:inline-block;animation:pulse-dot 1.5s ease-in-out infinite;"></span>
            Models ready: <strong style="color:#22d3ee !important;">{len(bundles)}</strong>
        </div>
        """, unsafe_allow_html=True)
        for b in bundles:
            st.markdown(f"""
            <div style="background:rgba(15,23,42,.6); border:1px solid #1e293b; border-radius:10px;
                        padding:.45rem .8rem; margin:.3rem 0; font-size:.8rem;
                        display:flex; justify-content:space-between; align-items:center;
                        backdrop-filter:blur(8px);">
                <span style="display:flex;align-items:center;gap:.4rem;">
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none"
                         stroke="#06b6d4" stroke-width="2">
                        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                    </svg>
                    {b}
                </span>
                <span style="color:#4ade80 !important; font-size:.65rem;
                            font-family:'JetBrains Mono',monospace;">READY</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("ยังไม่มีโมเดล — รัน train_backend.py ก่อน")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:#64748b; line-height:1.6;">
        <strong>Dataset:</strong> CIC-IDS2017<br>
        <strong>Pipeline:</strong> Hybrid v4<br>
        <strong>XGBoost:</strong> Bot, WebAttack<br>
        <strong>Neural Net:</strong> PortScan, DDoS
    </div>
    """, unsafe_allow_html=True)


# ─── Hero banner ────────────────────────────────────
st.markdown("""
<div class="hero-box">
    <svg class="tele-wave" viewBox="0 0 280 80" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M0 40 Q35 15 70 40 T140 40 T210 40 T280 40" stroke="#06b6d4" stroke-width="1.5" opacity=".4">
            <animate attributeName="d" dur="3s" repeatCount="indefinite"
                values="M0 40 Q35 15 70 40 T140 40 T210 40 T280 40;M0 40 Q35 65 70 40 T140 40 T210 40 T280 40;M0 40 Q35 15 70 40 T140 40 T210 40 T280 40" />
        </path>
        <path d="M0 50 Q35 25 70 50 T140 50 T210 50 T280 50" stroke="#3b82f6" stroke-width="1" opacity=".2">
            <animate attributeName="d" dur="4s" repeatCount="indefinite"
                values="M0 50 Q35 25 70 50 T140 50 T210 50 T280 50;M0 50 Q35 75 70 50 T140 50 T210 50 T280 50;M0 50 Q35 25 70 50 T140 50 T210 50 T280 50" />
        </path>
        <circle cx="40" cy="40" r="2" fill="#06b6d4" opacity=".6">
            <animate attributeName="opacity" values=".6;.2;.6" dur="2s" repeatCount="indefinite" />
        </circle>
        <circle cx="140" cy="35" r="2" fill="#3b82f6" opacity=".5">
            <animate attributeName="opacity" values=".5;.15;.5" dur="2.5s" repeatCount="indefinite" />
        </circle>
        <circle cx="220" cy="45" r="2" fill="#06b6d4" opacity=".4">
            <animate attributeName="opacity" values=".4;.1;.4" dur="1.8s" repeatCount="indefinite" />
        </circle>
        <line x1="40" y1="40" x2="140" y2="35" stroke="#06b6d4" stroke-width=".5" opacity=".2" stroke-dasharray="4 4">
            <animate attributeName="stroke-dashoffset" values="8;0" dur="1.5s" repeatCount="indefinite" />
        </line>
        <line x1="140" y1="35" x2="220" y2="45" stroke="#3b82f6" stroke-width=".5" opacity=".2" stroke-dasharray="4 4">
            <animate attributeName="stroke-dashoffset" values="8;0" dur="2s" repeatCount="indefinite" />
        </line>
    </svg>
    <h1><span class="accent">Network Intrusion</span> Detection System</h1>
    <p>
        <span class="hero-badge">CIC-IDS2017</span>
        <span class="hero-badge">Hybrid v4</span>
        <span class="hero-badge">XGBoost + NN</span>
        Binary classifier x 4 attack types &mdash; Bot &middot; PortScan &middot; DDoS &middot; WebAttack
    </p>
</div>
""", unsafe_allow_html=True)


# ─── Tabs ───────────────────────────────────────────
tab_detect, tab_feat, tab_dash = st.tabs(
    ["🔎 ตรวจจับ", "🧹 Feature Tool", "📊 Dashboard"])


# ═════════════════════════════════════════════════════
# TAB 1: DETECT
# ═════════════════════════════════════════════════════
with tab_detect:
    st.markdown("""
    <div class="info-card">
        <p>📂 อัปโหลดไฟล์ traffic → โมเดลจะเลือก feature ที่ต้องใช้เอง → แสดงผลตรวจจับ</p>
    </div>
    """, unsafe_allow_html=True)

    up_detect = st.file_uploader("อัปโหลดไฟล์ traffic (.xlsx / .csv)",
                                  type=["xlsx", "xls", "csv"], key="detect_up")

    if up_detect is None:
        st.markdown("""
        <div class="upload-placeholder">
            <div class="icon">
                <svg viewBox="0 0 24 24" width="48" height="48" fill="none"
                     stroke="#334155" stroke-width="1.5" style="margin:0 auto;">
                    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
                    <polyline points="17 8 12 3 7 8"/>
                    <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
            </div>
            <div class="title">Drop your traffic file here</div>
            <div class="sub">.xlsx, .xls, .csv &middot; No feature cutting needed</div>
        </div>
        """, unsafe_allow_html=True)
    elif not bundles:
        st.error("ยังไม่มีโมเดล — รัน train_backend.py ก่อน")
    else:
        with st.spinner("กำลังโหลดข้อมูล..."):
            df_det = _load(up_detect.getvalue(), up_detect.name)

        label_col_det = core.detect_label_column(df_det)
        has_lbl_det = label_col_det is not None
        feats_det = [c for c in df_det.columns if c != label_col_det]

        # ─── data summary metrics ───
        if has_lbl_det:
            y_det, atk_name_det = core.make_binary_target(df_det, label_col_det)
            n_benign = int((y_det == 0).sum())
            n_attack = int((y_det == 1).sum())
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("BENIGN", f"{n_benign:,}")
            m2.metric(atk_name_det, f"{n_attack:,}")
            m3.metric("Total Rows", f"{len(df_det):,}")
            m4.metric("Features", f"{len(feats_det)}")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Rows", f"{len(df_det):,}")
            m2.metric("Features", f"{len(feats_det)}")
            m3.metric("Label", "ไม่มี — ทำนายอย่างเดียว")

        # ─── feature compatibility ───
        with st.expander("🔍 ดู feature ที่แต่ละโมเดลต้องการ vs ไฟล์ที่อัป"):
            for atk in bundles:
                meta = core.load_bundle(atk)[1]
                needed = meta["features"]
                missing = [f for f in needed if f not in df_det.columns]
                if missing:
                    st.markdown(f'<span class="pill-warn">⚠️ {atk}</span> ขาด {len(missing)} feature: {", ".join(missing)}', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="pill-ok">✅ {atk}</span> feature ครบ ({len(needed)} ตัว)', unsafe_allow_html=True)

        chosen = st.multiselect("เลือกโมเดล", bundles, default=bundles, key="det_models")

        if st.button("▶️ ตรวจจับ", type="primary", use_container_width=True) and chosen:
            from sklearn.metrics import classification_report, confusion_matrix
            results = df_det.copy()

            for atk in chosen:
                try:
                    model, meta = core.load_bundle(atk)
                    proba, pred = core.predict_with_bundle(model, meta, df_det)
                except KeyError as e:
                    missing_list = e.args[0] if isinstance(e.args[0], list) else [str(e)]
                    st.warning(f"⚠️ **{atk}**: ไฟล์ขาด feature: {', '.join(str(x) for x in missing_list)}")
                    continue
                except Exception as e:
                    st.error(f"{atk}: {e}")
                    continue

                results[f"proba_{atk}"] = np.round(proba, 4)
                results[f"pred_{atk}"] = pred

                # attack header
                st.markdown(f"""
                <div class="attack-header">
                    <h3>{atk}</h3>
                    <span class="attack-tag">{meta['model_type'].upper()}</span>
                    <span class="attack-tag">THR={meta['threshold']:.3f}</span>
                </div>
                """, unsafe_allow_html=True)

                if has_lbl_det:
                    yt, _ = core.make_binary_target(df_det, label_col_det)

                    # threshold = 0.5
                    pred_05 = (proba >= 0.5).astype(int)
                    report_05 = classification_report(
                        yt, pred_05, target_names=["Benign", atk], zero_division=0)
                    st.markdown("**Threshold = 0.5** (ค่าเริ่มต้น)")
                    st.code(report_05, language=None)

                    # tuned threshold
                    thr = meta["threshold"]
                    report_tuned = classification_report(
                        yt, pred, target_names=["Benign", atk], zero_division=0)
                    st.markdown(f"**Threshold = {thr:.3f}** (tuned เพื่อ F1 สูงสุด)")
                    st.code(report_tuned, language=None)

                    # accuracy metrics
                    acc_05 = (yt == pred_05).mean() * 100
                    acc_tuned = (yt == pred).mean() * 100
                    ac1, ac2 = st.columns(2)
                    ac1.metric("Accuracy (thr=0.5)", f"{acc_05:.2f}%")
                    ac2.metric("Accuracy (tuned)", f"{acc_tuned:.2f}%")

                    # confusion matrix
                    cm = confusion_matrix(yt, pred)
                    fig, ax = styled_fig(5.5, 3.5)
                    cmap = plt.cm.Blues
                    im = ax.imshow(cm, cmap=cmap, aspect="auto")
                    for (i, j), v in np.ndenumerate(cm):
                        ax.text(j, i, f"{v:,}", ha="center", va="center",
                                color="white" if v > cm.max()/2 else "#1e293b",
                                fontsize=14, fontweight="bold")
                    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
                    ax.set_xticklabels(["Pred Benign", f"Pred {atk}"])
                    ax.set_yticklabels(["Actual Benign", f"Actual {atk}"])
                    ax.set_title(f"Confusion Matrix: {atk} [{meta['model_type']}]",
                                 fontsize=11, fontweight="bold", color="#e2e8f0")
                    ax.set_ylabel("Reality", fontsize=9)
                    ax.set_xlabel("AI Prediction", fontsize=9)
                    fig.colorbar(im, fraction=0.046)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    # probability distribution (split by actual class)
                    proba_benign = proba[yt == 0]
                    proba_attack = proba[yt == 1]
                    fig, ax = styled_fig(8, 3.5)
                    ax.hist(proba_benign, bins=50, color="#3b82f6", alpha=0.7,
                            edgecolor="white", linewidth=0.5,
                            label=f"Actual Benign ({len(proba_benign):,})")
                    ax.hist(proba_attack, bins=50, color="#ef4444", alpha=0.7,
                            edgecolor="white", linewidth=0.5,
                            label=f"Actual {atk} ({len(proba_attack):,})")
                    ax.axvline(thr, color="#e2e8f0", linestyle="--", linewidth=2,
                               label=f"Threshold = {thr:.3f}")
                    ax.set_title(f"Probability Distribution: {atk}",
                                 fontsize=11, fontweight="bold", color="#e2e8f0")
                    ax.set_xlabel(f"P({atk})", fontsize=9)
                    ax.set_ylabel("Number of flows", fontsize=9)
                    ax.legend(fontsize=9, loc="upper center")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                else:
                    thr = meta["threshold"]
                    n_total = len(pred)
                    n_atk = int(pred.sum())
                    n_ben = n_total - n_atk

                    pred_05 = (proba >= 0.5).astype(int)
                    n_atk_05 = int(pred_05.sum())
                    n_ben_05 = n_total - n_atk_05

                    st.markdown("**Threshold = 0.5** (ค่าเริ่มต้น)")
                    report_05 = (
                        f"{'':>14}{'predicted':>12}{'% of total':>12}\n"
                        f"{'':>14}{'-'*24}\n"
                        f"{'Benign':>14}{n_ben_05:>12,}{n_ben_05/n_total*100:>11.2f}%\n"
                        f"{atk:>14}{n_atk_05:>12,}{n_atk_05/n_total*100:>11.2f}%\n"
                        f"{'':>14}{'-'*24}\n"
                        f"{'Total':>14}{n_total:>12,}{'100.00%':>12}"
                    )
                    st.code(report_05, language=None)

                    st.markdown(f"**Threshold = {thr:.3f}** (tuned เพื่อ F1 สูงสุด)")
                    report_tuned = (
                        f"{'':>14}{'predicted':>12}{'% of total':>12}\n"
                        f"{'':>14}{'-'*24}\n"
                        f"{'Benign':>14}{n_ben:>12,}{n_ben/n_total*100:>11.2f}%\n"
                        f"{atk:>14}{n_atk:>12,}{n_atk/n_total*100:>11.2f}%\n"
                        f"{'':>14}{'-'*24}\n"
                        f"{'Total':>14}{n_total:>12,}{'100.00%':>12}"
                    )
                    st.code(report_tuned, language=None)

                    ac1, ac2 = st.columns(2)
                    ac1.metric("Benign (ปกติ)", f"{n_ben:,}")
                    ac2.metric(f"Flag เป็น {atk}", f"{n_atk:,}")

                    # charts
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5),
                                                    facecolor=CHART_BG)
                    for _ax in (ax1, ax2):
                        _ax.set_facecolor(CHART_BG)
                        _ax.spines["top"].set_visible(False)
                        _ax.spines["right"].set_visible(False)
                        _ax.spines["left"].set_color("#1e293b")
                        _ax.spines["bottom"].set_color("#1e293b")
                        _ax.tick_params(colors="#64748b", labelsize=9)

                    bars = ax1.bar(["Benign", atk], [n_ben, n_atk],
                                   color=["#3b82f6", "#ef4444"], width=0.5,
                                   edgecolor="white", linewidth=1.5)
                    for bar, val in zip(bars, [n_ben, n_atk]):
                        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                f"{val:,}", ha="center", va="bottom", fontsize=10,
                                fontweight="bold", color="#e2e8f0")
                    ax1.set_title(f"Prediction Summary (thr={thr:.3f})",
                                  fontsize=10, fontweight="bold", color="#e2e8f0")
                    ax1.set_ylabel("Number of flows", fontsize=9)

                    # split histogram by predicted class
                    proba_pred_ben = proba[pred == 0]
                    proba_pred_atk = proba[pred == 1]
                    ax2.hist(proba_pred_ben, bins=50, color="#3b82f6", alpha=0.7,
                             edgecolor="white", linewidth=0.5,
                             label=f"Predicted Benign ({len(proba_pred_ben):,})")
                    ax2.hist(proba_pred_atk, bins=50, color="#ef4444", alpha=0.7,
                             edgecolor="white", linewidth=0.5,
                             label=f"Predicted {atk} ({len(proba_pred_atk):,})")
                    ax2.axvline(thr, color="#e2e8f0", linestyle="--", linewidth=2,
                               label=f"Threshold = {thr:.3f}")
                    ax2.set_title("Probability Distribution",
                                  fontsize=10, fontweight="bold", color="#e2e8f0")
                    ax2.set_xlabel(f"P({atk})", fontsize=9)
                    ax2.set_ylabel("Number of flows", fontsize=9)
                    ax2.legend(fontsize=7, loc="upper center")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    st.caption("⚠️ ไม่มี Label — ประเมิน precision/recall ไม่ได้ แสดงเฉพาะผลทำนาย")

                st.markdown('<hr class="sep">', unsafe_allow_html=True)

            st.download_button(
                "⬇️ ดาวน์โหลดผลตรวจจับ (.csv)",
                results.to_csv(index=False).encode("utf-8-sig"),
                file_name=up_detect.name.rsplit(".", 1)[0] + "_detected.csv",
                mime="text/csv",
                use_container_width=True)


# ═════════════════════════════════════════════════════
# TAB 2: FEATURE TOOL
# ═════════════════════════════════════════════════════
with tab_feat:
    st.markdown("""
    <div class="info-card">
        <p>🧹 เครื่องมือสำรวจ/ตัด feature แยกต่างหาก — ไม่กระทบแท็บตรวจจับ ·
        ใช้สำหรับเตรียมไฟล์ก่อนเทรนใหม่ หรือวินิจฉัยไฟล์จากแหล่งอื่น</p>
    </div>
    """, unsafe_allow_html=True)

    up_feat = st.file_uploader("อัปโหลดไฟล์สำหรับสำรวจ feature",
                                type=["xlsx", "xls", "csv"], key="feat_up")

    if up_feat is None:
        st.markdown("""
        <div class="upload-placeholder">
            <div class="icon">
                <svg viewBox="0 0 24 24" width="48" height="48" fill="none"
                     stroke="#334155" stroke-width="1.5" style="margin:0 auto;">
                    <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>
                </svg>
            </div>
            <div class="title">Upload a file to explore features</div>
            <div class="sub">Feature stats, compatibility check, RF baseline</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("กำลังโหลดข้อมูล..."):
            df_ft = _load(up_feat.getvalue(), up_feat.name)

        label_col_ft = core.detect_label_column(df_ft)
        has_lbl_ft = label_col_ft is not None
        feats_ft = [c for c in df_ft.columns if c != label_col_ft]

        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("แถว", f"{len(df_ft):,}")
        fc2.metric("Features", len(feats_ft))
        fc3.metric("มี Label?", "มี ✅" if has_lbl_ft else "ไม่มี")

        # compatibility
        comp = core.compatibility_check(feats_ft)
        with st.expander(f"🔍 เช็คความเข้ากันได้ — ใกล้ {comp['best_set']} "
                         f"({comp['overlap']}/{comp['ref_size']})"):
            if not comp["missing"] and not comp["extra"]:
                st.markdown('<span class="pill-ok">✅ feature ตรงชุดมาตรฐานพอดี</span>', unsafe_allow_html=True)
            else:
                if comp["missing"]:
                    st.warning(f"ขาด {len(comp['missing'])}: {', '.join(comp['missing'])}")
                if comp["extra"]:
                    st.info(f"เกิน {len(comp['extra'])}: {', '.join(comp['extra'])}")

        # stats
        st.markdown("**สถิติราย feature**")
        stats = core.feature_stats(df_ft, feats_ft)
        st.dataframe(stats, use_container_width=True, height=260)

        # auto-drop
        cc1, cc2 = st.columns(2)
        with cc1:
            drop_nzv = st.checkbox("ตัด feature ค่าคงที่ (variance≈0)", value=True, key="ft_nzv")
        with cc2:
            corr_thr = st.slider("ตัด feature correlation สูงเกิน",
                                  0.90, 1.00, 0.98, 0.01, key="ft_corr")

        suggested = core.auto_drop_suggestion(df_ft, feats_ft, corr_thr, drop_nzv)
        if suggested:
            st.caption(f"แนะนำให้ตัด ({len(suggested)}): {', '.join(suggested)}")

        default_keep = [f for f in feats_ft if f not in suggested]
        selected = st.multiselect("feature ที่จะเก็บไว้ (เอาออก = ตัดทิ้ง)",
                                  options=feats_ft, default=default_keep, key="ft_sel")
        st.write(f"เก็บไว้ **{len(selected)}** / {len(feats_ft)}")

        # RF baseline
        if has_lbl_ft and selected:
            st.markdown("**RandomForest Baseline** (วินิจฉัย)")
            rc1, rc2 = st.columns(2)
            with rc1:
                quick = st.checkbox("Quick mode", value=len(df_ft) > 80000, key="ft_quick")
            with rc2:
                n_est = st.slider("n_estimators", 50, 300, 150, 50, key="ft_nest")
            if st.button("▶️ รัน RF baseline", key="ft_rf"):
                y, _ = core.make_binary_target(df_ft, label_col_ft)
                with st.spinner("กำลังเทรน RandomForest..."):
                    res = core.run_random_forest(
                        df_ft, selected, y,
                        subsample_n=min(60000, len(df_ft)) if quick else None,
                        n_estimators=n_est)
                st.session_state["rf"] = res
            if "rf" in st.session_state:
                res = st.session_state["rf"]
                rm1, rm2, rm3 = st.columns(3)
                rm1.metric("ROC-AUC", f"{res['auc']:.4f}")
                rm2.metric("PR-AUC", f"{res['ap']:.4f}")
                rm3.metric("F1", f"{res['f1']:.3f}")
                auc = res["auc"]
                if auc >= 0.99:
                    st.success("🟢 feature ดีมาก — ถ้าโมเดลจริงพลาด ปัญหาอยู่ที่ config/threshold")
                elif auc >= 0.95:
                    st.info("🟡 feature ใช้ได้ดี")
                elif auc >= 0.85:
                    st.warning("🟠 feature พอใช้ — ควรทบทวน")
                else:
                    st.error("🔴 มีปัญหาที่ feature/data")

                imp = pd.DataFrame({"feature": res["features"],
                                    "importance": res["importances"]}).sort_values("importance")
                fig, ax = styled_fig(5, max(3, 0.25 * len(imp)))
                ax.barh(imp["feature"], imp["importance"], color="#3b82f6",
                        edgecolor="white", linewidth=0.5)
                ax.set_title("Feature Importance (RF)", fontsize=11,
                             fontweight="bold", color="#e2e8f0")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # export
        if selected:
            st.markdown("**Export ไฟล์ที่ตัดแล้ว**")
            cols = selected + ([label_col_ft] if has_lbl_ft else [])
            st.download_button(
                "⬇️ ดาวน์โหลด .csv",
                df_ft[cols].to_csv(index=False).encode("utf-8-sig"),
                file_name=up_feat.name.rsplit(".", 1)[0] + "_selected.csv",
                mime="text/csv",
                use_container_width=True)


# ═════════════════════════════════════════════════════
# TAB 3: DASHBOARD
# ═════════════════════════════════════════════════════
with tab_dash:
    if not bundles:
        st.error("ยังไม่มีโมเดล — รัน train_backend.py ก่อน")
    else:
        import joblib, os

        # ─── load all metas ───
        all_meta = {}
        for atk in bundles:
            all_meta[atk] = joblib.load(os.path.join(core.MODELS_DIR, f"{atk}_meta.pkl"))

        # ─── F1 score cards ───
        st.markdown("#### ผลลัพธ์โมเดล (tuned threshold)")
        cards_html = '<div class="score-grid">'
        for atk, meta in all_meta.items():
            m = meta.get("metrics", {})
            f1 = m.get("f1", 0)
            color = CHART_COLORS.get(atk, "#3b82f6")
            cards_html += f"""
            <div class="score-card" style="border-top: 4px solid {color};">
                <div class="atk-name">{atk}</div>
                <div class="atk-f1" style="color:{color};">{f1:.3f}</div>
                <div style="font-size:0.78rem; color:#64748b;">
                    P={m.get('precision',0):.3f} · R={m.get('recall',0):.3f} · AUC={m.get('auc',0):.4f}
                </div>
                <div class="atk-model">{meta['model_type'].upper()} · {len(meta['features'])} feat · thr={meta['threshold']:.3f}</div>
            </div>"""
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

        st.markdown('<hr class="sep">', unsafe_allow_html=True)

        # ─── v1 → v4 progression ───
        st.markdown("#### พัฒนาการ F1-score: v1 → v4")

        versions = ["v1", "v2", "v3", "v4"]
        f1_data = {
            "Bot":       [0.22, 0.73, 0.80, 0.97],
            "WebAttack": [0.57, 0.79, 0.80, 0.95],
            "PortScan":  [0.999, 0.999, 0.999, 0.999],
            "DDoS":      [0.999, 0.999, 0.999, 0.999],
        }

        fig, ax = styled_fig(9, 4.5)
        x = np.arange(len(versions))
        width = 0.19
        for i, (atk_name, f1s) in enumerate(f1_data.items()):
            c = CHART_COLORS.get(atk_name, "#3b82f6")
            bars = ax.bar(x + i * width, f1s, width, label=atk_name, color=c,
                          edgecolor="white", linewidth=0.8)
            for bar, val in zip(bars, f1s):
                if val < 0.99:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=8,
                            fontweight="bold", color="#e2e8f0")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(versions, fontsize=11, fontweight="bold")
        ax.set_ylabel("F1-score", fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_title("F1-score Progression: v1 to v4", fontsize=13,
                      fontweight="bold", color="#e2e8f0", pad=12)
        ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
        ax.axhline(y=0.95, color="#94a3b8", linestyle=":", alpha=0.5, linewidth=1)
        ax.text(3.7, 0.955, "target 0.95", fontsize=7, color="#94a3b8")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # version explanation
        st.markdown("""
| Version | การแก้ไข | ผลลัพธ์หลัก |
|:--------|:---------|:------------|
| **v1** | NN + inverse class weight ตรงๆ | Bot P=13%, WebAttack P=40% — FP ล้น |
| **v2** | √weight + threshold tuning | Bot F1: 0.22→0.73, WebAttack: 0.57→0.79 |
| **v3** | +SMOTE (train set only) | Bot: 0.73→0.80, WebAttack: 0.79→0.80 |
| **v4** | **Hybrid** XGBoost (imbalanced) + NN (balanced) | Bot: **0.97**, WebAttack: **0.95** |
""")

        st.markdown('<hr class="sep">', unsafe_allow_html=True)

        # ─── Feature importance ───
        st.markdown("#### Feature Importance เปรียบเทียบข้าม Attack")
        st.caption("แสดงเฉพาะ XGBoost (Bot/WebAttack) — NN ไม่มี built-in feature importance")

        xgb_attacks = [atk for atk in bundles if all_meta[atk]["model_type"] == "xgboost"]

        if xgb_attacks:
            n_xgb = len(xgb_attacks)
            fig, axes = plt.subplots(1, n_xgb, figsize=(6 * n_xgb, 6), facecolor=CHART_BG)
            if n_xgb == 1:
                axes = [axes]
            for ax, atk in zip(axes, xgb_attacks):
                ax.set_facecolor(CHART_BG)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_color("#1e293b")
                ax.spines["bottom"].set_color("#1e293b")

                model_xgb, meta_xgb = core.load_bundle(atk)
                feats = meta_xgb["features"]
                importances = model_xgb.feature_importances_
                imp_df = pd.DataFrame({"feature": feats, "importance": importances})
                imp_df = imp_df.sort_values("importance", ascending=True)
                c = CHART_COLORS.get(atk, "#3b82f6")
                ax.barh(imp_df["feature"], imp_df["importance"], color=c,
                        edgecolor="white", linewidth=0.5, height=0.7)
                ax.set_title(f"Feature Importance: {atk}", fontsize=11,
                             fontweight="bold", color="#e2e8f0")
                ax.set_xlabel("Importance", fontsize=9)
                ax.tick_params(labelsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # top 5 comparison
            st.markdown("##### Top 5 features ที่สำคัญที่สุด")
            top_cols = st.columns(n_xgb)
            for col, atk in zip(top_cols, xgb_attacks):
                model_xgb, meta_xgb = core.load_bundle(atk)
                feats = meta_xgb["features"]
                importances = model_xgb.feature_importances_
                top5 = sorted(zip(feats, importances), key=lambda x: -x[1])[:5]
                c = CHART_COLORS.get(atk, "#3b82f6")
                with col:
                    st.markdown(f"""
                    <div style="border:1px solid #e2e8f0; border-top:3px solid {c};
                                border-radius:10px; padding:1rem; margin-bottom:0.5rem;">
                        <div style="font-weight:700; font-size:0.95rem; margin-bottom:0.8rem;
                                    color:#0f172a;">{atk}</div>
                    """ + "".join([
                        f'<div style="display:flex; justify-content:space-between; '
                        f'padding:0.25rem 0; font-size:0.82rem; color:#334155;">'
                        f'<span>{rank}. {f}</span>'
                        f'<span style="font-family:JetBrains Mono,monospace; color:{c}; '
                        f'font-weight:600;">{v:.4f}</span></div>'
                        for rank, (f, v) in enumerate(top5, 1)
                    ]) + "</div>", unsafe_allow_html=True)

            st.caption("💡 feature importance ขึ้นกับชนิด attack — "
                       "นี่คือเหตุผลที่ต้องแยกโมเดล 1 ตัวต่อ 1 attack")
        else:
            st.info("ไม่มีโมเดล XGBoost — ไม่สามารถแสดง feature importance")
