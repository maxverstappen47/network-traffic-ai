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
/* ──── import Google font ──── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ──── global ──── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans Thai', sans-serif;
}
code, pre, .stCodeBlock code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* ──── sidebar ──── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #334155;
}

/* ──── hero banner ──── */
.hero-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid #1e40af44;
    position: relative;
    overflow: hidden;
}
.hero-box::before {
    content: "";
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 70% 30%, #3b82f622 0%, transparent 60%);
}
.hero-box h1 {
    color: #f1f5f9;
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    position: relative;
}
.hero-box p {
    color: #94a3b8;
    font-size: 0.95rem;
    margin: 0;
    position: relative;
}
.hero-badge {
    display: inline-block;
    background: #3b82f620;
    border: 1px solid #3b82f650;
    color: #60a5fa !important;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    margin-right: 0.5rem;
    position: relative;
}

/* ──── metric cards ──── */
div[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 3px #0000000a;
}
div[data-testid="stMetric"] label {
    color: #64748b !important;
    font-weight: 500;
    font-size: 0.82rem !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-weight: 700;
    font-size: 1.6rem !important;
}

/* ──── tabs ──── */
button[data-baseweb="tab"] {
    font-family: 'IBM Plex Sans Thai', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 1.5rem !important;
}

/* ──── attack result header ──── */
.attack-header {
    background: linear-gradient(90deg, #1e293b 0%, #334155 100%);
    border-radius: 10px;
    padding: 0.8rem 1.5rem;
    margin: 1.5rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.attack-header h3 {
    color: #f1f5f9;
    margin: 0;
    font-size: 1.2rem;
    font-weight: 600;
}
.attack-tag {
    background: #3b82f630;
    color: #60a5fa;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.15rem 0.6rem;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
}

/* ──── info box ──── */
.info-card {
    background: #f0f9ff;
    border-left: 4px solid #0ea5e9;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.3rem;
    margin: 0.5rem 0 1rem 0;
}
.info-card p { color: #0c4a6e; margin: 0; font-size: 0.9rem; }

/* ──── status pill ──── */
.pill-ok {
    display: inline-block;
    background: #dcfce7; color: #166534;
    font-size: 0.78rem; font-weight: 500;
    padding: 0.15rem 0.7rem; border-radius: 20px;
}
.pill-warn {
    display: inline-block;
    background: #fef9c3; color: #854d0e;
    font-size: 0.78rem; font-weight: 500;
    padding: 0.15rem 0.7rem; border-radius: 20px;
}

/* ──── separator line ──── */
.sep {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 1.8rem 0;
}

/* ──── dashboard score cards ──── */
.score-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}
.score-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 2px 8px #0000000a;
    transition: transform 0.15s;
}
.score-card:hover { transform: translateY(-2px); }
.score-card .atk-name {
    font-size: 0.82rem; color: #64748b;
    font-weight: 500; text-transform: uppercase;
    letter-spacing: 0.05em;
}
.score-card .atk-f1 {
    font-size: 2.2rem; font-weight: 700; color: #0f172a;
    margin: 0.3rem 0;
}
.score-card .atk-model {
    font-size: 0.72rem; color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
}

/* ──── download button ──── */
.stDownloadButton > button {
    background: #0f172a !important;
    color: #f1f5f9 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
}
.stDownloadButton > button:hover {
    background: #1e293b !important;
}

/* ──── run button ──── */
.stButton > button[kind="primary"] {
    border-radius: 10px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Matplotlib style ───────────────────────────────
CHART_BG = "#fafbfc"
CHART_COLORS = {"Bot": "#ef4444", "WebAttack": "#f59e0b",
                "PortScan": "#3b82f6", "DDoS": "#10b981",
                "Web Attack": "#f59e0b"}

def styled_fig(w=8, h=4):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=CHART_BG)
    ax.set_facecolor(CHART_BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors="#64748b", labelsize=9)
    return fig, ax


@st.cache_data(show_spinner=False)
def _load(file_bytes, name):
    return core.load_dataframe(file_bytes, name)


# ─── Sidebar ────────────────────────────────────────
bundles = core.list_bundles()

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.5rem 0 0.5rem 0;">
        <div style="font-size:2.8rem;">🛡️</div>
        <div style="font-size:1.3rem; font-weight:700; letter-spacing:0.03em; margin-top:0.3rem;">
            NIDS Detector
        </div>
        <div style="font-size:0.75rem; color:#94a3b8; margin-top:0.2rem;">
            Network Intrusion Detection System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if bundles:
        st.markdown(f"""
        <div style="background:#166534; border-radius:8px; padding:0.6rem 1rem;
                    font-size:0.85rem; text-align:center; margin-bottom:0.8rem;">
            ✅ โมเดลพร้อมใช้: <strong>{len(bundles)}</strong> ตัว
        </div>
        """, unsafe_allow_html=True)
        for b in bundles:
            st.markdown(f"""
            <div style="background:#1e293b; border:1px solid #334155; border-radius:8px;
                        padding:0.4rem 0.8rem; margin:0.3rem 0; font-size:0.82rem;
                        display:flex; justify-content:space-between; align-items:center;">
                <span>🎯 {b}</span>
                <span style="color:#4ade80; font-size:0.7rem;">READY</span>
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
    <h1>🛡️ ตรวจจับการบุกรุกเครือข่าย</h1>
    <p>
        <span class="hero-badge">CIC-IDS2017</span>
        <span class="hero-badge">Hybrid v4</span>
        <span class="hero-badge">XGBoost + PyTorch NN</span>
        ระบบตรวจจับ 4 ประเภท attack — Bot · PortScan · DDoS · WebAttack
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
        <div style="text-align:center; padding:3rem 1rem; color:#94a3b8;">
            <div style="font-size:3rem; margin-bottom:0.8rem;">📁</div>
            <div style="font-size:1.1rem; font-weight:500; color:#64748b;">
                ลากไฟล์มาวาง หรือกด Browse files
            </div>
            <div style="font-size:0.85rem; margin-top:0.3rem;">
                รองรับ .xlsx, .xls, .csv · ไม่ต้องตัด feature ก่อน
            </div>
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
                    <h3>🎯 {atk}</h3>
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
                                 fontsize=11, fontweight="bold", color="#1e293b")
                    ax.set_ylabel("Reality", fontsize=9)
                    ax.set_xlabel("AI Prediction", fontsize=9)
                    fig.colorbar(im, fraction=0.046)
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
                        _ax.spines["left"].set_color("#cbd5e1")
                        _ax.spines["bottom"].set_color("#cbd5e1")
                        _ax.tick_params(colors="#64748b", labelsize=9)

                    bars = ax1.bar(["Benign", atk], [n_ben, n_atk],
                                   color=["#3b82f6", "#ef4444"], width=0.5,
                                   edgecolor="white", linewidth=1.5)
                    for bar, val in zip(bars, [n_ben, n_atk]):
                        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                f"{val:,}", ha="center", va="bottom", fontsize=10,
                                fontweight="bold", color="#1e293b")
                    ax1.set_title(f"Prediction Summary (thr={thr:.3f})",
                                  fontsize=10, fontweight="bold", color="#1e293b")
                    ax1.set_ylabel("จำนวน flow", fontsize=9)

                    ax2.hist(proba, bins=50, color="#3b82f6", edgecolor="white",
                             alpha=0.85, linewidth=0.5)
                    ax2.axvline(thr, color="#ef4444", linestyle="--", linewidth=2,
                               label=f"threshold={thr:.3f}")
                    ax2.set_title("Probability Distribution",
                                  fontsize=10, fontweight="bold", color="#1e293b")
                    ax2.set_xlabel(f"P({atk})", fontsize=9)
                    ax2.set_ylabel("จำนวน flow", fontsize=9)
                    ax2.legend(fontsize=8)
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
        <div style="text-align:center; padding:3rem 1rem; color:#94a3b8;">
            <div style="font-size:3rem; margin-bottom:0.8rem;">🔬</div>
            <div style="font-size:1.1rem; font-weight:500; color:#64748b;">
                อัปโหลดไฟล์เพื่อเริ่มสำรวจ feature
            </div>
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
                             fontweight="bold", color="#1e293b")
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
                            fontweight="bold", color="#1e293b")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(versions, fontsize=11, fontweight="bold")
        ax.set_ylabel("F1-score", fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_title("F1-score Progression: v1 → v4", fontsize=13,
                      fontweight="bold", color="#0f172a", pad=12)
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
                ax.spines["left"].set_color("#cbd5e1")
                ax.spines["bottom"].set_color("#cbd5e1")

                model_xgb, meta_xgb = core.load_bundle(atk)
                feats = meta_xgb["features"]
                importances = model_xgb.feature_importances_
                imp_df = pd.DataFrame({"feature": feats, "importance": importances})
                imp_df = imp_df.sort_values("importance", ascending=True)
                c = CHART_COLORS.get(atk, "#3b82f6")
                ax.barh(imp_df["feature"], imp_df["importance"], color=c,
                        edgecolor="white", linewidth=0.5, height=0.7)
                ax.set_title(f"Feature Importance: {atk}", fontsize=11,
                             fontweight="bold", color="#0f172a")
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
