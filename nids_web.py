# -*- coding: utf-8 -*-
"""
nids_web.py — เว็บสำหรับผู้ใช้ (โยนไฟล์ -> ตัด feature -> ตรวจจับ)
การเทรนอยู่หลังบ้าน (train_backend.py) เว็บนี้แค่โหลดโมเดลที่เทรนไว้มาใช้

รัน:
    pip install streamlit joblib scikit-learn imbalanced-learn xgboost pandas openpyxl matplotlib
    (ถ้าโมเดลมี nn ต้องมี torch ด้วย)
    python train_backend.py      # <- รันก่อนครั้งเดียว เพื่อสร้างโมเดล
    streamlit run nids_web.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nids_core as core

st.set_page_config(page_title="NIDS Detector", page_icon="🛡️", layout="wide")


@st.cache_data(show_spinner=False)
def _load(file_bytes, name):
    return core.load_dataframe(file_bytes, name)


# ---------------- Sidebar: อัปโหลด + สถานะโมเดล ----------------
with st.sidebar:
    st.header("🛡️ NIDS Detector")
    up = st.file_uploader("โยนไฟล์ traffic ที่นี่", type=["xlsx", "xls", "csv"])
    st.markdown("---")
    bundles = core.list_bundles()
    if bundles:
        st.success(f"โมเดลพร้อมใช้: {len(bundles)}")
        st.caption(" · ".join(bundles))
    else:
        st.error("ยังไม่มีโมเดล — รัน `python train_backend.py` ก่อน")
    st.caption("การเทรนอยู่หลังบ้าน ผู้ใช้แค่โยนไฟล์เข้ามาตรวจ")

st.title("ตรวจจับการบุกรุกเครือข่าย (NIDS)")
st.caption("โยนไฟล์ traffic → ตัด/ตรวจ feature → ตรวจจับ attack · CIC-IDS2017 · โมเดล hybrid v4")

if up is None:
    st.info("👈 อัปโหลดไฟล์ traffic (.xlsx / .csv) ที่แถบซ้ายเพื่อเริ่ม")
    st.stop()

df = _load(up.getvalue(), up.name)
label_col = core.detect_label_column(df)
has_lbl = label_col is not None
all_features = [c for c in df.columns if c != label_col]

# reset work_df เมื่อเปลี่ยนไฟล์
if st.session_state.get("_file") != up.name:
    st.session_state["_file"] = up.name
    st.session_state["selected"] = all_features

c1, c2, c3 = st.columns(3)
c1.metric("แถว", f"{len(df):,}")
c2.metric("feature", len(all_features))
c3.metric("มี Label?", "มี (ประเมินผลได้)" if has_lbl else "ไม่มี (ทำนายอย่างเดียว)")

tab_feat, tab_detect = st.tabs(["🧹 ตัด / ตรวจ Feature", "🔎 ตรวจจับ (Detect)"])

# =====================================================
# TAB 1 : feature tool
# =====================================================
with tab_feat:
    # เช็คความเข้ากันได้กับ feature set มาตรฐาน (ช่วยเคสไฟล์แหล่งอื่น)
    comp = core.compatibility_check(all_features)
    with st.expander(f"ความเข้ากันได้กับชุดมาตรฐาน — ใกล้ {comp['best_set']} "
                     f"({comp['overlap']}/{comp['ref_size']})", expanded=bool(comp["missing"])):
        if not comp["missing"] and not comp["extra"]:
            st.success("✅ feature ตรงชุดมาตรฐานพอดี")
        else:
            if comp["missing"]:
                st.warning(f"ขาด {len(comp['missing'])}: {', '.join(comp['missing'])}")
            if comp["extra"]:
                st.info(f"เกิน {len(comp['extra'])}: {', '.join(comp['extra'])}")

    st.subheader("สถิติราย feature")
    stats = core.feature_stats(df, all_features)
    st.dataframe(stats, use_container_width=True, height=260)

    cc1, cc2 = st.columns(2)
    with cc1:
        drop_nzv = st.checkbox("ตัด feature ค่าคงที่ (variance≈0)", value=True)
    with cc2:
        corr_thr = st.slider("ตัด feature ที่ correlation สูงเกิน", 0.90, 1.00, 0.98, 0.01)

    suggested = core.auto_drop_suggestion(df, all_features, corr_thr, drop_nzv)
    if suggested:
        st.caption(f"แนะนำให้ตัด ({len(suggested)}): {', '.join(suggested)}")

    default_keep = [f for f in all_features if f not in suggested]
    selected = st.multiselect("feature ที่จะเก็บไว้ (เอาออก = ตัดทิ้ง)",
                              options=all_features, default=default_keep)
    st.session_state["selected"] = selected
    st.write(f"เก็บไว้ **{len(selected)}** / {len(all_features)}")

    # RF baseline (เฉพาะเมื่อมี Label)
    if has_lbl:
        st.markdown("#### RandomForest baseline (วินิจฉัยคุณภาพ feature)")
        st.caption("ใช้ตรวจว่าไฟล์นี้แยก attack ออกจาก benign ได้ดีแค่ไหน "
                   "เหมาะกับไฟล์จากแหล่งอื่นที่สงสัยว่ามีปัญหา")
        rc1, rc2 = st.columns(2)
        with rc1:
            quick = st.checkbox("Quick (สุ่มตัวอย่างให้เร็ว)", value=len(df) > 80000)
        with rc2:
            n_est = st.slider("n_estimators", 50, 300, 150, 50)
        if st.button("▶️ รัน RF baseline") and selected:
            y, attack = core.make_binary_target(df, label_col)
            with st.spinner("กำลังเทรน RandomForest..."):
                res = core.run_random_forest(
                    df, selected, y,
                    subsample_n=min(60000, len(df)) if quick else None,
                    n_estimators=n_est)
            st.session_state["rf"] = res
        if "rf" in st.session_state:
            res = st.session_state["rf"]
            m1, m2, m3 = st.columns(3)
            m1.metric("ROC-AUC", f"{res['auc']:.4f}")
            m2.metric("PR-AUC", f"{res['ap']:.4f}")
            m3.metric("F1", f"{res['f1']:.3f}")
            auc = res["auc"]
            if auc >= 0.99:
                st.success("🟢 feature ดีมาก — ถ้าโมเดลจริงพลาด ปัญหาอยู่ที่ config/threshold")
            elif auc >= 0.95:
                st.info("🟡 feature ใช้ได้ดี")
            elif auc >= 0.85:
                st.warning("🟠 feature พอใช้ — ควรทบทวน")
            else:
                st.error("🔴 มีปัญหาที่ feature/data (label ผิด, feature หาย, ค่าเพี้ยน, leakage)")
            imp = pd.DataFrame({"feature": res["features"],
                                "importance": res["importances"]}).sort_values("importance")
            fig, ax = plt.subplots(figsize=(5, max(3, 0.25 * len(imp))))
            ax.barh(imp["feature"], imp["importance"], color="#2563eb")
            ax.set_title("Feature importance (RF)")
            fig.tight_layout()
            st.pyplot(fig)

    # export ไฟล์ที่ตัดแล้ว (ไว้เตรียมข้อมูล / ส่งเข้าหลังบ้าน)
    st.markdown("#### Export ไฟล์ที่ตัด feature แล้ว")
    if selected:
        cols = selected + ([label_col] if has_lbl else [])
        out_df = df[cols]
        st.download_button(
            "⬇️ ดาวน์โหลด .csv",
            out_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=up.name.rsplit(".", 1)[0] + "_selected.csv", mime="text/csv")

# =====================================================
# TAB 2 : detect
# =====================================================
with tab_detect:
    st.subheader("ตรวจจับด้วยโมเดลที่เทรนไว้")
    if not bundles:
        st.error("ยังไม่มีโมเดล — รัน `python train_backend.py` ก่อน")
        st.stop()

    work_cols = st.session_state.get("selected", all_features)
    work_df = df[work_cols + ([label_col] if has_lbl else [])]
    st.caption(f"ใช้ข้อมูลจากแท็บก่อนหน้า: {len(work_cols)} feature "
               f"(โมเดลจะเลือกเฉพาะ feature ที่มันต้องใช้เอง)")

    chosen = st.multiselect("เลือกโมเดล", bundles, default=bundles)
    if st.button("▶️ ตรวจจับ", type="primary") and chosen:
        results = df.copy()
        rows = []
        for atk in chosen:
            try:
                model, meta = core.load_bundle(atk)
                proba, pred = core.predict_with_bundle(model, meta, work_df)
            except KeyError as e:
                st.warning(f"⚠️ {atk}: ใช้ไม่ได้เพราะไฟล์/การตัด feature ทำให้ขาด: {e.args[0]}")
                continue
            except Exception as e:
                st.error(f"{atk}: {e}")
                continue
            results[f"proba_{atk}"] = np.round(proba, 4)
            results[f"pred_{atk}"] = pred
            row = {"model": atk, "type": meta["model_type"],
                   "thr": round(meta["threshold"], 3), "flagged": int(pred.sum())}
            if has_lbl:
                yt, _ = core.make_binary_target(df, label_col)
                from sklearn.metrics import precision_recall_fscore_support
                p, r, f1, _ = precision_recall_fscore_support(
                    yt, pred, average="binary", zero_division=0)
                row.update(precision=round(p, 3), recall=round(r, 3), f1=round(f1, 3))
            rows.append(row)
        st.session_state["detect_rows"] = rows
        st.session_state["detect_results"] = results

    if "detect_rows" in st.session_state and st.session_state["detect_rows"]:
        rows = st.session_state["detect_rows"]
        results = st.session_state["detect_results"]
        st.markdown("#### ผลตรวจจับ")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        total_flagged = int(sum(r["flagged"] for r in rows))
        st.metric("รวม flow ที่ถูก flag เป็น attack", f"{total_flagged:,} / {len(df):,}")

        pred_cols = [c for c in results.columns if c.startswith(("pred_", "proba_"))]
        show = ([label_col] if has_lbl else []) + pred_cols
        st.dataframe(results[show].head(30), use_container_width=True)
        st.download_button(
            "⬇️ ดาวน์โหลดผลตรวจจับ (.csv)",
            results.to_csv(index=False).encode("utf-8-sig"),
            file_name=up.name.rsplit(".", 1)[0] + "_detected.csv", mime="text/csv")