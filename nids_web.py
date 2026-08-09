# -*- coding: utf-8 -*-
"""
nids_web.py — เว็บสำหรับผู้ใช้ (2 ส่วนแยกอิสระ)
1) ตรวจจับ: โยนไฟล์ -> โมเดลหยิบ feature ที่ต้องใช้เอง -> ผลลัพธ์
2) Feature Tool: สำรวจ/ตัด feature + RF baseline + export (ไม่กระทบ Detect)
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


# =====================================================
# Sidebar
# =====================================================
with st.sidebar:
    st.header("🛡️ NIDS Detector")
    bundles = core.list_bundles()
    if bundles:
        st.success(f"โมเดลพร้อมใช้: {len(bundles)}")
        st.caption(" · ".join(bundles))
    else:
        st.error("ยังไม่มีโมเดล — รัน train_backend.py ก่อน")
    st.markdown("---")
    st.caption("การเทรนอยู่หลังบ้าน ผู้ใช้แค่โยนไฟล์เข้ามาตรวจ")

st.title("ตรวจจับการบุกรุกเครือข่าย (NIDS)")
st.caption("CIC-IDS2017 · โมเดล hybrid v4 (XGBoost + PyTorch NN)")

tab_detect, tab_feat, tab_dash = st.tabs(
    ["🔎 ตรวจจับ (Detect)", "🧹 Feature Tool", "📊 Dashboard"])

# =====================================================
# TAB 1 : DETECT (ส่วนหลัก — อิสระจาก Feature Tool)
# =====================================================
with tab_detect:
    st.subheader("ตรวจจับ attack จากไฟล์ traffic")
    st.caption("อัปโหลดไฟล์ → โมเดลจะเลือก feature ที่มันต้องใช้เองจากไฟล์ต้นฉบับ "
               "ไม่ต้องตัด feature ก่อน")

    up_detect = st.file_uploader("อัปโหลดไฟล์ traffic (.xlsx / .csv)",
                                  type=["xlsx", "xls", "csv"], key="detect_up")

    if up_detect is None:
        st.info("👆 อัปโหลดไฟล์ traffic เพื่อเริ่มตรวจจับ")
    elif not bundles:
        st.error("ยังไม่มีโมเดล — รัน train_backend.py ก่อน")
    else:
        with st.spinner("กำลังโหลดข้อมูล..."):
            df_det = _load(up_detect.getvalue(), up_detect.name)

        label_col_det = core.detect_label_column(df_det)
        has_lbl_det = label_col_det is not None
        feats_det = [c for c in df_det.columns if c != label_col_det]

        # --- สรุปข้อมูลแบบ v4 ---
        if has_lbl_det:
            y_det, atk_name_det = core.make_binary_target(df_det, label_col_det)
            n_benign = int((y_det == 0).sum())
            n_attack = int((y_det == 1).sum())
            summary_text = (
                f"{'='*50}\n"
                f"  โหลดข้อมูล: {atk_name_det}\n"
                f"{'='*50}\n"
                f"  BENIGN  : {n_benign:,}\n"
                f"  {atk_name_det:<10}: {n_attack:,}\n"
                f"  รวม     : {len(df_det):,} แถว, {len(feats_det)} features"
            )
        else:
            summary_text = (
                f"{'='*50}\n"
                f"  โหลดข้อมูล\n"
                f"{'='*50}\n"
                f"  รวม     : {len(df_det):,} แถว, {len(feats_det)} features\n"
                f"  (ไม่มีคอลัมน์ Label — ทำนายอย่างเดียว)"
            )
        st.code(summary_text, language=None)

        # เช็คว่าแต่ละโมเดลใช้ feature อะไร และไฟล์มีครบไหม
        with st.expander("ดู feature ที่แต่ละโมเดลต้องการ vs ไฟล์ที่อัป"):
            for atk in bundles:
                meta = core.load_bundle(atk)[1]
                needed = meta["features"]
                missing = [f for f in needed if f not in df_det.columns]
                if missing:
                    st.warning(f"**{atk}** — ขาด {len(missing)} feature: {', '.join(missing)}")
                else:
                    st.success(f"**{atk}** — feature ครบ ✅ ({len(needed)} ตัว)")

        chosen = st.multiselect("เลือกโมเดล", bundles, default=bundles, key="det_models")

        if st.button("▶️ ตรวจจับ", type="primary") and chosen:
            from sklearn.metrics import classification_report, confusion_matrix

            results = df_det.copy()

            for atk in chosen:
                try:
                    model, meta = core.load_bundle(atk)
                    proba, pred = core.predict_with_bundle(model, meta, df_det)
                except KeyError as e:
                    missing_list = e.args[0] if isinstance(e.args[0], list) else [str(e)]
                    st.warning(f"⚠️ **{atk}**: ไฟล์ขาด feature ที่โมเดลต้องใช้: "
                               f"{', '.join(str(x) for x in missing_list)}")
                    continue
                except Exception as e:
                    st.error(f"{atk}: {e}")
                    continue

                results[f"proba_{atk}"] = np.round(proba, 4)
                results[f"pred_{atk}"] = pred

                st.markdown(f"## {atk}  `[{meta['model_type'].upper()}]`")

                if has_lbl_det:
                    yt, _ = core.make_binary_target(df_det, label_col_det)

                    # --- threshold = 0.5 ---
                    pred_05 = (proba >= 0.5).astype(int)
                    report_05 = classification_report(
                        yt, pred_05, target_names=["Benign", atk], zero_division=0)
                    st.markdown(f"#### Threshold = 0.5 (ค่าเริ่มต้น)")
                    st.code(report_05, language=None)

                    # --- tuned threshold ---
                    thr = meta["threshold"]
                    report_tuned = classification_report(
                        yt, pred, target_names=["Benign", atk], zero_division=0)
                    st.markdown(f"#### Threshold = {thr:.3f} (tuned เพื่อ F1 สูงสุด)")
                    st.code(report_tuned, language=None)

                    # --- accuracy ---
                    acc_05 = (yt == pred_05).mean() * 100
                    acc_tuned = (yt == pred).mean() * 100
                    ac1, ac2 = st.columns(2)
                    ac1.metric("Accuracy (threshold=0.5)", f"{acc_05:.2f}%")
                    ac2.metric("Accuracy (tuned threshold)", f"{acc_tuned:.2f}%")

                    # --- confusion matrix heatmap (matplotlib ล้วน) ---
                    cm = confusion_matrix(yt, pred)
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    im = ax.imshow(cm, cmap='Blues')
                    for (i, j), v in np.ndenumerate(cm):
                        ax.text(j, i, f"{v:,}", ha="center", va="center",
                                color="white" if v > cm.max()/2 else "black", fontsize=14)
                    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
                    ax.set_xticklabels([f'Pred Benign', f'Pred {atk}'])
                    ax.set_yticklabels([f'Actual Benign', f'Actual {atk}'])
                    ax.set_title(f'Confusion Matrix: Benign vs {atk} '
                                 f'[{meta["model_type"]}] (threshold={thr:.3f})')
                    ax.set_ylabel('Reality')
                    ax.set_xlabel('AI Prediction')
                    fig.colorbar(im, fraction=0.046)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                else:
                    # ไม่มี label — แสดงผลให้คล้ายแบบมี label มากที่สุด
                    thr = meta["threshold"]
                    n_total = len(pred)
                    n_attack = int(pred.sum())
                    n_benign = n_total - n_attack

                    pred_05 = (proba >= 0.5).astype(int)
                    n_attack_05 = int(pred_05.sum())
                    n_benign_05 = n_total - n_attack_05

                    # --- prediction report (threshold=0.5) ---
                    st.markdown(f"#### Threshold = 0.5 (ค่าเริ่มต้น)")
                    report_05 = (
                        f"{'':>14}{'predicted':>12}{'% of total':>12}\n"
                        f"{'':>14}{'-'*24}\n"
                        f"{'Benign':>14}{n_benign_05:>12,}{n_benign_05/n_total*100:>11.2f}%\n"
                        f"{atk:>14}{n_attack_05:>12,}{n_attack_05/n_total*100:>11.2f}%\n"
                        f"{'':>14}{'-'*24}\n"
                        f"{'Total':>14}{n_total:>12,}{'100.00%':>12}"
                    )
                    st.code(report_05, language=None)

                    # --- prediction report (tuned threshold) ---
                    st.markdown(f"#### Threshold = {thr:.3f} (tuned เพื่อ F1 สูงสุด)")
                    report_tuned = (
                        f"{'':>14}{'predicted':>12}{'% of total':>12}\n"
                        f"{'':>14}{'-'*24}\n"
                        f"{'Benign':>14}{n_benign:>12,}{n_benign/n_total*100:>11.2f}%\n"
                        f"{atk:>14}{n_attack:>12,}{n_attack/n_total*100:>11.2f}%\n"
                        f"{'':>14}{'-'*24}\n"
                        f"{'Total':>14}{n_total:>12,}{'100.00%':>12}"
                    )
                    st.code(report_tuned, language=None)

                    # --- สรุปจำนวน ---
                    ac1, ac2 = st.columns(2)
                    ac1.metric("Benign (ปกติ)", f"{n_benign:,}")
                    ac2.metric(f"flag เป็น {atk}", f"{n_attack:,}")

                    # --- prediction distribution chart ---
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

                    # bar chart: benign vs attack
                    bars = ax1.bar(["Benign", atk], [n_benign, n_attack],
                                   color=["#3b82f6", "#ef4444"])
                    for bar, val in zip(bars, [n_benign, n_attack]):
                        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                f"{val:,}", ha="center", va="bottom", fontsize=11)
                    ax1.set_title(f"Prediction Summary (thr={thr:.3f})")
                    ax1.set_ylabel("จำนวน flow")

                    # probability histogram
                    ax2.hist(proba, bins=50, color="#3b82f6", edgecolor="white", alpha=0.8)
                    ax2.axvline(thr, color="#ef4444", linestyle="--", linewidth=2,
                               label=f"threshold={thr:.3f}")
                    ax2.set_title("Probability Distribution")
                    ax2.set_xlabel(f"P({atk})")
                    ax2.set_ylabel("จำนวน flow")
                    ax2.legend()

                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    st.caption("⚠️ ไม่มีคอลัมน์ Label — ประเมิน precision/recall/confusion matrix ไม่ได้ "
                               "แสดงเฉพาะผลการทำนาย")

                st.markdown("---")

            # ดาวน์โหลดผล
            st.download_button(
                "⬇️ ดาวน์โหลดผลตรวจจับ (.csv)",
                results.to_csv(index=False).encode("utf-8-sig"),
                file_name=up_detect.name.rsplit(".", 1)[0] + "_detected.csv",
                mime="text/csv")

# =====================================================
# TAB 2 : FEATURE TOOL (อิสระ — ไม่กระทบ Detect)
# =====================================================
with tab_feat:
    st.subheader("สำรวจ / ตัด Feature + RF Baseline")
    st.caption("เครื่องมือเตรียมข้อมูลแยกต่างหาก — ไม่กระทบแท็บตรวจจับ · "
               "ใช้สำหรับเตรียมไฟล์ก่อนเทรนใหม่ หรือวินิจฉัยไฟล์จากแหล่งอื่น")

    up_feat = st.file_uploader("อัปโหลดไฟล์สำหรับสำรวจ feature",
                                type=["xlsx", "xls", "csv"], key="feat_up")

    if up_feat is None:
        st.info("👆 อัปโหลดไฟล์เพื่อใช้ Feature Tool")
    else:
        with st.spinner("กำลังโหลดข้อมูล..."):
            df_ft = _load(up_feat.getvalue(), up_feat.name)

        label_col_ft = core.detect_label_column(df_ft)
        has_lbl_ft = label_col_ft is not None
        feats_ft = [c for c in df_ft.columns if c != label_col_ft]

        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("แถว", f"{len(df_ft):,}")
        fc2.metric("feature", len(feats_ft))
        fc3.metric("มี Label?", "มี ✅" if has_lbl_ft else "ไม่มี")

        # เช็คความเข้ากันได้
        comp = core.compatibility_check(feats_ft)
        with st.expander(f"เช็คความเข้ากันได้ — ใกล้ {comp['best_set']} "
                         f"({comp['overlap']}/{comp['ref_size']})"):
            if not comp["missing"] and not comp["extra"]:
                st.success("✅ feature ตรงชุดมาตรฐานพอดี")
            else:
                if comp["missing"]:
                    st.warning(f"ขาด {len(comp['missing'])}: {', '.join(comp['missing'])}")
                if comp["extra"]:
                    st.info(f"เกิน {len(comp['extra'])}: {', '.join(comp['extra'])}")

        # สถิติ
        st.markdown("#### สถิติราย feature")
        stats = core.feature_stats(df_ft, feats_ft)
        st.dataframe(stats, use_container_width=True, height=260)

        # ตัว auto-drop
        cc1, cc2 = st.columns(2)
        with cc1:
            drop_nzv = st.checkbox("ตัด feature ค่าคงที่ (variance≈0)", value=True,
                                    key="ft_nzv")
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

        # RF baseline (เฉพาะมี Label)
        if has_lbl_ft and selected:
            st.markdown("#### RandomForest baseline (วินิจฉัย)")
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
                    st.error("🔴 มีปัญหาที่ feature/data")
                imp = pd.DataFrame({"feature": res["features"],
                                    "importance": res["importances"]}).sort_values("importance")
                fig, ax = plt.subplots(figsize=(5, max(3, 0.25 * len(imp))))
                ax.barh(imp["feature"], imp["importance"], color="#2563eb")
                ax.set_title("Feature Importance (RF)")
                fig.tight_layout()
                st.pyplot(fig)

        # export
        if selected:
            st.markdown("#### Export ไฟล์ที่ตัดแล้ว")
            cols = selected + ([label_col_ft] if has_lbl_ft else [])
            st.download_button(
                "⬇️ ดาวน์โหลด .csv (เอาไปเทรนใหม่หรือใช้ต่อ)",
                df_ft[cols].to_csv(index=False).encode("utf-8-sig"),
                file_name=up_feat.name.rsplit(".", 1)[0] + "_selected.csv",
                mime="text/csv")

# =====================================================
# TAB 3 : DASHBOARD (ดูได้เลย ไม่ต้องอัปโหลด)
# =====================================================
with tab_dash:
    st.subheader("สรุปผลโมเดลทั้ง 4 ตัว + พัฒนาการ v1→v4")

    if not bundles:
        st.error("ยังไม่มีโมเดล — รัน train_backend.py ก่อน")
    else:
        # --- ตารางสรุป ---
        import joblib, os
        info_rows = []
        model_features = {}
        for atk in bundles:
            meta = joblib.load(os.path.join(core.MODELS_DIR, f"{atk}_meta.pkl"))
            m = meta.get("metrics", {})
            info_rows.append({
                "Attack Type": atk,
                "Model": meta["model_type"].upper(),
                "Threshold": f"{meta['threshold']:.3f}",
                "Precision": f"{m.get('precision', 0):.4f}",
                "Recall": f"{m.get('recall', 0):.4f}",
                "F1-score": f"{m.get('f1', 0):.4f}",
                "AUC": f"{m.get('auc', 0):.4f}",
                "Features": len(meta["features"]),
            })
            model_features[atk] = meta["features"]
        st.markdown("#### ผลลัพธ์ทุก Attack (tuned threshold, hybrid model)")
        summary_text = (
            f"{'Attack Type':<14}{'Model':<10}{'Threshold':>10}{'Precision':>11}"
            f"{'Recall':>9}{'F1':>9}{'AUC':>9}{'Feat':>6}\n"
            f"{'-'*78}\n"
        )
        for r in info_rows:
            summary_text += (
                f"{r['Attack Type']:<14}{r['Model']:<10}{r['Threshold']:>10}"
                f"{r['Precision']:>11}{r['Recall']:>9}{r['F1-score']:>9}"
                f"{r['AUC']:>9}{r['Features']:>6}\n"
            )
        st.code(summary_text, language=None)

        # --- กราฟ v1→v4 ---
        st.markdown("#### พัฒนาการ F1-score: v1 → v4")
        st.caption("แต่ละ version แก้ปัญหาอะไร: v1=baseline → v2=sqrt weight+threshold tuning "
                   "→ v3=+SMOTE → v4=hybrid XGBoost/NN")

        versions = ["v1", "v2", "v3", "v4"]
        f1_data = {
            "Bot":       [0.22, 0.73, 0.80, 0.97],
            "WebAttack": [0.57, 0.79, 0.80, 0.95],
            "PortScan":  [0.999, 0.999, 0.999, 0.999],
            "DDoS":      [0.999, 0.999, 0.999, 0.999],
        }
        colors = {"Bot": "#ef4444", "WebAttack": "#f59e0b",
                  "PortScan": "#3b82f6", "DDoS": "#10b981"}

        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(versions))
        width = 0.2
        for i, (atk, f1s) in enumerate(f1_data.items()):
            bars = ax.bar(x + i * width, f1s, width, label=atk, color=colors[atk])
            for bar, val in zip(bars, f1s):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(versions)
        ax.set_ylabel("F1-score")
        ax.set_ylim(0, 1.15)
        ax.set_title("F1-score Progression: v1 → v4")
        ax.legend(loc="lower right")
        ax.axhline(y=0.95, color="gray", linestyle=":", alpha=0.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # annotations
        st.markdown("""
| Version | การแก้ไข | ผลลัพธ์หลัก |
|---------|---------|------------|
| **v1** | NN + inverse class weight ตรงๆ | Bot P=13%, WebAttack P=40% (weight รุนแรงเกินทำให้ FP เยอะ) |
| **v2** | เปลี่ยนเป็น √weight + threshold tuning | Bot F1: 0.22→0.73, WebAttack F1: 0.57→0.79 |
| **v3** | +SMOTE oversample เฉพาะ train set | Bot F1: 0.73→0.80, WebAttack F1: 0.79→0.80 |
| **v4** | Hybrid: XGBoost สำหรับ imbalance, NN สำหรับ balanced | Bot F1: 0.80→**0.97**, WebAttack F1: 0.80→**0.95** |
""")

        # --- Feature Importance (เฉพาะ XGBoost) ---
        st.markdown("#### Feature Importance เปรียบเทียบข้าม Attack Type")
        st.caption("แสดงเฉพาะโมเดล XGBoost (Bot/WebAttack) — "
                   "NN (PortScan/DDoS) ไม่มี built-in feature importance")

        xgb_attacks = [atk for atk in bundles
                       if joblib.load(os.path.join(core.MODELS_DIR,
                          f"{atk}_meta.pkl"))["model_type"] == "xgboost"]

        if xgb_attacks:
            n_xgb = len(xgb_attacks)
            fig, axes = plt.subplots(1, n_xgb, figsize=(6 * n_xgb, max(5, 6)))
            if n_xgb == 1:
                axes = [axes]
            for ax, atk in zip(axes, xgb_attacks):
                model_xgb, meta_xgb = core.load_bundle(atk)
                feats = meta_xgb["features"]
                importances = model_xgb.feature_importances_
                imp_df = pd.DataFrame({"feature": feats, "importance": importances})
                imp_df = imp_df.sort_values("importance", ascending=True)
                ax.barh(imp_df["feature"], imp_df["importance"], color=colors.get(atk, "#3b82f6"))
                ax.set_title(f"Feature Importance: {atk}")
                ax.set_xlabel("Importance")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # top feature comparison
            st.markdown("##### Top 5 features ที่สำคัญที่สุดของแต่ละ attack")
            comp_text = ""
            for atk in xgb_attacks:
                model_xgb, meta_xgb = core.load_bundle(atk)
                feats = meta_xgb["features"]
                importances = model_xgb.feature_importances_
                top5 = sorted(zip(feats, importances), key=lambda x: -x[1])[:5]
                comp_text += f"\n  {atk}:\n"
                for rank, (f, v) in enumerate(top5, 1):
                    comp_text += f"    {rank}. {f:<35} {v:.4f}\n"
            st.code(comp_text, language=None)
            st.caption("⚠️ feature importance ขึ้นกับชนิด attack — "
                       "feature สำคัญของ Bot อาจไม่สำคัญกับ WebAttack "
                       "นี่คือเหตุผลที่ต้องแยกโมเดล 1 ตัวต่อ 1 attack")
        else:
            st.info("ไม่มีโมเดล XGBoost ใน bundle — ไม่สามารถแสดง feature importance ได้")
