# -*- coding: utf-8 -*-
"""
nids_core.py — ฟังก์ชันกลางของระบบ NIDS (ไม่มี UI)
ใช้ร่วมกันโดย train_backend.py (หลังบ้าน) และ nids_web.py (เว็บผู้ใช้)

รวม: โหลดข้อมูล · pipeline เทรน v4 (XGBoost/NN + SMOTE + threshold tuning) ·
     bundle (model + scaler + threshold + features) · feature tool + RF baseline
"""

import os
import io
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score,
                             average_precision_score)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# torch เป็น optional (ฝั่ง NN). ถ้าไม่มี XGBoost ยังใช้ได้
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
except Exception:
    TORCH_OK = False

# ---------- ค่าคงที่ (ตรงกับ v4) ----------
EPOCHS_DEFAULT = 15
BATCH_SIZE = 512
LR = 0.001
WEIGHT_MODE = "sqrt"
SMOTE_MINORITY_THRESHOLD = 0.15
MODELS_DIR = "nids_models"

# ชุด feature มาตรฐาน (ใช้เช็คความเข้ากันได้ของไฟล์จากแหล่งอื่น)
FEATURE_SET_A = [  # Bot, PortScan (26)
    "Flow Duration", "Total Length of Fwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Max", "Bwd IAT Std",
    "Fwd PSH Flags", "Bwd Packets/s", "Max Packet Length", "Packet Length Mean",
    "Packet Length Variance", "FIN Flag Count", "RST Flag Count", "PSH Flag Count",
    "Down/Up Ratio", "Init_Win_bytes_forward", "Active Mean", "Active Std",
    "Active Max", "Active Min", "Idle Mean", "Idle Std"]
FEATURE_SET_B = [  # DDoS, WebAttack (24)
    "Destination Port", "Total Fwd Packets", "Total Length of Fwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Mean", "Bwd Packet Length Max",
    "Bwd Packet Length Mean", "Fwd IAT Mean", "Fwd IAT Std", "Fwd PSH Flags",
    "Max Packet Length", "Packet Length Mean", "Packet Length Variance",
    "FIN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "min_seg_size_forward", "Active Mean", "Active Std", "Active Max",
    "Idle Mean", "Idle Std", "Idle Min"]
CANONICAL_SETS = {"Set A (Bot/PortScan, 26)": FEATURE_SET_A,
                  "Set B (DDoS/WebAttack, 24)": FEATURE_SET_B}

# =====================================================
# Column alias mapping: ชื่อคอลัมน์จากแหล่งอื่น → ชื่อมาตรฐาน 2017
# รองรับ CIC-IDS 2018 และ dataset อื่นที่ใช้ชื่อต่างออกไป
# =====================================================
COLUMN_ALIASES = {
    # CIC-IDS 2018 → 2017
    "TotLen Fwd Pkts":   "Total Length of Fwd Packets",
    "Fwd Pkt Len Max":   "Fwd Packet Length Max",
    "Fwd Pkt Len Mean":  "Fwd Packet Length Mean",
    "Fwd Pkt Len Std":   "Fwd Packet Length Std",
    "Bwd Pkt Len Std":   "Bwd Packet Length Std",
    "Bwd Pkt Len Max":   "Bwd Packet Length Max",
    "Bwd Pkt Len Mean":  "Bwd Packet Length Mean",
    "Flow Byts/s":       "Flow Bytes/s",
    "Flow Pkts/s":       "Flow Packets/s",
    "Bwd Pkts/s":        "Bwd Packets/s",
    "Fwd Pkts/s":        "Fwd Packets/s",
    "Pkt Len Max":       "Max Packet Length",
    "Pkt Len Mean":      "Packet Length Mean",
    "Pkt Len Var":       "Packet Length Variance",
    "FIN Flag Cnt":      "FIN Flag Count",
    "RST Flag Cnt":      "RST Flag Count",
    "PSH Flag Cnt":      "PSH Flag Count",
    "ACK Flag Cnt":      "ACK Flag Count",
    "Init Fwd Win Byts": "Init_Win_bytes_forward",
    "Init Bwd Win Byts": "Init_Win_bytes_backward",
    "Dst Port":          "Destination Port",
    "Tot Fwd Pkts":      "Total Fwd Packets",
    "TotLen Bwd Pkts":   "Total Length of Bwd Packets",
    "Fwd IAT Mean":      "Fwd IAT Mean",
    "Fwd IAT Std":       "Fwd IAT Std",
    "Bwd IAT Std":       "Bwd IAT Std",
}


def auto_map_columns(df):
    """ตรวจสอบและ rename คอลัมน์อัตโนมัติ ให้ตรงกับชื่อมาตรฐาน (2017)
    Returns: (df_renamed, renamed_dict)
        - df_renamed: DataFrame ที่ rename แล้ว
        - renamed_dict: {old_name: new_name} เฉพาะคอลัมน์ที่เปลี่ยน
    """
    rename_map = {}
    for col in df.columns:
        col_stripped = col.strip()
        if col_stripped in COLUMN_ALIASES:
            new_name = COLUMN_ALIASES[col_stripped]
            # rename เฉพาะเมื่อชื่อใหม่ยังไม่มีอยู่แล้ว
            if new_name not in df.columns:
                rename_map[col] = new_name
    if rename_map:
        df = df.rename(columns=rename_map)
    return df, rename_map


def auto_trim_features(df, attack_name=None):
    """ตัด feature ให้เหลือเฉพาะที่โมเดลต้องการ
    1. auto-rename ก่อน
    2. ถ้าระบุ attack_name → ใช้ feature list จาก bundle (meta)
       ถ้าไม่ระบุ → ใช้ canonical set ที่ match มากที่สุด
    3. คืน DataFrame ที่ตัดแล้ว + สรุปสิ่งที่ทำ

    Returns: (df_trimmed, summary_dict)
    """
    df, renamed = auto_map_columns(df)

    label_col = detect_label_column(df)
    feats = [c for c in df.columns if c != label_col]

    # หา target feature set
    if attack_name:
        try:
            _, meta = load_bundle(attack_name)
            target_features = meta["features"]
            set_name = f"{attack_name} model"
        except Exception:
            target_features = None
            set_name = None
    else:
        target_features = None
        set_name = None

    if target_features is None:
        # auto-detect: เลือก canonical set ที่ overlap มากที่สุด
        best_name, best_overlap = None, -1
        for sname, sset in CANONICAL_SETS.items():
            ov = len(set(feats) & set(sset))
            if ov > best_overlap:
                best_name, best_overlap = sname, ov
        target_features = CANONICAL_SETS[best_name]
        set_name = best_name

    present = [f for f in target_features if f in df.columns]
    missing = [f for f in target_features if f not in df.columns]
    extra = [f for f in feats if f not in target_features]

    # ตัดเหลือเฉพาะที่ต้องการ + Label (ถ้ามี)
    keep_cols = present + ([label_col] if label_col else [])
    df_trimmed = df[keep_cols].copy()

    summary = {
        "renamed": renamed,
        "target_set": set_name,
        "target_count": len(target_features),
        "present": present,
        "missing": missing,
        "extra_removed": extra,
        "kept_count": len(present),
        "has_label": label_col is not None,
    }
    return df_trimmed, summary


# =====================================================
# โมเดล NN (เหมือน v4)
# =====================================================
if TORCH_OK:
    class TrafficClassifier(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 2))

        def forward(self, x):
            return self.model(x)
else:
    TrafficClassifier = None


# =====================================================
# โหลด/เตรียมข้อมูล
# =====================================================
def load_dataframe(src, name=None):
    """src = path (str) หรือ bytes. name ใช้เดานามสกุลตอนเป็น bytes"""
    if isinstance(src, (bytes, bytearray)):
        buf = io.BytesIO(src)
        nm = (name or "").lower()
    else:
        buf = src
        nm = str(src).lower()
    if nm.endswith((".xlsx", ".xls")):
        df = pd.read_excel(buf)
    elif nm.endswith(".csv"):
        df = pd.read_csv(buf)
    else:
        raise ValueError("รองรับ .xlsx / .xls / .csv")
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def detect_label_column(df):
    for c in df.columns:
        if c.strip().lower() == "label":
            return c
    return None


def has_label(df):
    return detect_label_column(df) is not None


def make_binary_target(df, label_col):
    """ทุกอย่างที่ไม่ใช่ BENIGN = attack(1). รองรับ prefix 'Web Attack - XSS'"""
    labels = df[label_col].astype(str).str.strip()
    benign = labels.str.upper() == "BENIGN"
    y = (~benign).astype(int).values
    non_benign = labels[~benign].unique().tolist()
    if not non_benign:
        name = "Attack"
    elif len(non_benign) == 1:
        name = non_benign[0]
    else:
        name = non_benign[0].split(" - ")[0].split("-")[0].strip() or "Attack"
    return y, name


# =====================================================
# feature tool
# =====================================================
def feature_stats(df, feature_cols):
    sub = df[list(feature_cols)]
    stats = pd.DataFrame({
        "feature": list(feature_cols),
        "variance": [float(sub[c].var()) for c in feature_cols],
        "n_unique": [int(sub[c].nunique()) for c in feature_cols],
    })
    stats["near_zero_var"] = stats["variance"] < 1e-9
    return stats


def correlated_pairs(df, feature_cols, thr=0.98, sample=20000):
    sub = df[list(feature_cols)]
    if len(sub) > sample:
        sub = sub.sample(sample, random_state=42)
    corr = sub.corr().abs()
    cols = list(feature_cols)
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if v >= thr:
                pairs.append((cols[i], cols[j], round(float(v), 4)))
    return sorted(pairs, key=lambda x: -x[2])


def auto_drop_suggestion(df, feature_cols, corr_thr=0.98, drop_nzv=True):
    stats = feature_stats(df, feature_cols)
    drop = set()
    if drop_nzv:
        drop |= set(stats.loc[stats["near_zero_var"], "feature"])
    kept, corr_drop = set(), []
    for a, b, _ in correlated_pairs(df, feature_cols, corr_thr):
        if a not in corr_drop:
            kept.add(a)
            if b not in kept and b not in corr_drop:
                corr_drop.append(b)
    drop |= set(corr_drop)
    return sorted(drop)


def run_random_forest(df, feature_cols, y, subsample_n=None, n_estimators=200):
    X = df[list(feature_cols)].values
    if subsample_n and subsample_n < len(X):
        idx = np.random.RandomState(42).choice(len(X), subsample_n, replace=False)
        X, y = X[idx], y[idx]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1,
                                random_state=42, class_weight="balanced")
    rf.fit(X_tr, y_tr)
    proba = rf.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_te, pred, average="binary", zero_division=0)
    return dict(auc=float(roc_auc_score(y_te, proba)),
                ap=float(average_precision_score(y_te, proba)),
                precision=float(p), recall=float(r), f1=float(f1),
                importances=rf.feature_importances_,
                features=list(feature_cols), y_te=y_te, proba=proba)


def compatibility_check(feature_cols):
    best_name, best_overlap = None, -1
    for sname, sset in CANONICAL_SETS.items():
        ov = len(set(feature_cols) & set(sset))
        if ov > best_overlap:
            best_name, best_overlap = sname, ov
    ref = CANONICAL_SETS[best_name]
    return dict(best_set=best_name, overlap=best_overlap, ref_size=len(ref),
                missing=[f for f in ref if f not in feature_cols],
                extra=[f for f in feature_cols if f not in ref])


# =====================================================
# Auto-select model type based on class balance
# =====================================================
IMBALANCE_THRESHOLD = 0.20   # minority < 20% → imbalanced → xgboost

def auto_select_model(y, threshold=IMBALANCE_THRESHOLD):
    """ดูสัดส่วน minority class แล้วเลือกโมเดลอัตโนมัติ
    - imbalanced (minority < threshold) → xgboost  (ดีกว่า NN บน tabular imbalanced)
    - balanced                          → nn       (NN ทำงานได้ดีเมื่อ class ใกล้เคียงกัน)
    """
    minority_ratio = min(y.mean(), 1 - y.mean())
    if minority_ratio < threshold:
        return "xgboost", minority_ratio
    else:
        return "nn", minority_ratio


# =====================================================
# class weight / เทรน NN + XGBoost (เหมือน v4)
# =====================================================
def compute_class_weights(n_benign, n_attack, mode="sqrt"):
    n_total = n_benign + n_attack
    wb, wa = n_total / n_benign, n_total / n_attack
    if mode == "sqrt":
        wb, wa = wb ** 0.5, wa ** 0.5
    elif mode == "none":
        wb, wa = 1.0, 1.0
    return torch.tensor([wb, wa], dtype=torch.float32)


def train_nn(X_train, y_train, input_dim, n_benign, n_attack, epochs, weight_mode, log=print):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.long).to(device)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=True)
    cw = compute_class_weights(n_benign, n_attack, mode=weight_mode).to(device)
    model = TrafficClassifier(input_dim).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    for epoch in range(epochs):
        model.train()
        tot = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            tot += loss.item()
        log(f"  Epoch {epoch+1:02d}/{epochs} | loss {tot/len(loader):.4f}")
    return model.cpu()


def predict_proba_nn(model, X):
    model.to("cpu").eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    out = []
    with torch.no_grad():
        loader = DataLoader(TensorDataset(X_t), batch_size=BATCH_SIZE, shuffle=False)
        for (xb,) in loader:
            out.extend(torch.softmax(model(xb), dim=1)[:, 1].numpy())
    return np.array(out)


def train_xgboost(X_train, y_train):
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                            eval_metric="aucpr", tree_method="hist",
                            n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    return clf


def find_best_threshold(y_true, y_prob):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    i = int(np.argmax(f1))
    return float(thr[i] if i < len(thr) else 0.999)


# =====================================================
# bundle: เซฟ/โหลด/ทำนาย
# =====================================================
def save_bundle(attack, model, model_type, scaler, threshold, features,
                input_dim, metrics, out_dir=MODELS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    meta = dict(attack_name=attack, model_type=model_type, scaler=scaler,
                threshold=float(threshold), features=list(features),
                input_dim=int(input_dim) if input_dim else None, metrics=metrics)
    joblib.dump(meta, os.path.join(out_dir, f"{attack}_meta.pkl"))
    if model_type == "xgboost":
        model.save_model(os.path.join(out_dir, f"{attack}_model.json"))
    else:
        torch.save(model.state_dict(), os.path.join(out_dir, f"{attack}_model.pth"))


def list_bundles(out_dir=MODELS_DIR):
    if not os.path.isdir(out_dir):
        return []
    return sorted(f[:-9] for f in os.listdir(out_dir) if f.endswith("_meta.pkl"))


def load_bundle(attack, out_dir=MODELS_DIR):
    meta = joblib.load(os.path.join(out_dir, f"{attack}_meta.pkl"))
    if meta["model_type"] == "xgboost":
        m = xgb.XGBClassifier()
        m.load_model(os.path.join(out_dir, f"{attack}_model.json"))
    else:
        m = TrafficClassifier(meta["input_dim"])
        m.load_state_dict(torch.load(
            os.path.join(out_dir, f"{attack}_model.pth"), map_location="cpu"))
        m.eval()
    return m, meta


def predict_with_bundle(model, meta, df):
    feats = meta["features"]
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise KeyError(missing)
    Xs = meta["scaler"].transform(df[feats].values)
    if meta["model_type"] == "xgboost":
        proba = model.predict_proba(Xs)[:, 1]
    else:
        proba = predict_proba_nn(model, Xs)
    return proba, (proba >= meta["threshold"]).astype(int)


# =====================================================
# pipeline เทรนครบ 1 attack
# =====================================================
# v5: Train 60% / Val 20% / Test 20%
#     - Train  → เทรนโมเดล
#     - Val    → tune threshold (หา F1 สูงสุด)
#     - Test   → วัดผลสุดท้าย (ไม่แตะจนจบ = ไม่มี bias)
# =====================================================
def train_one_attack(df, label_col, model_type, epochs=EPOCHS_DEFAULT,
                     use_smote_override=None, log=print):
    y, attack = make_binary_target(df, label_col)
    features = [c for c in df.columns if c != label_col]
    X = df[features].values

    # split 1: แยก test 20% ออกก่อน (ไม่แตะอีกเลยจนวัดผลสุดท้าย)
    X_dev, X_te, y_dev, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # split 2: แยก val 25% ของ dev (= 20% ของทั้งหมด) ที่เหลือเป็น train 60%
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, random_state=42, stratify=y_dev)

    log(f"  Split: Train {len(y_tr):,} (60%) / Val {len(y_val):,} (20%) / Test {len(y_te):,} (20%)")

    scaler = MinMaxScaler()
    X_tr  = scaler.fit_transform(X_tr)      # fit เฉพาะ train
    X_val = scaler.transform(X_val)
    X_te  = scaler.transform(X_te)

    minority = min((y_tr == 0).mean(), (y_tr == 1).mean())
    use_smote = (minority < SMOTE_MINORITY_THRESHOLD) if use_smote_override is None else use_smote_override
    weight_mode = WEIGHT_MODE
    if use_smote:
        log(f"[SMOTE] minority {minority:.2%} -> oversample train")
        X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
        weight_mode = "none"

    if model_type == "nn":
        model = train_nn(X_tr, y_tr, X_tr.shape[1],
                         int((y_tr == 0).sum()), int((y_tr == 1).sum()),
                         epochs, weight_mode, log=log)
        y_prob_val = predict_proba_nn(model, X_val)
        y_prob_te  = predict_proba_nn(model, X_te)
        input_dim = X_tr.shape[1]
    else:
        model = train_xgboost(X_tr, y_tr)
        y_prob_val = model.predict_proba(X_val)[:, 1]
        y_prob_te  = model.predict_proba(X_te)[:, 1]
        input_dim = None

    # tune threshold บน val (ไม่ใช่ test อีกต่อไป)
    thr = find_best_threshold(y_val, y_prob_val)
    log(f"  Threshold tuned on Val set: {thr:.4f}")

    # วัดผลสุดท้ายบน test (clean evaluation)
    y_pred = (y_prob_te >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_te, y_prob_te)
    except Exception:
        auc = float("nan")
    metrics = dict(precision=float(p), recall=float(r), f1=float(f1), auc=float(auc),
                   threshold=thr, n_train=int(len(y_tr)),
                   n_val=int(len(y_val)), n_test=int(len(y_te)))
    return dict(attack=attack, model=model, model_type=model_type, scaler=scaler,
                threshold=thr, features=features, input_dim=input_dim,
                metrics=metrics, cm=confusion_matrix(y_te, y_pred),
                y_te=y_te, y_prob=y_prob_te)
