# app.py
# LSTM Autoencoder Anomaly Detector — Streamlit UI
# ------------------------------------------------
# Tabs:
#  1) Data: upload & preview (XLSX for train, CSV for live), feature selection
#  2) Train: hyperparams, training progress, threshold from training errors
#  3) Detect: run on live CSV, results table, plots, CSV export
#  4) Settings/Export: threshold policy, save/load artifacts (model/scaler/config)
#
# Notes:
# - Mirrors your Colab logic: auto Top-K numeric selection, MinMaxScaler, windowing,
#   LSTMAE (encoder → latent → decoder), reconstruction MSE threshold = mu + K*sigma.
# - Uses Plotly for interactive charts.

import io
import os
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# ------------------------- Page config -------------------------
st.set_page_config(page_title="LSTM AE — Anomaly Detector", layout="wide")
st.markdown(
    """
    <style>
      .small-note { color:#6b7280; font-size:0.9rem; }
      .ok { color: #16a34a; }
      .warn { color: #b45309; }
      .err { color: #dc2626; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------- Session State Helpers -------------------------
def ss_get(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

# Initialize essential session slots
ss_get("train_df", None)
ss_get("live_df", None)
ss_get("selected_cols", None)         # list of column names in use
ss_get("feature_mode", "Auto Top-K")  # "Auto Top-K" | "Manual" | "Profile"
ss_get("feature_profile", None)       # {"columns":[...]}
ss_get("scaler", None)
ss_get("model", None)
ss_get("config", {
    "window": 60,
    "hidden": 64,
    "latent": 32,
    "epochs": 15,
    "batch": 64,
    "lr": 1e-3,
    "threshold_policy": "Mu+K*Sigma",
    "thresh_K": 2.0,
    "percentile": 97.5,
    "iqr_alpha": 1.5
})
ss_get("train_err_stats", None)       # dict with mu, sigma, thr, errors
ss_get("live_errors", None)           # dict with err_lv, thr, is_anom, end_idx, out_df

# ------------------------- Data Utilities -------------------------
def _numeric_quality(df0: pd.DataFrame):
    df = df0.apply(pd.to_numeric, errors='coerce')
    nnr = 1.0 - df.isna().mean(axis=0)             # non-null ratio
    var = df.var(axis=0, ddof=0).fillna(0)
    score = nnr + (var > 0).astype(float)          # prefer columns with values & variance
    return df, score.sort_values(ascending=False)

def _select_columns(df_raw: pd.DataFrame, feat_cols=None, top_k=None):
    if feat_cols is not None:
        # feat_cols is list of column names
        df = df_raw[feat_cols].copy()
        df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=0, how='any')
        df = df.loc[:, df.var(axis=0, ddof=0) > 0]
        if df.shape[1] == 0:
            raise ValueError("Selected columns have no usable numeric data.")
        return df
    df_num, score = _numeric_quality(df_raw)
    keep = list(score.index[: (top_k or df_num.shape[1])])
    df = df_num[keep].dropna(axis=0, how='any')
    df = df.loc[:, df.var(axis=0, ddof=0) > 0]
    if df.shape[1] == 0:
        raise ValueError("No usable numeric columns after auto-selection.")
    return df

def read_training_xlsx(file, sheet=0, skip_top=0):
    # more tolerant header handling: if skip_top>0 => header=None; else infer header
    header = None if skip_top > 0 else 0
    df = pd.read_excel(file, sheet_name=sheet, header=header, skiprows=skip_top)
    df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
    # if no header, assign generic col names
    if header is None:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
    return df

def read_live_csv(file, skip_top=0):
    header = "infer" if skip_top == 0 else None
    df = pd.read_csv(file, header=header, skiprows=skip_top)
    df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
    if header is None:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
    return df

def make_windows(arr: np.ndarray, win: int):
    if arr.shape[0] < win:
        raise ValueError(f"Dataset too short for WINDOW={win}. Got length={arr.shape[0]}")
    X = []
    for i in range(len(arr) - win + 1):
        X.append(arr[i:i+win])
    return np.stack(X)

# ------------------------- Model -------------------------
class LSTMAE(nn.Module):
    def __init__(self, input_size: int, hidden=64, latent=32):
        super().__init__()
        self.enc = nn.LSTM(input_size, hidden, batch_first=True)
        self.to_z = nn.Linear(hidden, latent)
        self.dec = nn.LSTM(latent, hidden, batch_first=True)
        self.out = nn.Linear(hidden, input_size)

    def forward(self, x):
        _, (h, _) = self.enc(x)
        z = self.to_z(h[-1])
        z_seq = z.unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.dec(z_seq)
        return self.out(dec_out)

def train_model(model, loader, epochs=15, lr=1e-3, device="cpu", progress_cb=None):
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()
    n = len(loader.dataset)
    for ep in range(1, epochs + 1):
        total = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            yb = model(xb)
            loss = crit(yb, xb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        if progress_cb:
            progress_cb(ep, epochs, total / n)
    return model

# ------------------------- Threshold Policies -------------------------
def threshold_from_policy(train_errs: np.ndarray, policy: str, cfg):
    mu = float(train_errs.mean())
    sigma = float(train_errs.std())
    thr = None
    if policy == "Mu+K*Sigma":
        thr = mu + float(cfg["thresh_K"]) * sigma
    elif policy == "Percentile":
        thr = float(np.percentile(train_errs, float(cfg["percentile"])))
    elif policy == "IQR":
        q1 = np.percentile(train_errs, 25)
        q3 = np.percentile(train_errs, 75)
        iqr = q3 - q1
        thr = float(q3 + float(cfg["iqr_alpha"]) * iqr)
    elif policy == "Fixed":
        thr = float(cfg.get("fixed_threshold", mu + 2.0 * sigma))
    else:
        thr = mu + 2.0 * sigma
    return {"mu": mu, "sigma": sigma, "thr": thr}

# ------------------------- Plots -------------------------
def plot_error_with_threshold(errors: np.ndarray, thr: float, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=errors, mode="lines", name="Reconstruction MSE"))
    fig.add_hline(y=thr, line_dash="dash", annotation_text=f"threshold={thr:.4g}", annotation_position="top left")
    fig.update_layout(title=title, xaxis_title="Window Index", yaxis_title="MSE")
    return fig

def plot_error_histogram(train_errs: np.ndarray, thr: float, title: str):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=train_errs, nbinsx=50, name="Train MSE", opacity=0.8))
    fig.add_vline(x=thr, line_dash="dash", annotation_text=f"threshold={thr:.4g}", annotation_position="top")
    fig.update_layout(title=title, xaxis_title="MSE", yaxis_title="Count", barmode="overlay")
    return fig

# ------------------------- Tab 1: Data -------------------------
def tab_data():
    st.header("1) Data — Upload & Feature Selection")
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Training Excel file (.xlsx)")
        xlsx_file = st.file_uploader("Upload training Excel file", type=["xlsx"], key="xlsx_up")
        train_sheet = st.number_input("Excel sheet index", min_value=0, step=1, value=0, key="sheet_idx")
        skip_excel = st.number_input("Skip top rows (Excel)", min_value=0, step=1, value=0, key="skip_excel")
    with colB:
        st.subheader("Data to be checked (.csv)")
        csv_file = st.file_uploader("Upload data file", type=["csv"], key="csv_up")
        skip_csv = st.number_input("Skip top rows (CSV)", min_value=0, step=1, value=0, key="skip_csv")

    if xlsx_file is not None:
        try:
            df_tr_raw = read_training_xlsx(xlsx_file, sheet=train_sheet, skip_top=skip_excel)
            st.write("**Train preview**", df_tr_raw.head())
            st.caption(f"Train shape: {df_tr_raw.shape}")
        except Exception as e:
            st.error(f"Error reading training Excel: {e}")
            df_tr_raw = None
    else:
        df_tr_raw = None

    if csv_file is not None:
        try:
            df_lv_raw = read_live_csv(csv_file, skip_top=skip_csv)
            st.write("**Live preview**", df_lv_raw.head())
            st.caption(f"Live shape: {df_lv_raw.shape}")
        except Exception as e:
            st.error(f"Error reading live CSV: {e}")
            df_lv_raw = None
    else:
        df_lv_raw = None

    st.markdown("---")
    st.subheader("Feature Selection")
    feature_mode = st.radio("Mode", ["Auto Top-K", "Manual", "Profile"], horizontal=True, key="feat_mode_radio")
    st.session_state["feature_mode"] = feature_mode

    selected_cols = None
    if feature_mode == "Auto Top-K":
        k = st.slider("Top-K numeric columns", min_value=1, max_value=16, value=2, step=1, key="topk_auto")
        if df_tr_raw is not None:
            try:
                # auto from training; show selected names
                df_auto = _select_columns(df_tr_raw, feat_cols=None, top_k=k)
                selected_cols = list(df_auto.columns)
                st.success(f"Auto-selected columns from training: {selected_cols}")
            except Exception as e:
                st.error(f"Auto-selection failed: {e}")
    elif feature_mode == "Manual":
        # Show union of available columns (train or live) so user can pick
        cols_union = []
        if df_tr_raw is not None:
            cols_union = list(df_tr_raw.columns)
        if df_lv_raw is not None:
            cols_union = sorted(list(set(cols_union) | set(df_lv_raw.columns)))
        picked = st.multiselect("Choose feature columns by name", options=cols_union, default=cols_union[:2] if cols_union else [], key="manual_cols")
        if picked:
            selected_cols = list(picked)
            st.info(f"Selected: {selected_cols}")
    else:  # Profile
        profile_json = st.text_area("Paste feature profile JSON", value=st.session_state["feature_profile"] and json.dumps(st.session_state["feature_profile"], indent=2) or "", key="profile_text")
        if st.button("Load profile", key="btn_load_profile"):
            try:
                prof = json.loads(profile_json)
                if not isinstance(prof, dict) or "columns" not in prof:
                    raise ValueError("Profile must be a dict with key 'columns'.")
                selected_cols = list(prof["columns"])
                st.session_state["feature_profile"] = prof
                st.success(f"Loaded profile. Columns = {selected_cols}")
            except Exception as e:
                st.error(f"Invalid profile: {e}")

    if st.button("Lock data & features", type="primary", disabled=(df_tr_raw is None or df_lv_raw is None or not selected_cols), key="btn_lock"):
        try:
            df_tr = _select_columns(df_tr_raw, feat_cols=selected_cols, top_k=None)
            # For live, map by name; drop extras; keep same order
            df_lv = df_lv_raw[selected_cols].copy()
            df_lv = df_lv.apply(pd.to_numeric, errors='coerce').dropna(axis=0, how='any')
            df_lv = df_lv.loc[:, df_lv.var(axis=0, ddof=0) > 0]
            # Must still contain all selected columns
            missing = [c for c in selected_cols if c not in df_lv.columns]
            if missing:
                raise ValueError(f"Live CSV missing selected columns: {missing}")

            st.session_state["train_df"] = df_tr
            st.session_state["live_df"] = df_lv
            st.session_state["selected_cols"] = selected_cols
            st.success(f"Locked. Train shape={df_tr.shape}, Live shape={df_lv.shape}. Columns={selected_cols}")
        except Exception as e:
            st.error(f"Failed to lock: {e}")

# ------------------------- Tab 2: Train -------------------------
def tab_train():
    st.header("2) Train — LSTM Autoencoder")
    cfg = st.session_state["config"]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        cfg["window"] = st.number_input("Window", min_value=8, max_value=2048, value=int(cfg["window"]), step=1, key="window_input")
    with c2:
        cfg["hidden"] = st.number_input("Hidden", min_value=8, max_value=1024, value=int(cfg["hidden"]), step=8, key="hidden_units")
    with c3:
        cfg["latent"] = st.number_input("Latent", min_value=4, max_value=512, value=int(cfg["latent"]), step=4, key="latent_units")
    with c4:
        cfg["epochs"] = st.number_input("Epochs", min_value=1, max_value=500, value=int(cfg["epochs"]), step=1, key="epochs_input")
    with c5:
        cfg["batch"] = st.number_input("Batch", min_value=1, max_value=4096, value=int(cfg["batch"]), step=1, key="batch_input")
    with c6:
        cfg["lr"] = float(st.text_input("LR", value=str(cfg["lr"]), key="lr_input"))

    st.session_state["config"] = cfg

    df_tr = st.session_state["train_df"]
    df_lv = st.session_state["live_df"]
    cols = st.session_state["selected_cols"]

    if df_tr is None or df_lv is None or not cols:
        st.warning("Please finish the Data tab (upload + lock) first.")
        return

    # Auto adjust window if needed
    min_len = min(len(df_tr), len(df_lv))
    win = int(cfg["window"])
    if min_len <= win:
        new_w = max(8, min_len - 1)
        st.warning(f"Reducing WINDOW from {win} to {new_w} due to short data.")
        win = new_w

    # Fit scaler on training
    scaler = MinMaxScaler()
    Xtr = scaler.fit_transform(df_tr.values)
    Xlv = scaler.transform(df_lv.values)

    # Make windows
    try:
        Xtr_w = make_windows(Xtr, win)
        Xlv_w = make_windows(Xlv, win)
    except Exception as e:
        st.error(f"Windowing error: {e}")
        return

    st.write(f"Train windows: {Xtr_w.shape}, Live windows: {Xlv_w.shape}")

    in_features = Xtr_w.shape[-1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"Using device: `{device}`")

    model = LSTMAE(input_size=in_features, hidden=int(cfg["hidden"]), latent=int(cfg["latent"]))

    tr_ds = TensorDataset(torch.tensor(Xtr_w, dtype=torch.float32))
    tr_dl = DataLoader(tr_ds, batch_size=int(cfg["batch"]), shuffle=True)

    prog = st.progress(0.0)
    loss_chart = st.empty()
    tracked = []

    def on_progress(ep, total, loss_val):
        tracked.append((ep, loss_val))
        prog.progress(ep / total)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[e for e, _ in tracked], y=[l for _, l in tracked], mode="lines+markers", name="train_loss"))
        fig.update_layout(title="Training loss (MSE)", xaxis_title="Epoch", yaxis_title="Loss")
        loss_chart.plotly_chart(fig, use_container_width=True)

    if st.button("Start Training", type="primary", key="btn_start_training"):
        model = train_model(model, tr_dl, epochs=int(cfg["epochs"]), lr=float(cfg["lr"]), device=device, progress_cb=on_progress)
        # Compute training errors for threshold
        model.eval()
        with torch.no_grad():
            Rtr = model(torch.tensor(Xtr_w, dtype=torch.float32).to(device)).cpu().numpy()
        err_tr = ((Rtr - Xtr_w) ** 2).mean(axis=(1, 2))

        # Threshold from selected policy (TRAIN TAB) — unique keys here
        st.subheader("Threshold Policy (for training errors)")
        policy = st.selectbox(
            "Policy",
            ["Mu+K*Sigma", "Percentile", "IQR", "Fixed"],
            index=["Mu+K*Sigma","Percentile","IQR","Fixed"].index(cfg["threshold_policy"]),
            key="policy_train",
        )
        if policy == "Mu+K*Sigma":
            cfg["thresh_K"] = st.slider("K (σ)", 0.5, 5.0, float(cfg["thresh_K"]), 0.1, key="thresh_k_train")
        elif policy == "Percentile":
            cfg["percentile"] = st.slider("Percentile", 90.0, 99.9, float(cfg["percentile"]), 0.1, key="percentile_train")
        elif policy == "IQR":
            cfg["iqr_alpha"] = st.slider("α (IQR multiplier)", 0.5, 5.0, float(cfg["iqr_alpha"]), 0.1, key="iqr_train")
        elif policy == "Fixed":
            default_mu2s = float(err_tr.mean() + 2.0 * err_tr.std())
            cfg["fixed_threshold"] = float(
                st.text_input("Fixed threshold (MSE)", value=f"{default_mu2s:.6g}", key="fixed_thr_train")
            )

        st.session_state["config"]["threshold_policy"] = policy

        stats = threshold_from_policy(err_tr, policy, cfg)
        mu, sigma, thr = stats["mu"], stats["sigma"], stats["thr"]

        st.success(f"Train errors → μ={mu:.6g}, σ={sigma:.6g}, threshold={thr:.6g}")
        st.plotly_chart(plot_error_histogram(err_tr, thr, "Training errors distribution"), use_container_width=True)

        # Save to session
        st.session_state["model"] = model
        st.session_state["scaler"] = scaler
        st.session_state["config"]["window"] = win
        st.session_state["train_err_stats"] = {"mu": mu, "sigma": sigma, "thr": thr, "errors": err_tr.tolist()}

# ------------------------- Tab 3: Detect -------------------------
def tab_detect():
    st.header("3) Detect — Live CSV Inference")

    df_lv = st.session_state["live_df"]
    df_tr = st.session_state["train_df"]
    cols = st.session_state["selected_cols"]
    model = st.session_state["model"]
    scaler = st.session_state["scaler"]
    cfg = st.session_state["config"]
    stats = st.session_state["train_err_stats"]

    if any(x is None for x in [df_lv, df_tr, cols, model, scaler, stats]):
        st.warning("Please complete Data and Train tabs first.")
        return

    win = int(cfg["window"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # scale and window live
    Xlv = scaler.transform(df_lv.values)
    try:
        Xlv_w = make_windows(Xlv, win)
    except Exception as e:
        st.error(f"Live windowing error: {e}")
        return

    model.eval()
    with torch.no_grad():
        Rlv = model(torch.tensor(Xlv_w, dtype=torch.float32).to(device)).cpu().numpy()
    err_lv = ((Rlv - Xlv_w) ** 2).mean(axis=(1, 2))

    # threshold (use current policy over training errors; recompute to reflect any latest UI changes)
    policy = cfg["threshold_policy"]
    thr_stats = threshold_from_policy(np.array(st.session_state["train_err_stats"]["errors"]), policy, cfg)
    thr = float(thr_stats["thr"])
    is_anom = err_lv > thr
    end_idx = np.arange(win - 1, win - 1 + len(err_lv))

    out = pd.DataFrame({"row_idx": end_idx, "recon_mse": err_lv, "is_anomaly": is_anom})
    for c in cols:
        out[c] = df_lv.iloc[end_idx][c].values

    st.session_state["live_errors"] = {
        "err_lv": err_lv.tolist(), "thr": float(thr),
        "is_anom": is_anom.tolist(), "end_idx": end_idx.tolist(),
        "out_df": out
    }

    st.subheader("Results")
    c1, c2 = st.columns([2,1])
    with c1:
        st.plotly_chart(plot_error_with_threshold(err_lv, thr, "Live reconstruction error vs threshold"), use_container_width=True)
    with c2:
        num_anom = int(is_anom.sum())
        st.metric("Anomalies detected", value=num_anom)
        st.caption(f"Threshold policy: {policy}")

    st.dataframe(out, use_container_width=True)

    # Download button
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button("Download anomaly_results.csv", data=csv_bytes, file_name="anomaly_results.csv", mime="text/csv", key="dl_results")

    # Optional: explore an anomaly window context
    st.markdown("### Inspect an anomaly window")
    anom_idxs = np.where(is_anom)[0]
    if len(anom_idxs) == 0:
        st.info("No anomalies to inspect at current threshold.")
        return
    k = int(st.slider("Choose anomaly # (by window index)", 0, len(anom_idxs)-1, 0, 1, key="inspect_anom_idx"))
    w_idx = int(anom_idxs[k])
    start = w_idx
    stop = w_idx + win
    fig = go.Figure()
    for c in cols:
        fig.add_trace(go.Scatter(x=list(range(start, stop)), y=df_lv[c].iloc[start:stop].values, mode="lines", name=c))
    fig.update_layout(title=f"Feature traces around anomaly window (rows {start}:{stop-1})", xaxis_title="Row", yaxis_title="Value")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------- Tab 4: Settings / Export -------------------------
def tab_settings_export():
    st.header("4) Settings / Export")

    cfg = st.session_state["config"]
    st.subheader("Threshold Defaults")
    policy = st.selectbox(
        "Default policy",
        ["Mu+K*Sigma", "Percentile", "IQR", "Fixed"],
        index=["Mu+K*Sigma","Percentile","IQR","Fixed"].index(cfg["threshold_policy"]),
        key="policy_settings",
    )
    cfg["threshold_policy"] = policy
    if policy == "Mu+K*Sigma":
        cfg["thresh_K"] = st.slider("K (σ)", 0.5, 5.0, float(cfg["thresh_K"]), 0.1, key="thresh_k_settings")
    elif policy == "Percentile":
        cfg["percentile"] = st.slider("Percentile", 90.0, 99.9, float(cfg["percentile"]), 0.1, key="percentile_settings")
    elif policy == "IQR":
        cfg["iqr_alpha"] = st.slider("α (IQR)", 0.5, 5.0, float(cfg["iqr_alpha"]), 0.1, key="iqr_settings")
    else:  # Fixed
        cfg["fixed_threshold"] = float(
            st.text_input("Fixed threshold (MSE)", value=str(cfg.get("fixed_threshold", 0.01)), key="fixed_thr_settings")
        )
    st.session_state["config"] = cfg

    # Save/Load artifacts
    st.subheader("Artifacts")
    model = st.session_state["model"]
    scaler = st.session_state["scaler"]
    selected_cols = st.session_state["selected_cols"]
    profile = {"columns": selected_cols} if selected_cols else None

    c1, c2, c3 = st.columns(3)
    with c1:
        if model is not None:
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            st.download_button("Download model.pt", data=buffer.getvalue(), file_name="model.pt", key="dl_model")
        else:
            st.caption("Model not trained yet.")
    with c2:
        if scaler is not None:
            st.download_button("Download scaler.pkl", data=pickle.dumps(scaler), file_name="scaler.pkl", key="dl_scaler")
        else:
            st.caption("Scaler not ready.")
    with c3:
        st.download_button("Download config.json", data=json.dumps(cfg, indent=2).encode("utf-8"), file_name="config.json", key="dl_config")

    st.subheader("Feature Profile")
    if profile:
        st.download_button("Download feature_profile.json", data=json.dumps(profile, indent=2).encode("utf-8"),
                           file_name="feature_profile.json", key="dl_profile")
    else:
        st.caption("No feature profile yet (lock data in Tab 1).")

    st.markdown("---")
    st.subheader("Load Existing Artifacts")
    up_model = st.file_uploader("Upload model.pt", type=["pt"], key="up_model")
    up_scaler = st.file_uploader("Upload scaler.pkl", type=["pkl"], key="up_scaler")
    up_config = st.file_uploader("Upload config.json", type=["json"], key="up_config")
    if st.button("Load artifacts", key="btn_load_artifacts"):
        try:
            if up_config:
                st.session_state["config"] = json.load(io.TextIOWrapper(up_config, encoding="utf-8"))
            if up_scaler:
                st.session_state["scaler"] = pickle.load(up_scaler)
            if up_model:
                # Need input_size to rebuild model correctly. Infer from current train/live if available.
                df_ref = st.session_state["train_df"] or st.session_state["live_df"]
                if df_ref is None:
                    st.warning("Cannot infer model input size (no data locked). Please lock data first, then load model.")
                else:
                    input_size = df_ref.shape[1]
                    cfg = st.session_state["config"]
                    model = LSTMAE(input_size=input_size, hidden=int(cfg["hidden"]), latent=int(cfg["latent"]))
                    model.load_state_dict(torch.load(up_model, map_location="cpu"))
                    st.session_state["model"] = model
            st.success("Artifacts loaded.")
        except Exception as e:
            st.error(f"Failed to load artifacts: {e}")

# ------------------------- Main -------------------------
def main():
    st.title("Water Wells LSTM Autoencoder — Time-Series Anomaly Detection")
    st.caption("Upload training file, pick features, train, and detect anomalies with a configurable threshold.")

    tabs = st.tabs(["Data", "Train", "Detect", "Settings/Export"])
    with tabs[0]:
        tab_data()
    with tabs[1]:
        tab_train()
    with tabs[2]:
        tab_detect()
    with tabs[3]:
        tab_settings_export()

if __name__ == "__main__":
    main()
