"""Pemuatan model, ambang batas, dan inferensi."""
import json
import os
import streamlit as st
import torch
import warnings
from config import PTBXLConfig
from constants import CLASSES
from model import build_model


@st.cache_resource(show_spinner=False)
def load_model_and_config(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    run_args = ckpt.get("config", {}) or {}

    ptb_config = PTBXLConfig(sampling_rate=run_args.get("sampling_rate", 100))
    if run_args.get("normalize_mode") is not None:
        ptb_config.normalize_mode = run_args["normalize_mode"]
    if run_args.get("use_lead_dropout") is not None:
        ptb_config.use_lead_dropout = run_args["use_lead_dropout"]
    if run_args.get("lead_dropout_prob") is not None:
        ptb_config.lead_dropout_prob = run_args["lead_dropout_prob"]

    model = build_model(
        variant=run_args.get("model_variant", "tiny"),
        in_channels=ptb_config.n_leads,
        num_classes=len(CLASSES),
        drop_path_rate=run_args.get("drop_path_rate", 0.1),
        se_reduction=run_args.get("se_reduction", 16),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ptb_config




def load_thresholds(path):
    default = {c: 0.5 for c in CLASSES}
    if not path or not os.path.exists(path):
        warnings.warn(f"File threshold tidak ditemukan ({path}), semua kelas pakai 0.5.")
        return default

    try:
        with open(path) as f:
            raw = json.load(f)
    except Exception as e:
        warnings.warn(f"Gagal baca file threshold ({e}), semua kelas pakai 0.5.")
        return default

    out = {}
    for c in CLASSES:
        v = raw.get(c, 0.5)
        # format nested dari tune_threshold.py
        if isinstance(v, dict):
            v = v.get("final_threshold", v.get("threshold", 0.5))
        try:
            out[c] = float(v)
        except (TypeError, ValueError):
            warnings.warn(f"Threshold kelas {c} tidak valid ({v!r}), pakai 0.5.")
            out[c] = 0.5
    return out


def run_inference(model, signal_12xN, device):
    x = torch.tensor(signal_12xN, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    return probs
