import os
import re
import json
import pickle
import numpy as np
import streamlit as st

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# TensorFlow opsional (hanya kalau pakai BiLSTM)
try:
    import tensorflow as tf
except Exception:
    tf = None


# =========================
# PATHS (SUDAH DISESUAIKAN DENGAN FOLDER KAMU)
# project/
# ├─ models/
# └─ src/app.py
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))               # .../src
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))         # .../
MODELS_DIR = os.path.join(PROJECT_DIR, "models")                    # .../models

INDOBERT_DIR = os.path.join(MODELS_DIR, "indobert")
DISTILBERT_DIR = os.path.join(MODELS_DIR, "distilbert")

LABELS_JSON = os.path.join(MODELS_DIR, "label_classes.json")
BILSTM_PATH = os.path.join(MODELS_DIR, "base_bilstm.keras")
TOKENIZER_PKL = os.path.join(MODELS_DIR, "tokenizer.pkl")


# =========================
# HELPERS
# =========================
def clean_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_labels() -> list:
    # ambil label dari label_classes.json kalau ada
    if os.path.exists(LABELS_JSON):
        with open(LABELS_JSON, "r", encoding="utf-8") as f:
            labels = json.load(f)
        if isinstance(labels, list) and len(labels) >= 2:
            return labels
    # fallback
    return ["negatif", "netral", "positif"]


@st.cache_resource
def load_hf(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
    mdl.eval()
    return tok, mdl


def predict_hf(tok, mdl, text: str):
    inputs = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        out = mdl(**inputs)
        probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    return pred, probs


@st.cache_resource
def load_bilstm():
    if tf is None:
        raise RuntimeError("TensorFlow belum terinstall. Install tensorflow untuk menggunakan BiLSTM.")
    if not os.path.exists(BILSTM_PATH):
        raise FileNotFoundError(f"Model BiLSTM tidak ditemukan: {BILSTM_PATH}")
    if not os.path.exists(TOKENIZER_PKL):
        raise FileNotFoundError(f"Tokenizer tidak ditemukan: {TOKENIZER_PKL}")

    model = tf.keras.models.load_model(BILSTM_PATH)
    with open(TOKENIZER_PKL, "rb") as f:
        tok = pickle.load(f)
    return model, tok


def predict_bilstm(model, tok, text: str, maxlen: int = 100):
    seq = tok.texts_to_sequences([text])
    pad = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=maxlen, padding="post")
    probs = model.predict(pad, verbose=0)[0]
    pred = int(np.argmax(probs))
    return pred, probs


# =========================
# UI
# =========================
st.set_page_config(page_title="Sentimen Review Spotify", page_icon="🎧", layout="centered")
st.title("🎧 Klasifikasi Sentimen Review Spotify (Bahasa Indonesia)")
st.caption("Demo Streamlit (BiLSTM / IndoBERT / DistilBERT)")

labels = load_labels()

# Deteksi model berdasarkan folder (lebih stabil daripada cek file satu-satu)
available_models = []
if os.path.isdir(INDOBERT_DIR):
    available_models.append("IndoBERT")
if os.path.isdir(DISTILBERT_DIR):
    available_models.append("DistilBERT")

bilstm_available = os.path.exists(BILSTM_PATH) and os.path.exists(TOKENIZER_PKL) and (tf is not None)
if bilstm_available:
    available_models.append("BiLSTM (Base)")

# Kalau tensorflow belum ada, tapi file bilstm ada, tampilkan info
if (os.path.exists(BILSTM_PATH) and os.path.exists(TOKENIZER_PKL) and tf is None):
    st.info("BiLSTM tersedia, tapi TensorFlow belum terinstall. Install `tensorflow` jika ingin memakai BiLSTM.")

# Validasi minimal ada transformer model
if not available_models:
    st.error(
        "Tidak menemukan model di folder `models/`.\n\n"
        "Pastikan struktur folder seperti ini:\n"
        "- models/indobert/\n"
        "- models/distilbert/\n"
        "- models/label_classes.json\n"
        "(opsional) models/base_bilstm.keras dan models/tokenizer.pkl\n\n"
        f"Path models yang dicari: {MODELS_DIR}"
    )
    st.stop()

model_choice = st.selectbox("Pilih model", available_models, index=0)

text = st.text_area(
    "Masukkan review:",
    placeholder="Contoh: Aplikasinya bagus, tapi kadang error pas login.",
    height=160
)

col1, col2 = st.columns([1, 1])
with col1:
    do_clean = st.checkbox("Cleaning teks", value=True)
with col2:
    show_probs = st.checkbox("Tampilkan probabilitas", value=True)

if st.button("🔮 Prediksi"):
    if not text.strip():
        st.warning("Teks masih kosong.")
        st.stop()

    text_proc = clean_text(text) if do_clean else text

    try:
        if model_choice == "IndoBERT":
            tok, mdl = load_hf(INDOBERT_DIR)
            pred_idx, probs = predict_hf(tok, mdl, text_proc)

        elif model_choice == "DistilBERT":
            tok, mdl = load_hf(DISTILBERT_DIR)
            pred_idx, probs = predict_hf(tok, mdl, text_proc)

        else:  # BiLSTM
            model, tok = load_bilstm()
            pred_idx, probs = predict_bilstm(model, tok, text_proc, maxlen=100)

        pred_label = labels[pred_idx] if pred_idx < len(labels) else f"kelas_{pred_idx}"

        st.success(f"Hasil Prediksi: **{pred_label.upper()}**")

        if show_probs:
            st.subheader("Probabilitas per kelas")
            for i, lab in enumerate(labels):
                p = float(probs[i]) if i < len(probs) else 0.0
                st.write(f"- **{lab}**: {p:.4f}")

            st.progress(float(np.max(probs)))

    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")

st.divider()
with st.expander("📌 Info Path (debug)"):
    st.write("BASE_DIR:", BASE_DIR)
    st.write("PROJECT_DIR:", PROJECT_DIR)
    st.write("MODELS_DIR:", MODELS_DIR)
    st.write("INDOBERT_DIR exists:", os.path.isdir(INDOBERT_DIR))
    st.write("DISTILBERT_DIR exists:", os.path.isdir(DISTILBERT_DIR))
    st.write("LABELS exists:", os.path.exists(LABELS_JSON))
    st.write("BiLSTM exists:", os.path.exists(BILSTM_PATH))
    st.write("Tokenizer exists:", os.path.exists(TOKENIZER_PKL))

with st.expander("▶️ Cara run (Windows)"):
    st.code(
        'cd "D:\\UAP ML\\src"\n'
        "python -m streamlit run app.py\n",
        language="powershell"
    )
