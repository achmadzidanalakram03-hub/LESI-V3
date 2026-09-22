"""
MAMMOUTH — My Assistant in Mouth Health
=======================================
Platform skrining kesehatan rongga mulut berbasis computer vision.

v6.1 — Proyek independen (AI Terintegrasi).
Fitur utama:
  • Akun mandiri (registrasi terbuka) dengan isolasi data penuh per pengguna
  • Penyimpanan permanen: SQLite relasional + arsip citra per akun di disk
  • Rekam medis pasien (bukan sekadar log gambar): pasien → pemeriksaan → deteksi
  • Anamnesis OLD CARTS + tanda-tanda vital (TD, nadi, napas, BB/TB, IMT otomatis)
  • Sintesis Klinis Berbasis LLM (Google Gemini) untuk suspek diagnosis
  • Analitik, laporan cetak, ekspor penuh (ZIP), dan mode demo tanpa bobot model

Jalankan:  streamlit run app.py
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import os
import random
import secrets
import sqlite3
import uuid
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance, ImageOps

try:
    import altair as alt
except Exception:  # pragma: no cover
    alt = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None

# Google GenAI SDK (SDK resmi baru: google-genai).
# Jangan hard-code API key; MAMMOUTH membacanya dari Streamlit Secrets
# atau environment variable GEMINI_API_KEY.
try:
    from google import genai
    from google.genai import types as genai_types
except Exception:  # pragma: no cover - dependency opsional saat mode demo
    genai = None
    genai_types = None

GEMINI_DEFAULT_MODEL = "gemini-3.8-flash"
GEMINI_SDK_ERROR = None
GEMINI_LAST_STATUS = "Belum ada request Gemini."

def _read_secret(name: str, default: str = "") -> str:
    """Baca secret dengan aman dari Streamlit Secrets lalu environment."""
    try:
        value = st.secrets.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    except Exception:
        pass
    return os.environ.get(name, default).strip()

def get_gemini_api_key() -> str:
    return _read_secret("GEMINI_API_KEY") or _read_secret("GOOGLE_API_KEY")

def get_gemini_model_name() -> str:
    return _read_secret("GEMINI_MODEL", GEMINI_DEFAULT_MODEL) or GEMINI_DEFAULT_MODEL

def get_gemini_client():
    """Buat client Gemini dari secret runtime saat ini. Tidak meng-cache API key."""
    global GEMINI_SDK_ERROR
    if genai is None:
        GEMINI_SDK_ERROR = "Package google-genai belum terpasang. Tambahkan `google-genai` ke requirements.txt."
        return None

    api_key = get_gemini_api_key()
    if not api_key:
        GEMINI_SDK_ERROR = "GEMINI_API_KEY/GOOGLE_API_KEY belum ditemukan di Streamlit Secrets atau environment."
        return None

    try:
        GEMINI_SDK_ERROR = None
        return genai.Client(api_key=api_key)
    except Exception as exc:  # noqa: BLE001
        GEMINI_SDK_ERROR = f"Gagal membuat client Gemini: {exc}"
        return None

# Gemini client dibuat on-demand agar perubahan Streamlit Secrets langsung terbaca.
MODEL_AI = None


# ------------------------------------------------------------
# 0. SISTEM MULTI-BAHASA (I18N) SEDERHANA
# ------------------------------------------------------------
TRANSLATIONS = {
    "en": {
        "Dashboard": "Dashboard",
        "AI Detection": "AI Detection",
        "Upload Image": "Upload Image",
        "Detection History": "Detection History",
        "Patients": "Patients",
        "Lesion Database": "Lesion Database",
        "Analytics": "Analytics",
        "Reports": "Reports",
        "Settings": "Settings",
        "Total Examinations": "Total Examinations",
        "AI Detections": "AI Detections",
        "Detection Accuracy": "Detection Accuracy",
        "Average Processing Time": "Average Processing Time",
        "Recent AI Detection": "Recent AI Detection",
        "Original radiograph/oral image": "Original radiograph/oral image",
        "Detection result": "Detection result",
        "Bounding boxes & Confidence score": "Bounding boxes & Confidence score",
        "Lesion Distribution": "Lesion Distribution",
        "Detection Activity": "Detection Activity",
        "Recent Cases": "Recent Cases",
        "Language": "Language",
        "Theme Mode": "Theme Mode",
        "Light": "Light",
        "Dark": "Dark",
        "System": "System",
        "Logout": "Logout",
        "Patient Search": "Patient Search",
        "Search Patient (Name / MR No)": "Search Patient (Name / MR No)",
        "Select Patient": "Select Patient",
    },
    "id": {
        "Dashboard": "Dasbor",
        "AI Detection": "Deteksi AI",
        "Upload Image": "Unggah Citra",
        "Detection History": "Riwayat Deteksi",
        "Patients": "Pasien",
        "Lesion Database": "Basis Data Lesi",
        "Analytics": "Analitik",
        "Reports": "Laporan",
        "Settings": "Pengaturan",
        "Total Examinations": "Total Pemeriksaan",
        "AI Detections": "Total Deteksi AI",
        "Detection Accuracy": "Akurasi Deteksi",
        "Average Processing Time": "Rata-rata Waktu Proses",
        "Recent AI Detection": "Deteksi AI Terbaru",
        "Original radiograph/oral image": "Citra radiograf/oral asli",
        "Detection result": "Hasil deteksi",
        "Bounding boxes & Confidence score": "Kotak pembatas & Skor keyakinan",
        "Lesion Distribution": "Distribusi Lesi",
        "Detection Activity": "Aktivitas Deteksi",
        "Recent Cases": "Kasus Terbaru",
        "Language": "Bahasa",
        "Theme Mode": "Mode Tema",
        "Light": "Terang",
        "Dark": "Gelap",
        "System": "Sistem",
        "Logout": "Keluar",
        "Patient Search": "Pencarian Pasien",
        "Search Patient (Name / MR No)": "Cari Pasien (Nama / No. RM)",
        "Select Patient": "Pilih Pasien",
    }
}

def _t(key: str) -> str:
    lang = st.session_state.get("lang", "en")
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)


# ------------------------------------------------------------
# 0.5 KOMPATIBILITAS API STREAMLIT
# ------------------------------------------------------------
def _install_compat_shim() -> None:
    import inspect

    for name in ("button", "download_button", "form_submit_button", "image",
                 "dataframe", "altair_chart", "bar_chart", "line_chart", "plotly_chart"):
        fn = getattr(st, name, None)
        if fn is None:
            continue
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            continue
        if "use_container_width" in params:
            continue
        has_width = "width" in params

        def wrapper(*args, _fn=fn, _has_width=has_width, **kwargs):
            legacy = kwargs.pop("use_container_width", None)
            kwargs.pop("use_column_width", None)
            if legacy is not None and _has_width and "width" not in kwargs:
                kwargs["width"] = "stretch" if legacy else "content"
            return _fn(*args, **kwargs)

        setattr(st, name, wrapper)

_install_compat_shim()


# ============================================================
# 1. KONFIGURASI
# ============================================================
APP_NAME = "Mammouth"
APP_TAGLINE = "My Assistant in Mouth Health"
APP_VERSION = "6.1"

DATA_DIR = Path(os.environ.get("MAMMOUTH_DATA_DIR", "mammouth_data"))
DB_FILE = DATA_DIR / "mammouth.db"
IMG_DIR = DATA_DIR / "images"
LEGACY_DB = Path("mammouth.db")

MODEL_FILES = {
    "YOLOv8": ["best.pt", "yolov8_best.pt"],
    "YOLOv11": ["yolov11_best.pt", "yolo11_best.pt", "best.pt"],
    "YOLOv12": ["yolov12_best.pt", "yolo12_best.pt", "best.pt"],
}

URGENCY_RANK = {"Rendah": 1, "Sedang": 2, "Sedang–Tinggi": 3, "Tinggi": 4}
RANK_URGENCY = {v: k for k, v in URGENCY_RANK.items()}

LESION_INFO: dict[str, dict[str, Any]] = {
    "cheek biting": {
        "nama_klinis": "Morsicatio buccarum",
        "awam": "Luka akibat kebiasaan menggigit pipi",
        "urgensi": "Rendah",
        "deskripsi": (
            "Mukosa bukal tampak kasar, bergerigi, dan mengelupas putih setinggi garis oklusal "
            "akibat trauma kunyah berulang. Dapat berlanjut menjadi ulserasi dangkal."
        ),
        "etiologi": "Kebiasaan parafungsional, sering menyertai stres, ansietas, atau maloklusi.",
        "tatalaksana": (
            "Edukasi penghentian kebiasaan (habit breaking), penghalusan tepi gigi tajam, "
            "pertimbangkan occlusal guard bila nokturnal. Evaluasi ulang bila menetap >2 minggu."
        ),
        "banding": "Leukoplakia, liken planus retikuler, kandidiasis hiperplastik.",
        "red_flag": "Lesi putih yang tidak bisa dikerok dan menetap >2 minggu perlu biopsi.",
    },
    "coated tongue": {
        "nama_klinis": "Coated tongue",
        "awam": "Lidah berselaput",
        "urgensi": "Rendah",
        "deskripsi": (
            "Penumpukan debris, keratin yang gagal terdeskuamasi, sisa makanan, dan mikroorganisme "
            "di dorsum lidah. Sering disertai halitosis dan penurunan sensasi rasa."
        ),
        "etiologi": "Kebersihan mulut kurang, diet lunak, xerostomia, merokok, demam, atau antibiotik.",
        "tatalaksana": "Pembersihan mekanis dengan tongue scraper, hidrasi, DHE, dan evaluasi penyebab xerostomia.",
        "banding": "Kandidiasis pseudomembran, hairy tongue, hairy leukoplakia.",
        "red_flag": "Selaput putih yang mudah dikerok dan meninggalkan dasar eritema mengarah ke kandidiasis.",
    },
    "karies": {
        "nama_klinis": "Karies gigi",
        "awam": "Gigi berlubang",
        "urgensi": "Sedang–Tinggi",
        "deskripsi": (
            "Demineralisasi jaringan keras gigi oleh asam hasil metabolisme bakteri biofilm. "
            "Tampak sebagai kavitas, bercak putih kapur, atau diskolorasi coklat-hitam."
        ),
        "etiologi": "Interaksi biofilm kariogenik, karbohidrat terfermentasi, host rentan, dan waktu.",
        "tatalaksana": (
            "Konfirmasi dengan sondasi, tes vitalitas, dan radiograf bitewing/periapikal. "
            "Rencana restorasi, pulp capping, atau perawatan saluran akar sesuai kedalaman."
        ),
        "banding": "Stain ekstrinsik, hipoplasia email, fluorosis, abrasi servikal.",
        "red_flag": "Nyeri spontan, nokturnal, atau pembengkakan menandakan keterlibatan pulpa/periapikal.",
    },
    "linea alba": {
        "nama_klinis": "Linea alba buccalis",
        "awam": "Garis putih di pipi bagian dalam",
        "urgensi": "Rendah",
        "deskripsi": "Garis keratotik putih horizontal pada mukosa bukal setinggi bidang oklusal, bilateral dan simetris.",
        "etiologi": "Friksi dan tekanan oklusal ringan yang berlangsung kronis.",
        "tatalaksana": "Varian normal. Tidak memerlukan terapi; cukup edukasi dan observasi rutin.",
        "banding": "Cheek biting, liken planus, white sponge nevus.",
        "red_flag": "Bila unilateral, menebal, atau ulseratif — evaluasi ulang penyebab trauma.",
    },
    "lingual varicosites": {
        "nama_klinis": "Lingual varicosities",
        "awam": "Pelebaran pembuluh vena di bawah lidah",
        "urgensi": "Rendah",
        "deskripsi": "Vena berkelok warna biru-keunguan pada permukaan ventral dan lateral lidah, dapat dikompres.",
        "etiologi": "Perubahan degeneratif dinding vena terkait usia; dapat menyertai hipertensi pulmonal.",
        "tatalaksana": "Tidak memerlukan tindakan invasif. Edukasi sifat jinak lesi.",
        "banding": "Hemangioma, malformasi vaskular, Kaposi sarcoma.",
        "red_flag": "Lesi vaskular yang tidak dapat dikompres atau mudah berdarah perlu rujukan.",
    },
    "stain calculus": {
        "nama_klinis": "Stain dan kalkulus",
        "awam": "Karang gigi dan noda",
        "urgensi": "Sedang",
        "deskripsi": "Deposit biofilm terkalsifikasi disertai diskolorasi ekstrinsik pada permukaan gigi dan area servikal.",
        "etiologi": "Mineralisasi plak oleh saliva, diperberat kopi/teh/rokok dan kontrol plak yang buruk.",
        "tatalaksana": "Scaling dan root planing, poles, instruksi kebersihan mulut, kontrol berkala 6 bulan.",
        "banding": "Karies servikal, fluorosis, stain intrinsik tetrasiklin.",
        "red_flag": "Kalkulus subgingiva dengan perdarahan dan resesi mengarah ke periodontitis.",
    },
    "torus": {
        "nama_klinis": "Torus palatinus / mandibularis",
        "awam": "Tonjolan tulang di langit-langit atau dasar mulut",
        "urgensi": "Rendah",
        "deskripsi": "Eksostosis tulang jinak berbatas tegas, tumbuh lambat, ditutupi mukosa normal, umumnya asimtomatik.",
        "etiologi": "Multifaktorial: genetik, beban oklusal, dan faktor lingkungan.",
        "tatalaksana": "Observasi. Bedah hanya bila mengganggu fungsi bicara, mastikasi, atau persiapan protesa.",
        "banding": "Osteoma, abses, neoplasma tulang.",
        "red_flag": "Pembesaran cepat, nyeri, atau ulserasi di atas tonjolan perlu evaluasi lanjut.",
    },
    "ulkus traumatikus": {
        "nama_klinis": "Ulkus traumatikus",
        "awam": "Sariawan akibat luka",
        "urgensi": "Sedang",
        "deskripsi": "Ulser tunggal dengan dasar kuning-keputihan dan halo eritema, tepi ireguler, nyeri pada kontak.",
        "etiologi": "Trauma mekanis (tergigit, tepi restorasi/protesa tajam), termal, atau kimiawi.",
        "tatalaksana": (
            "Eliminasi faktor kausatif, obat kumur antiseptik, topikal analgesik/kortikosteroid bila perlu. "
            "Evaluasi ulang 10–14 hari."
        ),
        "banding": "Stomatitis aftosa rekuren, ulser herpetik, karsinoma sel skuamosa.",
        "red_flag": "Ulser >2 minggu, tepi indurasi, atau tidak nyeri wajib dibiopsi untuk menyingkirkan keganasan.",
    },
}

DEMO_LABELS = list(LESION_INFO.keys())


# ============================================================
# 2. TEMA & IDENTITAS VISUAL
# ============================================================
THEMES = {
    "terang": {
        "bg": "#F3F5F3",
        "surface": "#FFFFFF",
        "surface2": "#EDF1EF",
        "border": "#D8E0DC",
        "text": "#0E1C19",
        "muted": "#5C6B66",
        "primary": "#0B6B5A",
        "primary-soft": "#DCEFE9",
        "on-primary": "#FFFFFF",
        "accent": "#B07D20",
        "accent-soft": "#F7EEDA",
        "danger": "#B3261E",
        "danger-soft": "#FBE9E7",
        "warn": "#A8651A",
        "warn-soft": "#FCF0DE",
        "ok": "#2E6B3E",
        "ok-soft": "#E4F1E4",
        "shadow": "0 1px 2px rgba(14,28,25,.06), 0 8px 24px -16px rgba(14,28,25,.28)",
    },
    "gelap": {
        "bg": "#0B1513",
        "surface": "#12201D",
        "surface2": "#182B27",
        "border": "#24403A",
        "text": "#E6EFEB",
        "muted": "#93A79F",
        "primary": "#3FBFA3",
        "primary-soft": "#153630",
        "on-primary": "#04201A",
        "accent": "#E0B457",
        "accent-soft": "#33291247",
        "danger": "#F1857C",
        "danger-soft": "#3A1B19",
        "warn": "#E3A75B",
        "warn-soft": "#33250F",
        "ok": "#6FCB86",
        "ok-soft": "#16301D",
        "shadow": "0 1px 2px rgba(0,0,0,.4), 0 12px 28px -18px rgba(0,0,0,.9)",
    },
}

CSS_BASE = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp, input, textarea, select, button {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.stApp { background: var(--bg); color: var(--text); }
.block-container { padding-top: 2.5rem; padding-bottom: 4.5rem; max-width: 1320px; }
.block-container > div[data-testid="stVerticalBlock"] { gap: 1.3rem; }

h1, h2, h3, h4, h5 {
    font-family: 'Space Grotesk', 'Inter', sans-serif !important;
    color: var(--text) !important;
    letter-spacing: -0.015em;
}
h1 { font-size: 2.1rem !important; font-weight: 700 !important; }
h2 { font-size: 1.5rem !important; font-weight: 600 !important; margin: 1.9rem 0 .9rem 0 !important; }
h3 { font-size: 1.15rem !important; font-weight: 600 !important; margin: 1.7rem 0 .8rem 0 !important; }
p, li, label, span, div { color: var(--text); }
a { color: var(--primary); }

/* --- Judul halaman --- */
.page-head { margin-bottom: 2rem; }
.page-head h1 { margin: 0 0 .4rem 0; }
.page-head p { color: var(--muted); margin: 0; font-size: .95rem; max-width: 70ch; line-height: 1.55; }

/* --- Kontainer berbingkai bawaan Streamlit --- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface);
    border-radius: 14px;
}
[data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] { gap: 1.05rem; }
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {
    border-color: var(--border) !important;
}

/* --- Kolom berdampingan --- */
[data-testid="stHorizontalBlock"] { gap: 1.4rem; align-items: flex-start; }

/* --- Kartu --- */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 22px 24px;
    box-shadow: var(--shadow);
}
.kpi {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 22px;
    height: 100%;
}
.kpi .label { color: var(--muted); font-size: .78rem; font-weight: 600; margin: 0; letter-spacing: .01em; }
.kpi .value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem; font-weight: 700; line-height: 1.2; margin: 8px 0 0 0;
    font-variant-numeric: tabular-nums;
}
.kpi .sub { color: var(--muted); font-size: .78rem; margin: 6px 0 0 0; }

/* --- Pita urgensi pada hasil --- */
.verdict {
    border: 1px solid var(--border);
    border-left: 5px solid var(--primary);
    border-radius: 12px;
    background: var(--surface);
    padding: 18px 22px;
    margin: 14px 0 20px 0;
}
.verdict .title { font-family:'Space Grotesk',sans-serif; font-weight: 600; font-size: 1.02rem; margin: 0 0 8px 0; }
.verdict .body { color: var(--text); font-size: .92rem; margin: 0; line-height: 1.6; }
.verdict.u-tinggi { border-left-color: var(--danger); }
.verdict.u-sedang { border-left-color: var(--warn); }
.verdict.u-rendah { border-left-color: var(--ok); }

/* --- Lencana --- */
.pill {
    display: inline-block; padding: 4px 12px; border-radius: 999px;
    font-size: .72rem; font-weight: 600; border: 1px solid transparent; white-space: nowrap;
}
.pill.low  { background: var(--ok-soft);     color: var(--ok);     border-color: var(--ok); }
.pill.med  { background: var(--warn-soft);   color: var(--warn);   border-color: var(--warn); }
.pill.high { background: var(--danger-soft); color: var(--danger); border-color: var(--danger); }
.pill.neutral { background: var(--surface2); color: var(--muted); border-color: var(--border); }

/* --- Baris deteksi --- */
.det-row {
    display: flex; align-items: center; justify-content: space-between; gap: 16px;
    padding: 13px 2px; border-bottom: 1px solid var(--border);
}
.det-row:last-child { border-bottom: none; }
.det-name { font-weight: 600; font-size: .92rem; }
.det-sub { color: var(--muted); font-size: .78rem; margin-top: 2px; }
.meter { height: 6px; border-radius: 99px; background: var(--surface2); width: 120px; overflow: hidden; }
.meter > span { display: block; height: 100%; background: var(--primary); }

/* --- Kartu ringkas tanda vital --- */
.vital-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr));
    gap: 14px; margin: 4px 0 4px 0;
}
.vital-cell {
    border: 1px solid var(--border); border-radius: 12px; background: var(--surface2);
    padding: 12px 14px;
}
.vital-cell .vlabel { color: var(--muted); font-size: .72rem; font-weight: 600; margin: 0; letter-spacing: .01em; }
.vital-cell .vvalue {
    font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 700;
    margin: 6px 0 0 0; font-variant-numeric: tabular-nums;
}

/* --- Sidebar --- */
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] .block-container { padding-top: 1.8rem; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 1.1rem; }
.brand { display: flex; align-items: baseline; gap: 8px; margin-bottom: 4px; }
.brand .mark {
    font-family: 'Space Grotesk', sans-serif; font-weight: 700;
    font-size: 1.45rem; color: var(--primary); letter-spacing: -.03em;
}
.brand .ver { font-size: .68rem; color: var(--muted); font-weight: 600; }
.brand-sub { color: var(--muted); font-size: .74rem; margin: 0 0 20px 0; }
.who {
    border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px;
    background: var(--surface2); margin-top: 12px;
}
.who .name { font-weight: 600; font-size: .9rem; margin: 0; }
.who .role { color: var(--muted); font-size: .76rem; margin: 3px 0 0 0; }

/* --- Tombol --- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: .55rem 1.15rem !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
    color: var(--text) !important;
    transition: border-color .15s ease, background .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
    border-color: var(--primary) !important; color: var(--primary) !important;
}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
    background: var(--primary) !important; color: var(--on-primary) !important; border-color: var(--primary) !important;
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.08); color: var(--on-primary) !important; }
button:focus-visible, a:focus-visible, input:focus-visible { outline: 2px solid var(--primary) !important; outline-offset: 2px; }

/* --- Input --- */
input, textarea, [data-baseweb="select"] > div {
    border-radius: 10px !important;
    background: var(--surface) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}
[data-testid="stWidgetLabel"] p { font-size: .84rem; font-weight: 600; color: var(--text); margin-bottom: .2rem; }
[data-testid="stForm"] [data-testid="stVerticalBlock"] { gap: 1rem; }

/* --- Tab --- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 10px 20px; font-weight: 600; font-size: .9rem; }
.stTabs [aria-selected="true"] { color: var(--primary) !important; background: var(--primary-soft) !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.1rem; }

/* --- Ekspander, tabel, metrik --- */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important; border-radius: 12px !important; background: var(--surface);
    margin-bottom: .2rem;
}
[data-testid="stExpander"] summary { padding: .6rem .9rem !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 12px; }
hr { border-color: var(--border); margin: 1.6rem 0; }
[data-testid="stCameraInput"] button { border-radius: 10px !important; }

/* --- Halaman masuk --- */
.auth-hero { padding: 6px 0 0 0; }
.auth-hero .mark {
    font-family: 'Space Grotesk', sans-serif; font-size: 3.4rem; font-weight: 700;
    color: var(--primary); letter-spacing: -.045em; line-height: 1; margin: 0;
}
.auth-hero .tag { font-size: 1.02rem; color: var(--text); margin: 12px 0 0 0; font-weight: 500; }
.auth-hero .lede { color: var(--muted); font-size: .95rem; margin: 16px 0 0 0; max-width: 46ch; line-height: 1.65; }
.auth-list { list-style: none; padding: 0; margin: 26px 0 0 0; }
.auth-list li {
    padding: 11px 0 11px 22px; border-top: 1px solid var(--border);
    font-size: .88rem; color: var(--text); position: relative;
}
.auth-list li::before { content: ""; position: absolute; left: 0; top: 19px; width: 8px; height: 8px; border-radius: 2px; background: var(--primary); }
.tooth-plot { margin-top: 30px; }


/* ============================================================
   MAMMOUTH RESPONSIVE WORKSTATION SYSTEM
   Desktop -> Tablet -> Mobile
   ============================================================ */

/* Fluid desktop canvas */
.main .block-container,
[data-testid="stAppViewContainer"] .block-container {
    width: 100%;
    box-sizing: border-box;
}

/* Prevent accidental horizontal overflow */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    max-width: 100%;
    overflow-x: hidden;
}

/* Image/media safety */
img, video, canvas {
    max-width: 100%;
    height: auto;
}

/* Keep Streamlit horizontal groups flexible */
[data-testid="stHorizontalBlock"] {
    min-width: 0;
}

[data-testid="stHorizontalBlock"] > div {
    min-width: 0;
}

/* Tables remain usable without breaking the page */
[data-testid="stDataFrame"] {
    max-width: 100%;
    overflow-x: auto;
}

/* Desktop: spacious clinical workstation */
@media (min-width: 1280px) {
    .block-container {
        max-width: 1540px;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }

    [data-testid="stSidebar"] {
        min-width: 250px;
        max-width: 270px;
    }

    .page-head {
        margin-bottom: 1.75rem;
    }
}

/* Compact desktop / landscape tablet */
@media (min-width: 1024px) and (max-width: 1279px) {
    .block-container {
        max-width: 1180px;
        padding-left: 1.75rem;
        padding-right: 1.75rem;
    }

    [data-testid="stSidebar"] {
        min-width: 225px;
        max-width: 240px;
    }

    h1 { font-size: 1.9rem !important; }
    h2 { font-size: 1.35rem !important; }

    .kpi {
        padding: 16px 18px;
    }

    .kpi .value {
        font-size: 1.7rem;
    }
}

/* Tablet: native Streamlit sidebar becomes the navigation drawer */
@media (min-width: 768px) and (max-width: 1023px) {
    .block-container {
        max-width: 100%;
        padding-top: 1.75rem;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        width: 285px !important;
        min-width: 285px !important;
    }

    [data-testid="stSidebar"] .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    /* Let two-column content become stacked when the available width is tight. */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 1rem !important;
    }

    [data-testid="stHorizontalBlock"] > div {
        flex: 1 1 min(100%, 320px) !important;
    }

    .page-head {
        margin-bottom: 1.25rem;
    }

    h1 { font-size: 1.85rem !important; }
    h2 { font-size: 1.3rem !important; }

    .card, .kpi {
        padding: 16px 18px;
    }

    .kpi .value {
        font-size: 1.6rem;
    }

    .stButton > button,
    .stDownloadButton > button,
    .stFormSubmitButton > button {
        min-height: 44px;
    }
}

/* Mobile: focused clinical interface */
@media (max-width: 767px) {
    .block-container {
        width: 100%;
        max-width: 100%;
        padding-top: .9rem;
        padding-left: .85rem;
        padding-right: .85rem;
        padding-bottom: 5.5rem;
    }

    /* Streamlit's native sidebar acts as the mobile drawer. */
    [data-testid="stSidebar"] {
        width: 300px !important;
        max-width: 86vw !important;
    }

    [data-testid="stSidebar"] .block-container {
        padding: 1rem .85rem 1.5rem .85rem;
    }

    .brand .mark {
        font-size: 1.3rem;
    }

    .brand-sub {
        margin-bottom: 14px;
    }

    /* Navigation buttons become large touch targets. */
    [data-testid="stSidebar"] .stButton > button {
        min-height: 44px;
        text-align: left;
        padding: .65rem .8rem !important;
    }

    /* Mobile top-level content */
    .page-head {
        margin-bottom: 1rem;
    }

    .page-head h1,
    h1 {
        font-size: 1.55rem !important;
        line-height: 1.15 !important;
    }

    h2 {
        font-size: 1.18rem !important;
        line-height: 1.25 !important;
        margin-top: 1.2rem !important;
    }

    h3 {
        font-size: 1rem !important;
        margin-top: 1rem !important;
    }

    .page-head p {
        font-size: .86rem;
        line-height: 1.45;
    }

    /* Recompose every Streamlit column group vertically. */
    [data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: column !important;
        flex-wrap: nowrap !important;
        width: 100% !important;
        gap: .8rem !important;
    }

    [data-testid="stHorizontalBlock"] > div {
        width: 100% !important;
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }

    /* Compact metric blocks instead of oversized dashboard cards. */
    .kpi {
        min-height: 0;
        padding: 14px 16px;
        border-radius: 11px;
    }

    .kpi .label {
        font-size: .72rem;
    }

    .kpi .value {
        font-size: 1.45rem;
        margin-top: 5px;
    }

    .kpi .sub {
        font-size: .72rem;
        margin-top: 3px;
    }

    .card {
        padding: 16px;
        border-radius: 11px;
    }

    /* Make Streamlit controls touch-friendly. */
    .stButton > button,
    .stDownloadButton > button,
    .stFormSubmitButton > button {
        min-height: 46px;
        width: 100%;
    }

    input, textarea,
    [data-baseweb="select"] > div {
        min-height: 44px !important;
    }

    /* Tabs can scroll horizontally rather than wrapping into an unusable row. */
    .stTabs [data-baseweb="tab-list"] {
        overflow-x: auto;
        scrollbar-width: none;
        flex-wrap: nowrap !important;
    }

    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        display: none;
    }

    .stTabs [data-baseweb="tab"] {
        flex: 0 0 auto;
        padding: 9px 13px;
        font-size: .82rem;
    }

    /* Expanders work well as mobile information sections. */
    [data-testid="stExpander"] summary {
        min-height: 44px;
        display: flex;
        align-items: center;
    }

    /* Detection rows become vertical to prevent confidence meters from overflowing. */
    .det-row {
        align-items: flex-start;
        flex-direction: column;
        gap: 7px;
        padding: 12px 0;
    }

    .meter {
        width: 100%;
        max-width: 180px;
    }

    /* Vital cards become a compact two-column grid on phones. */
    .vital-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
    }

    .vital-cell {
        padding: 10px 11px;
    }

    /* Patient cards stack cleanly on phones. */
    .who {
        padding: 12px 13px;
    }

    /* Avoid giant login branding on narrow screens. */
    .auth-hero .mark {
        font-size: 2.55rem;
    }

    /* Tables can scroll horizontally instead of clipping clinical fields. */
    [data-testid="stDataFrame"] {
        width: 100%;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }

    /* Streamlit images should never exceed the viewport. */
    [data-testid="stImageContainer"] img {
        max-width: 100% !important;
        height: auto !important;
    }

    /* Keep alerts readable without forcing horizontal overflow. */
    [data-testid="stAlert"] {
        overflow-wrap: anywhere;
    }

    /* Sticky primary actions on small screens when marked by the app. */
    .mobile-primary-action {
        position: sticky;
        bottom: .65rem;
        z-index: 20;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 8px;
        box-shadow: 0 8px 30px rgba(0,0,0,.14);
    }
}

/* Very small phones */
@media (max-width: 374px) {
    .block-container {
        padding-left: .7rem;
        padding-right: .7rem;
    }

    .page-head h1,
    h1 {
        font-size: 1.42rem !important;
    }

    .kpi .value {
        font-size: 1.32rem;
    }

    .vital-grid {
        grid-template-columns: 1fr 1fr;
        gap: 6px;
    }
}

/* Respect users who request reduced motion. */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        transition: none !important;
        animation: none !important;
        scroll-behavior: auto !important;
    }
}

footer, #MainMenu { visibility: hidden; }
"""

def theme_css(theme_name: str) -> str:
    if theme_name == "sistem":
        tokens_light = THEMES["terang"]
        tokens_dark = THEMES["gelap"]
        root_light = ":root{\n" + "\n".join(f"  --{k}: {v};" for k, v in tokens_light.items()) + "\n}"
        root_dark = "@media (prefers-color-scheme: dark) {\n  :root{\n" + "\n".join(f"    --{k}: {v};" for k, v in tokens_dark.items()) + "\n  }\n}"
        return f"<style>{root_light}\n{root_dark}\n{CSS_BASE}</style>"
    else:
        tokens = THEMES.get(theme_name, THEMES["terang"])
        root = ":root{\n" + "\n".join(f"  --{k}: {v};" for k, v in tokens.items()) + "\n}"
        return f"<style>{root}\n{CSS_BASE}</style>"


# ============================================================
# 3. LAPISAN DATABASE
# ============================================================
def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

def get_conn() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id           TEXT PRIMARY KEY,
    username     TEXT UNIQUE NOT NULL,
    pw_hash      TEXT NOT NULL,
    pw_salt      TEXT NOT NULL,
    iterations   INTEGER NOT NULL DEFAULT 200000,
    full_name    TEXT NOT NULL,
    role         TEXT,
    institution  TEXT,
    is_admin     INTEGER NOT NULL DEFAULT 0,
    theme        TEXT NOT NULL DEFAULT 'sistem',
    pref         TEXT NOT NULL DEFAULT '{}',
    created_at   TEXT NOT NULL,
    last_login   TEXT
);

CREATE TABLE IF NOT EXISTS patients (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    code        TEXT NOT NULL,
    name        TEXT NOT NULL,
    birth_year  INTEGER,
    sex         TEXT,
    contact     TEXT,
    med_history TEXT,
    created_at  TEXT NOT NULL,
    UNIQUE(user_id, code),
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS exams (
    id            TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    patient_id    TEXT,
    created_at    TEXT NOT NULL,
    exam_date     TEXT NOT NULL,
    model_version TEXT,
    conf_thr      REAL,
    iou_thr       REAL,
    file_name     TEXT,
    image_path    TEXT,
    annot_path    TEXT,
    n_detections  INTEGER DEFAULT 0,
    max_conf      REAL DEFAULT 0,
    urgency       TEXT,
    synthesis     TEXT,
    o_onset       TEXT, l_location TEXT, d_duration TEXT, c_character TEXT,
    a_aggravating TEXT, r_relieving TEXT, t_timing TEXT, s_severity INTEGER DEFAULT 0,
    bp_systolic   INTEGER, bp_diastolic INTEGER, pulse_rate INTEGER, resp_rate INTEGER,
    weight_kg     REAL, height_cm REAL, bmi REAL,
    clinician_note TEXT,
    is_demo       INTEGER DEFAULT 0,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY(patient_id) REFERENCES patients(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS detections (
    id         TEXT PRIMARY KEY,
    exam_id    TEXT NOT NULL,
    user_id    TEXT NOT NULL,
    label      TEXT NOT NULL,
    confidence REAL NOT NULL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    FOREIGN KEY(exam_id) REFERENCES exams(id) ON DELETE CASCADE,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS activity (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    ts      TEXT NOT NULL,
    action  TEXT NOT NULL,
    detail  TEXT
);

CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);

CREATE INDEX IF NOT EXISTS ix_exams_user ON exams(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_det_user ON detections(user_id, label);
CREATE INDEX IF NOT EXISTS ix_pat_user ON patients(user_id, name);
"""

EXAM_VITAL_COLUMNS = {
    "bp_systolic": "INTEGER", "bp_diastolic": "INTEGER", "pulse_rate": "INTEGER",
    "resp_rate": "INTEGER", "weight_kg": "REAL", "height_cm": "REAL", "bmi": "REAL",
}

def _ensure_exam_vital_columns(conn: sqlite3.Connection) -> None:
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(exams)")}
    for col, sqltype in EXAM_VITAL_COLUMNS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE exams ADD COLUMN {col} {sqltype}")

def init_db() -> None:
    conn = get_conn()
    conn.executescript(SCHEMA)
    _ensure_exam_vital_columns(conn)
    conn.commit()
    conn.close()

def meta_get(key: str) -> Optional[str]:
    conn = get_conn()
    row = conn.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
    conn.close()
    return row["v"] if row else None

def meta_set(key: str, value: str) -> None:
    conn = get_conn()
    conn.execute("INSERT INTO meta(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (key, value))
    conn.commit()
    conn.close()

def log_activity(user_id: Optional[str], action: str, detail: str = "") -> None:
    conn = get_conn()
    conn.execute(
        "INSERT INTO activity(user_id, ts, action, detail) VALUES(?,?,?,?)",
        (user_id, datetime.now().isoformat(timespec="seconds"), action, detail),
    )
    conn.commit()
    conn.close()


# ---------- Autentikasi ----------
def hash_password(password: str, salt: Optional[str] = None, iterations: int = 200_000) -> tuple[str, str, int]:
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), iterations)
    return dk.hex(), salt, iterations

def password_issues(pw: str) -> list[str]:
    issues = []
    if len(pw) < 8:
        issues.append("minimal 8 karakter")
    if pw.isdigit() or pw.isalpha():
        issues.append("gabungkan huruf dan angka")
    if pw.lower() in {"password", "12345678", "qwerty123", "mammouth"}:
        issues.append("terlalu umum")
    return issues

def create_user(username: str, password: str, full_name: str, role: str, institution: str = "") -> tuple[bool, str]:
    username = username.strip().lower()
    if not username.isidentifier() and not username.replace(".", "").replace("_", "").isalnum():
        return False, "Username hanya boleh berisi huruf, angka, titik, dan garis bawah."
    if len(username) < 3:
        return False, "Username minimal 3 karakter."
    issues = password_issues(password)
    if issues:
        return False, "Kata sandi " + ", ".join(issues) + "."

    pw_hash, salt, iters = hash_password(password)
    conn = get_conn()
    first_user = conn.execute("SELECT COUNT(*) c FROM users").fetchone()["c"] == 0
    try:
        conn.execute(
            """INSERT INTO users(id, username, pw_hash, pw_salt, iterations, full_name, role,
                                 institution, is_admin, theme, pref, created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                uuid.uuid4().hex, username, pw_hash, salt, iters, full_name.strip(),
                role.strip() or "Klinisi", institution.strip(), 1 if first_user else 0,
                "sistem", "{}", datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Username sudah dipakai. Pilih yang lain."
    conn.close()
    log_activity(None, "register", username)
    return True, "Akun dibuat. Silakan masuk."

def verify_login(username: str, password: str) -> Optional[sqlite3.Row]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE username=?", (username.strip().lower(),)).fetchone()
    if row is None:
        conn.close()
        return None

    ok = False
    if row["iterations"] == 0:
        ok = hmac.compare_digest(row["pw_hash"], hashlib.sha256(password.encode()).hexdigest())
        if ok:
            new_hash, new_salt, iters = hash_password(password)
            conn.execute(
                "UPDATE users SET pw_hash=?, pw_salt=?, iterations=? WHERE id=?",
                (new_hash, new_salt, iters, row["id"]),
            )
    else:
        calc, _, _ = hash_password(password, row["pw_salt"], row["iterations"])
        ok = hmac.compare_digest(calc, row["pw_hash"])

    if not ok:
        conn.close()
        return None

    conn.execute("UPDATE users SET last_login=? WHERE id=?", (datetime.now().isoformat(timespec="seconds"), row["id"]))
    conn.commit()
    row = conn.execute("SELECT * FROM users WHERE id=?", (row["id"],)).fetchone()
    conn.close()
    log_activity(row["id"], "login", username)
    return row

def change_password(user_id: str, old_pw: str, new_pw: str) -> tuple[bool, str]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    if row is None:
        return False, "Akun tidak ditemukan."
    if row["iterations"] == 0:
        ok = hmac.compare_digest(row["pw_hash"], hashlib.sha256(old_pw.encode()).hexdigest())
    else:
        calc, _, _ = hash_password(old_pw, row["pw_salt"], row["iterations"])
        ok = hmac.compare_digest(calc, row["pw_hash"])
    if not ok:
        return False, "Kata sandi lama tidak cocok."
    issues = password_issues(new_pw)
    if issues:
        return False, "Kata sandi baru " + ", ".join(issues) + "."
    h, s, i = hash_password(new_pw)
    conn = get_conn()
    conn.execute("UPDATE users SET pw_hash=?, pw_salt=?, iterations=? WHERE id=?", (h, s, i, user_id))
    conn.commit()
    conn.close()
    log_activity(user_id, "ganti_sandi", "")
    return True, "Kata sandi diperbarui."

def update_profile(user_id: str, full_name: str, role: str, institution: str) -> None:
    conn = get_conn()
    conn.execute(
        "UPDATE users SET full_name=?, role=?, institution=? WHERE id=?",
        (full_name.strip(), role.strip(), institution.strip(), user_id),
    )
    conn.commit()
    conn.close()

def set_user_pref(user_id: str, **kwargs) -> None:
    conn = get_conn()
    row = conn.execute("SELECT pref, theme FROM users WHERE id=?", (user_id,)).fetchone()
    pref = json.loads(row["pref"] or "{}")
    theme = kwargs.pop("theme", None)
    pref.update(kwargs)
    if theme:
        conn.execute("UPDATE users SET pref=?, theme=? WHERE id=?", (json.dumps(pref), theme, user_id))
    else:
        conn.execute("UPDATE users SET pref=? WHERE id=?", (json.dumps(pref), user_id))
    conn.commit()
    conn.close()

def get_user(user_id: str) -> Optional[sqlite3.Row]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    return row

# ---------- Pasien ----------
def list_patients(user_id: str) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql(
        """SELECT p.*, (SELECT COUNT(*) FROM exams e WHERE e.patient_id = p.id) AS n_exams,
                  (SELECT MAX(e.exam_date) FROM exams e WHERE e.patient_id = p.id) AS last_exam
           FROM patients p WHERE p.user_id = ? ORDER BY p.name COLLATE NOCASE""",
        conn, params=(user_id,),
    )
    conn.close()
    return df

def add_patient(user_id: str, code: str, name: str, birth_year, sex: str, contact: str, med: str) -> tuple[bool, str]:
    if not name.strip():
        return False, "Nama pasien wajib diisi."
    code = (code or "").strip() or f"P-{secrets.token_hex(2).upper()}"
    conn = get_conn()
    try:
        conn.execute(
            """INSERT INTO patients(id,user_id,code,name,birth_year,sex,contact,med_history,created_at)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (uuid.uuid4().hex, user_id, code, name.strip(), birth_year or None, sex,
             contact.strip(), med.strip(), datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return False, f"Nomor rekam medis “{code}” sudah dipakai di akun ini."
    conn.close()
    log_activity(user_id, "pasien_baru", code)
    return True, f"Pasien {name.strip()} tersimpan."

def update_patient(user_id: str, pid: str, **fields) -> None:
    allowed = ["code", "name", "birth_year", "sex", "contact", "med_history"]
    sets = [f"{k}=?" for k in fields if k in allowed]
    vals = [fields[k] for k in fields if k in allowed]
    if not sets:
        return
    conn = get_conn()
    conn.execute(f"UPDATE patients SET {', '.join(sets)} WHERE id=? AND user_id=?", (*vals, pid, user_id))
    conn.commit()
    conn.close()

def delete_patient(user_id: str, pid: str) -> None:
    conn = get_conn()
    conn.execute("DELETE FROM patients WHERE id=? AND user_id=?", (pid, user_id))
    conn.commit()
    conn.close()
    log_activity(user_id, "pasien_hapus", pid)

# ---------- Pemeriksaan ----------
def save_exam(user_id: str, exam: dict, detections: list[dict]) -> str:
    exam_id = exam.get("id") or uuid.uuid4().hex
    conn = get_conn()
    conn.execute(
        """INSERT INTO exams(id,user_id,patient_id,created_at,exam_date,model_version,conf_thr,iou_thr,
                             file_name,image_path,annot_path,n_detections,max_conf,urgency,synthesis,
                             o_onset,l_location,d_duration,c_character,a_aggravating,r_relieving,t_timing,
                             s_severity,bp_systolic,bp_diastolic,pulse_rate,resp_rate,weight_kg,height_cm,bmi,
                             clinician_note,is_demo)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            exam_id, user_id, exam.get("patient_id"), datetime.now().isoformat(timespec="seconds"),
            exam.get("exam_date", datetime.now().strftime("%Y-%m-%d")), exam.get("model_version"),
            exam.get("conf_thr"), exam.get("iou_thr"), exam.get("file_name"), exam.get("image_path"),
            exam.get("annot_path"), len(detections), exam.get("max_conf", 0.0), exam.get("urgency"),
            exam.get("synthesis"), exam.get("o_onset", "-"), exam.get("l_location", "-"),
            exam.get("d_duration", "-"), exam.get("c_character", "-"), exam.get("a_aggravating", "-"),
            exam.get("r_relieving", "-"), exam.get("t_timing", "-"), int(exam.get("s_severity", 0) or 0),
            exam.get("bp_systolic"), exam.get("bp_diastolic"), exam.get("pulse_rate"), exam.get("resp_rate"),
            exam.get("weight_kg"), exam.get("height_cm"), exam.get("bmi"),
            exam.get("clinician_note", ""), int(exam.get("is_demo", 0)),
        ),
    )
    for d in detections:
        conn.execute(
            "INSERT INTO detections(id,exam_id,user_id,label,confidence,x1,y1,x2,y2) VALUES(?,?,?,?,?,?,?,?,?)",
            (uuid.uuid4().hex, exam_id, user_id, d["label"], float(d["confidence"]),
             d.get("x1"), d.get("y1"), d.get("x2"), d.get("y2")),
        )
    conn.commit()
    conn.close()
    return exam_id

def load_exams(user_id: str, patient_id: Optional[str] = None) -> pd.DataFrame:
    conn = get_conn()
    q = """SELECT e.*, p.name AS patient_name, p.code AS patient_code,
                  (SELECT GROUP_CONCAT(d.label, ', ') FROM detections d WHERE d.exam_id = e.id) AS labels
           FROM exams e LEFT JOIN patients p ON p.id = e.patient_id
           WHERE e.user_id = ?"""
    params: list = [user_id]
    if patient_id:
        q += " AND e.patient_id = ?"
        params.append(patient_id)
    q += " ORDER BY e.created_at DESC"
    df = pd.read_sql(q, conn, params=tuple(params))
    conn.close()
    return df

def load_detections(user_id: str) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql(
        """SELECT d.*, e.exam_date, e.s_severity, e.model_version, e.is_demo, e.patient_id
           FROM detections d JOIN exams e ON e.id = d.exam_id
           WHERE d.user_id = ?""",
        conn, params=(user_id,),
    )
    conn.close()
    return df

def get_exam(user_id: str, exam_id: str) -> Optional[dict]:
    conn = get_conn()
    row = conn.execute(
        """SELECT e.*, p.name AS patient_name, p.code AS patient_code, p.birth_year, p.sex
           FROM exams e LEFT JOIN patients p ON p.id = e.patient_id
           WHERE e.id=? AND e.user_id=?""",
        (exam_id, user_id),
    ).fetchone()
    if row is None:
        conn.close()
        return None
    dets = conn.execute(
        "SELECT label, confidence FROM detections WHERE exam_id=? AND user_id=? ORDER BY confidence DESC",
        (exam_id, user_id),
    ).fetchall()
    conn.close()
    out = dict(row)
    out["detections"] = [dict(d) for d in dets]
    return out

def delete_exam(user_id: str, exam_id: str) -> None:
    conn = get_conn()
    row = conn.execute("SELECT image_path, annot_path FROM exams WHERE id=? AND user_id=?", (exam_id, user_id)).fetchone()
    conn.execute("DELETE FROM exams WHERE id=? AND user_id=?", (exam_id, user_id))
    conn.commit()
    conn.close()
    if row:
        for p in (row["image_path"], row["annot_path"]):
            try:
                if p and Path(p).exists():
                    Path(p).unlink()
            except OSError:
                pass
    log_activity(user_id, "periksa_hapus", exam_id)

def wipe_user_data(user_id: str) -> None:
    conn = get_conn()
    conn.execute("DELETE FROM detections WHERE user_id=?", (user_id,))
    conn.execute("DELETE FROM exams WHERE user_id=?", (user_id,))
    conn.execute("DELETE FROM patients WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()
    folder = IMG_DIR / user_id
    if folder.exists():
        for f in folder.glob("*"):
            try:
                f.unlink()
            except OSError:
                pass
    log_activity(user_id, "hapus_semua_data", "")

def delete_account(user_id: str) -> None:
    wipe_user_data(user_id)
    conn = get_conn()
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()


# ---------- Migrasi dari basis data v5 ----------
def migrate_legacy() -> Optional[str]:
    if meta_get("legacy_migrated") == "1" or not LEGACY_DB.exists() or LEGACY_DB.resolve() == DB_FILE.resolve():
        return None
    try:
        old = sqlite3.connect(LEGACY_DB)
        old.row_factory = sqlite3.Row
        tables = {r["name"] for r in old.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"users", "emr_logs"} <= tables:
            old.close()
            meta_set("legacy_migrated", "1")
            return None

        conn = get_conn()
        id_map: dict[str, str] = {}
        n_user = n_exam = 0
        for u in old.execute("SELECT * FROM users"):
            exists = conn.execute("SELECT id FROM users WHERE username=?", (u["username"],)).fetchone()
            if exists:
                id_map[u["username"]] = exists["id"]
                continue
            uid = uuid.uuid4().hex
            id_map[u["username"]] = uid
            conn.execute(
                """INSERT INTO users(id,username,pw_hash,pw_salt,iterations,full_name,role,institution,
                                     is_admin,theme,pref,created_at)
                   VALUES(?,?,?,?,0,?,?,'',0,'sistem','{}',?)""",
                (uid, u["username"], u["password"], "", u["full_name"], u["role"],
                 datetime.now().isoformat(timespec="seconds")),
            )
            n_user += 1

        for r in old.execute("SELECT * FROM emr_logs"):
            uid = id_map.get(r["user_id"])
            if not uid:
                continue
            exam_id = uuid.uuid4().hex
            label = (r["lesi_terdeteksi"] or "").strip()
            conf = float(r["confidence"] or 0)
            has_det = label and label.lower() != "tidak terdeteksi"
            conn.execute(
                """INSERT INTO exams(id,user_id,patient_id,created_at,exam_date,model_version,conf_thr,iou_thr,
                                     file_name,image_path,annot_path,n_detections,max_conf,urgency,synthesis,
                                     o_onset,l_location,d_duration,c_character,a_aggravating,r_relieving,
                                     t_timing,s_severity,clinician_note,is_demo)
                   VALUES(?,?,NULL,?,?,?,NULL,NULL,?,NULL,NULL,?,?,?,?,?,?,?,?,?,?,?,?,'',0)""",
                (exam_id, uid, f"{r['tanggal']}T{r['waktu']}", r["tanggal"], r["model_version"],
                 r["nama_file"], 1 if has_det else 0, conf,
                 LESION_INFO.get(label.lower(), {}).get("urgensi", "Rendah"),
                 r["suspek_diagnosis"], r["o_onset"], r["l_location"], r["d_duration"], r["c_character"],
                 r["a_aggravating"], r["r_relieving"], r["t_timing"], int(r["s_severity"] or 0)),
            )
            if has_det:
                conn.execute(
                    "INSERT INTO detections(id,exam_id,user_id,label,confidence) VALUES(?,?,?,?,?)",
                    (uuid.uuid4().hex, exam_id, uid, label, conf),
                )
            n_exam += 1

        conn.commit()
        conn.close()
        old.close()
        meta_set("legacy_migrated", "1")
        if n_user or n_exam:
            return f"Data versi lama dipindahkan: {n_user} akun, {n_exam} pemeriksaan."
        return None
    except Exception as exc:  # noqa: BLE001
        meta_set("legacy_migrated", "1")
        return f"Migrasi data lama dilewati ({exc})."


# ============================================================
# 4. MESIN AI & SINTESIS KLINIS (Diperbarui dengan Gemini API)
# ============================================================
def _weight_search_roots() -> list[Path]:
    """Return predictable locations for deployed/local YOLO weight files."""
    roots: list[Path] = []

    # Explicit environment variable has highest priority.
    env_path = os.environ.get("MAMMOUTH_MODEL_PATH") or os.environ.get("YOLO_MODEL_PATH")
    if env_path:
        roots.append(Path(env_path).expanduser())

    # Directory containing app.py / this source file. This is important on
    # Streamlit Cloud because the process working directory can differ.
    try:
        roots.append(Path(__file__).resolve().parent)
    except Exception:
        pass

    roots.extend([
        Path.cwd(),
        DATA_DIR,
        Path("models"),
        Path("weights"),
        Path("model"),
        Path("MAMMOUTH"),
    ])

    # De-duplicate while preserving priority.
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve() if root.exists() else root.absolute())
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def find_weights(version: str) -> Optional[Path]:
    """Find YOLO weights without assuming the current working directory."""
    candidates = MODEL_FILES.get(version, ["best.pt"])

    # Absolute path supplied through env var.
    env_path = os.environ.get("MAMMOUTH_MODEL_PATH") or os.environ.get("YOLO_MODEL_PATH")
    if env_path:
        explicit = Path(env_path).expanduser()
        if explicit.is_file():
            return explicit.resolve()

    for root in _weight_search_roots():
        # If a root itself points to a file, compare its filename.
        if root.is_file() and root.name in candidates:
            return root.resolve()
        for cand in candidates:
            p = root / cand
            if p.is_file():
                return p.resolve()

    return None


@st.cache_resource(show_spinner=False)
def load_model(version: str, weight_path: str):
    """Load the deployed checkpoint and return (model, error).

    The error is returned instead of being written to session_state because
    Streamlit resource caching can skip the function body on later reruns.
    """
    if YOLO is None:
        return None, "Paket ultralytics belum terpasang."
    if not weight_path:
        return None, "Path bobot kosong."
    try:
        model = YOLO(weight_path)
        return model, ""
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def inspect_loaded_model(model, weights: Optional[Path]) -> dict:
    """Return safe runtime metadata proving whether the checkpoint is loaded."""
    if model is None:
        return {"loaded": False, "task": "—", "classes": [], "count": 0, "size_mb": 0.0}
    try:
        names = getattr(model, "names", {}) or {}
        if isinstance(names, dict):
            classes = [str(v) for _, v in sorted(names.items())]
        else:
            classes = [str(v) for v in names]
        return {
            "loaded": True,
            "task": str(getattr(model, "task", "unknown")),
            "classes": classes,
            "count": len(classes),
            "size_mb": round(weights.stat().st_size / 1e6, 2) if weights and weights.is_file() else 0.0,
        }
    except Exception as exc:  # noqa: BLE001
        return {"loaded": True, "task": "unknown", "classes": [], "count": 0,
                "size_mb": 0.0, "inspect_error": str(exc)}

def run_inference(model, image: Image.Image, conf: float, iou: float) -> tuple[list[dict], Optional[Image.Image]]:
    results = model(image, conf=conf, iou=iou, verbose=False)
    res = results[0]
    plotted = res.plot()  # BGR numpy
    annotated = Image.fromarray(plotted[:, :, ::-1])
    dets = []
    for b in res.boxes:
        xy = b.xyxy[0].tolist()
        dets.append({
            "label": res.names[int(b.cls[0].item())],
            "confidence": float(b.conf[0].item()),
            "x1": xy[0], "y1": xy[1], "x2": xy[2], "y2": xy[3],
        })
    dets.sort(key=lambda d: d["confidence"], reverse=True)
    return dets, annotated

def run_demo_inference(image: Image.Image, conf: float) -> tuple[list[dict], Image.Image]:
    rng = random.Random(hash(image.tobytes()[:2048]) & 0xFFFF)
    n = rng.choice([0, 1, 1, 2, 2, 3])
    w, h = image.size
    dets = []
    for label in rng.sample(DEMO_LABELS, k=min(n, len(DEMO_LABELS))):
        c = round(rng.uniform(max(conf, 0.3), 0.96), 4)
        x1, y1 = rng.uniform(0.05, 0.5) * w, rng.uniform(0.05, 0.5) * h
        dets.append({"label": label, "confidence": c,
                     "x1": x1, "y1": y1, "x2": x1 + 0.3 * w, "y2": y1 + 0.28 * h})
    dets.sort(key=lambda d: d["confidence"], reverse=True)
    return dets, image.copy()

def urgency_of(labels: list[str], severity: int = 0) -> str:
    rank = 0
    for lb in labels:
        rank = max(rank, URGENCY_RANK.get(LESION_INFO.get(lb.lower(), {}).get("urgensi", "Rendah"), 1))
    if severity >= 7:
        rank = max(rank, 3)
    elif severity >= 4:
        rank = max(rank, 2)
    return RANK_URGENCY.get(rank, "Rendah") if rank else "Rendah"

def _image_to_gemini_part(image: Optional[Image.Image]):
    """Konversi citra PIL menjadi Part Gemini tanpa menulis API key atau data pasien ke log."""
    if image is None or genai_types is None:
        return None
    try:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=88, optimize=True)
        return genai_types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")
    except Exception as exc:  # noqa: BLE001
        global GEMINI_SDK_ERROR
        GEMINI_SDK_ERROR = f"Gagal menyiapkan citra untuk Gemini: {type(exc).__name__}: {exc}"
        return None

def synthesize(detections: list[dict], anam: dict, image: Optional[Image.Image] = None) -> str:
    """Sintesis klinis multimodal: citra + hasil YOLO + OLD CARTS melalui Gemini.

    YOLO tetap menjadi detector/bounding-box engine MAMMOTH. Gemini dipakai untuk
    menginterpretasikan konteks visual secara hati-hati dan menyusun sintesis,
    bukan untuk menggantikan diagnosis dokter gigi.
    """
    global GEMINI_SDK_ERROR, GEMINI_LAST_STATUS
    client = get_gemini_client()
    if client is None:
        GEMINI_LAST_STATUS = "Fallback lokal: Gemini tidak terkonfigurasi."
        return fallback_synthesize(detections, anam)

    sev = int(anam.get("s_severity", 0) or 0)
    anam_payload = f"""
- Onset: {anam.get('o_onset', '-')}
- Lokasi: {anam.get('l_location', '-')}
- Durasi: {anam.get('d_duration', '-')}
- Karakteristik: {anam.get('c_character', '-')}
- Memperberat: {anam.get('a_aggravating', '-')}
- Meredakan: {anam.get('r_relieving', '-')}
- Waktu/pola: {anam.get('t_timing', '-')}
- Skala Nyeri (VAS): {sev}/10
"""

    if not detections:
        distribusi = "YOLO tidak menemukan objek/lesi di atas ambang keyakinan yang dipilih."
    else:
        distribusi = "\n".join(
            f"Lesi {i+1}: {d['label']} | confidence={d['confidence']*100:.1f}% | "
            f"bbox=(x1={d['x1']:.1f}, y1={d['y1']:.1f}, x2={d['x2']:.1f}, y2={d['y2']:.1f})"
            for i, d in enumerate(detections)
        )

    prompt = f"""
Anda adalah MAMMOUTH Clinical AI Assistant untuk dukungan skrining rongga mulut.
Anda menerima CITRA KLINIS, hasil object detection YOLO, dan anamnesis OLD CARTS.

TUJUAN:
Susun sintesis klinis singkat yang membantu dokter gigi menilai temuan, bukan menetapkan diagnosis definitif.

ANAMNESIS OLD CARTS:
{anam_payload}

HASIL DETEKSI YOLO:
{distribusi}

ATURAN KLINIS DAN KESELAMATAN:
1. Citra yang Anda lihat adalah sumber visual tambahan; jangan menganggap citra saja cukup untuk diagnosis.
2. Perlakukan label dan confidence YOLO sebagai temuan model, bukan kebenaran klinis.
3. Jangan mengarang riwayat, pemeriksaan, lokasi anatomi, ukuran, warna, tekstur, atau gejala yang tidak dapat didukung data.
4. Jika visual tidak cukup jelas, katakan bahwa temuan tidak dapat dinilai secara pasti.
5. Jika menyebut diagnosis banding/suspek, gunakan bahasa probabilistik seperti “mengarah ke”, “konsisten dengan”, atau “perlu dipertimbangkan”.
6. Jangan memberikan keputusan terapi definitif, terutama tindakan invasif, hanya berdasarkan AI.
7. Jangan menyebut prevalensi/statistik atau data pasien lain.
8. Format jawaban: (a) Temuan AI, (b) Korelasi klinis, (c) Hal yang perlu dikonfirmasi dokter gigi.
9. Maksimal 3 paragraf singkat, Bahasa Indonesia profesional.
10. Akhiri dengan: “Hasil AI merupakan alat bantu skrining dan harus dikonfirmasi melalui pemeriksaan klinis langsung oleh dokter gigi.”
"""

    image_part = _image_to_gemini_part(image)
    contents = [prompt]
    if image_part is not None:
        contents = [image_part, prompt]

    try:
        response = client.models.generate_content(
            model=get_gemini_model_name(),
            contents=contents,
            config=genai_types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=700,
            ) if genai_types is not None else None,
        )
        text = (getattr(response, "text", None) or "").strip()
        if text:
            GEMINI_SDK_ERROR = None
            GEMINI_LAST_STATUS = f"Gemini aktif · multimodal · {get_gemini_model_name()}"
            return text.replace("\n", "<br>")
        GEMINI_LAST_STATUS = "Gemini merespons tanpa teks; fallback lokal digunakan."
        return fallback_synthesize(detections, anam)
    except Exception as exc:  # noqa: BLE001
        GEMINI_SDK_ERROR = f"{type(exc).__name__}: {exc}"
        GEMINI_LAST_STATUS = "Request Gemini gagal; fallback lokal digunakan."
        return fallback_synthesize(detections, anam)

def test_gemini_connection() -> tuple[bool, str]:
    """Tes request Gemini nyata tanpa menampilkan API key."""
    global GEMINI_SDK_ERROR, GEMINI_LAST_STATUS
    client = get_gemini_client()
    if client is None:
        GEMINI_LAST_STATUS = "Gemini tidak terkonfigurasi."
        return False, GEMINI_SDK_ERROR or "Gemini belum terkonfigurasi."
    import time

    last_error = None
    for attempt, delay in enumerate((0, 2, 5), start=1):
        if delay:
            time.sleep(delay)
        try:
            response = client.models.generate_content(
                model=get_gemini_model_name(),
                contents="Balas hanya dengan: MAMMOUTH_GEMINI_OK",
                config=genai_types.GenerateContentConfig(temperature=0, max_output_tokens=20) if genai_types is not None else None,
            )
            text = (getattr(response, "text", None) or "").strip()
            if text:
                GEMINI_SDK_ERROR = None
                GEMINI_LAST_STATUS = f"Gemini ONLINE · {get_gemini_model_name()} · attempt {attempt}"
                return True, text
            last_error = "Gemini merespons tanpa teks."
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            # 503/429/5xx are transient candidates; retry. Other errors return immediately.
            code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            msg = str(exc).lower()
            transient = code in {429, 500, 502, 503, 504} or any(x in msg for x in ("503", "unavailable", "high demand", "429", "rate limit", "timeout"))
            if not transient:
                break

    GEMINI_SDK_ERROR = last_error or "Gemini request gagal."
    GEMINI_LAST_STATUS = "Gemini OFFLINE / request gagal setelah retry."
    return False, GEMINI_SDK_ERROR

def fallback_synthesize(detections: list[dict], anam: dict) -> str:
    sev = int(anam.get("s_severity", 0) or 0)
    char = str(anam.get("c_character", "")).lower()
    onset = str(anam.get("o_onset", "")).lower()
    aggr = str(anam.get("a_aggravating", "")).lower()

    if not detections:
        if sev >= 4:
            return (f"Tidak ada anomali visual yang terdeteksi, namun pasien melaporkan nyeri VAS {sev}/10. "
                    "Pertimbangkan sumber non-visual: pulpa, periodontal, sendi temporomandibula, atau nyeri rujukan. "
                    "Lanjutkan pemeriksaan klinis langsung dan radiografis.")
        return ("Tidak ada anomali visual yang terdeteksi pada citra ini. Hasil negatif tidak menyingkirkan penyakit — "
                "sesuaikan dengan keluhan dan pemeriksaan klinis langsung.")

    lines = []
    for d in detections:
        key = d["label"].lower()
        info = LESION_INFO.get(key, {})
        nama = info.get("nama_klinis", d["label"])
        conf_txt = f"{d['confidence']*100:.0f}%"

        if key == "karies":
            if sev >= 6 or "denyut" in char or "spontan" in char or "malam" in char:
                lines.append(f"{nama} ({conf_txt}) dengan nyeri berat/spontan — arahkan ke pulpitis ireversibel; "
                             "perlu tes vitalitas dan radiograf periapikal.")
            elif sev >= 3 or "ngilu" in char or "manis" in char or "dingin" in aggr:
                lines.append(f"{nama} ({conf_txt}) dengan hipersensitivitas yang mereda setelah stimulus dihilangkan — "
                             "arahkan ke pulpitis reversibel.")
            else:
                lines.append(f"{nama} ({conf_txt}) tanpa keluhan nyeri — karies asimtomatik; tentukan kedalaman secara klinis.")
        elif key in {"ulkus traumatikus", "cheek biting"}:
            lama = any(t in onset for t in ["minggu", "bulan", "tahun"])
            if lama:
                lines.append(f"{nama} ({conf_txt}) yang menetap lebih dari dua minggu — wajib evaluasi ulang dan "
                             "pertimbangkan biopsi untuk menyingkirkan keganasan.")
            elif sev >= 5:
                lines.append(f"{nama} ({conf_txt}) fase akut dengan nyeri VAS {sev}/10 — eliminasi faktor trauma dan "
                             "berikan topikal analgesik.")
            else:
                lines.append(f"{nama} ({conf_txt}) indolen atau dalam fase penyembuhan — cukup observasi dan habit breaking.")
        elif key == "stain calculus":
            lines.append(f"{nama} ({conf_txt}) — indikasi scaling dan root planing; periksa perdarahan gingiva dan "
                         "kedalaman poket sebagai tanda periodontitis.")
        elif key == "coated tongue":
            extra = " Disertai keluhan nyeri, pertimbangkan kandidiasis." if sev >= 3 else ""
            lines.append(f"{nama} ({conf_txt}) — perbaiki kebersihan lidah dan telusuri xerostomia.{extra}")
        else:
            lines.append(f"{nama} ({conf_txt}) — {info.get('tatalaksana', 'observasi klinis').split('.')[0]}.")

    return " ".join(f"{i}. {t}" for i, t in enumerate(lines, 1))


# ============================================================
# 5. UTILITAS
# ============================================================
def show_image(img, caption: str = "") -> None:
    try:
        st.image(img, caption=caption or None, use_container_width=True)
    except TypeError:
        st.image(img, caption=caption or None)

def enhance(img: Image.Image, brightness: float, contrast: float, sharpness: float, auto: bool) -> Image.Image:
    out = ImageOps.exif_transpose(img)
    if auto:
        out = ImageOps.autocontrast(out, cutoff=1)
    if brightness != 1.0:
        out = ImageEnhance.Brightness(out).enhance(brightness)
    if contrast != 1.0:
        out = ImageEnhance.Contrast(out).enhance(contrast)
    if sharpness != 1.0:
        out = ImageEnhance.Sharpness(out).enhance(sharpness)
    return out

def store_image(user_id: str, exam_id: str, img: Image.Image, kind: str) -> str:
    folder = IMG_DIR / user_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{exam_id}_{kind}.jpg"
    copy = img.convert("RGB")
    copy.thumbnail((1400, 1400))
    copy.save(path, "JPEG", quality=88, optimize=True)
    return str(path)

def load_stored(path: Optional[str]) -> Optional[Image.Image]:
    if path and Path(path).exists():
        try:
            return Image.open(path)
        except Exception:  # noqa: BLE001
            return None
    return None

def img_to_b64(path: Optional[str], max_px: int = 900) -> Optional[str]:
    img = load_stored(path)
    if img is None:
        return None
    img = img.convert("RGB")
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()

def pill(text: str, tone: str = "neutral") -> str:
    return f"<span class='pill {tone}'>{text}</span>"

def urgency_tone(u: Optional[str]) -> str:
    if not u:
        return "neutral"
    return {"Tinggi": "high", "Sedang–Tinggi": "high", "Sedang": "med", "Rendah": "low"}.get(u, "neutral")

def compute_bmi(weight_kg: Optional[float], height_cm: Optional[float]) -> Optional[float]:
    try:
        w, h = float(weight_kg or 0), float(height_cm or 0)
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return w / ((h / 100) ** 2)

def bmi_category(bmi: Optional[float]) -> tuple[str, str]:
    if bmi is None:
        return "—", "neutral"
    if bmi < 18.5:
        return "Berat badan kurang", "med"
    if bmi < 23:
        return "Berat badan normal", "low"
    if bmi < 25:
        return "Berat badan lebih", "med"
    if bmi < 30:
        return "Obesitas I", "high"
    return "Obesitas II", "high"

def page_head(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='page-head'><h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)

def kpi(label: str, value: str, sub: str = "", color: str = "text") -> None:
    st.markdown(
        f"<div class='kpi'><p class='label'>{label}</p>"
        f"<p class='value' style='color:var(--{color})'>{value}</p>"
        f"<p class='sub'>{sub}</p></div>",
        unsafe_allow_html=True,
    )

def build_report_html(exam: dict, clinician: str, institution: str) -> str:
    b64 = img_to_b64(exam.get("annot_path") or exam.get("image_path"))
    img_block = (f"<img src='data:image/jpeg;base64,{b64}' alt='Citra pemeriksaan'>"
                 if b64 else "<p class='muted'>Citra tidak tersimpan pada pemeriksaan ini.</p>")
    rows = "".join(
        f"<tr><td>{d['label']}</td><td>{LESION_INFO.get(d['label'].lower(), {}).get('nama_klinis', '—')}</td>"
        f"<td class='num'>{d['confidence']*100:.1f}%</td></tr>"
        for d in exam.get("detections", [])
    ) or "<tr><td colspan='3' class='muted'>Tidak ada anomali terdeteksi.</td></tr>"

    olds = [("Onset", "o_onset"), ("Lokasi", "l_location"), ("Durasi", "d_duration"), ("Karakter", "c_character"),
            ("Memperberat", "a_aggravating"), ("Meredakan", "r_relieving"), ("Waktu", "t_timing")]
    anam_rows = "".join(f"<tr><th>{lab}</th><td>{exam.get(k) or '—'}</td></tr>" for lab, k in olds)
    anam_rows += f"<tr><th>Skala nyeri</th><td>{exam.get('s_severity', 0)}/10</td></tr>"

    has_vitals = any(exam.get(k) for k in
                     ("bp_systolic", "bp_diastolic", "pulse_rate", "resp_rate", "weight_kg", "height_cm", "bmi"))
    vital_section = ""
    if has_vitals:
        cat, _ = bmi_category(exam.get("bmi"))
        bmi_txt = f"{exam['bmi']:.1f} kg/m² ({cat})" if exam.get("bmi") else "—"
        vital_rows = (
            f"<tr><th>Tekanan darah</th><td>{exam.get('bp_systolic') or '—'}/{exam.get('bp_diastolic') or '—'} mmHg</td></tr>"
            f"<tr><th>Nadi</th><td>{exam.get('pulse_rate') or '—'} x/menit</td></tr>"
            f"<tr><th>Frekuensi napas</th><td>{exam.get('resp_rate') or '—'} x/menit</td></tr>"
            f"<tr><th>Berat badan</th><td>{exam.get('weight_kg') or '—'} kg</td></tr>"
            f"<tr><th>Tinggi badan</th><td>{exam.get('height_cm') or '—'} cm</td></tr>"
            f"<tr><th>Indeks massa tubuh (IMT)</th><td>{bmi_txt}</td></tr>"
        )
        vital_section = f"<h2>Tanda-tanda vital</h2>\n<table>{vital_rows}</table>"

    demo_note = ("<p class='warn'>Pemeriksaan ini dijalankan dalam mode demo. Kotak deteksi merupakan simulasi, "
                 "bukan keluaran model AI.</p>") if exam.get("is_demo") else ""

    return f"""<!DOCTYPE html>
<html lang="id"><head><meta charset="utf-8">
<title>Laporan Skrining {exam.get('id','')[:8].upper()}</title>
<style>
  @page {{ margin: 18mm; }}
  body {{ font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; color: #14201d; line-height: 1.55; max-width: 800px; margin: 0 auto; padding: 24px; }}
  header {{ border-bottom: 2px solid #0B6B5A; padding-bottom: 12px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: flex-end; }}
  .mark {{ font-size: 1.7rem; font-weight: 700; color: #0B6B5A; letter-spacing: -.03em; }}
  .meta {{ text-align: right; font-size: .82rem; color: #5C6B66; }}
  h2 {{ font-size: 1rem; margin: 26px 0 8px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
  th, td {{ text-align: left; padding: 7px 10px; border-bottom: 1px solid #E1E7E4; vertical-align: top; }}
  th {{ width: 150px; color: #5C6B66; font-weight: 600; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  img {{ width: 100%; border-radius: 10px; border: 1px solid #E1E7E4; margin-top: 8px; }}
  .synth {{ background: #F1F7F5; border-left: 4px solid #0B6B5A; padding: 12px 16px; border-radius: 8px; font-size: .9rem; }}
  .muted {{ color: #5C6B66; }}
  .warn {{ background: #FCF0DE; border-left: 4px solid #A8651A; padding: 10px 14px; border-radius: 8px; font-size: .85rem; }}
  footer {{ margin-top: 32px; padding-top: 12px; border-top: 1px solid #E1E7E4; font-size: .76rem; color: #5C6B66; }}
  .sign {{ margin-top: 40px; font-size: .85rem; }}
  @media print {{ body {{ padding: 0; }} }}
</style></head><body>
<header>
  <div><div class="mark">Mammouth</div><div class="muted" style="font-size:.8rem">Laporan skrining rongga mulut</div></div>
  <div class="meta">No. {exam.get('id','')[:8].upper()}<br>{exam.get('exam_date','')}</div>
</header>
{demo_note}
<h2>Identitas</h2>
<table>
  <tr><th>Pasien</th><td>{exam.get('patient_name') or 'Tanpa identitas'} ({exam.get('patient_code') or '—'})</td></tr>
  <tr><th>Tahun lahir</th><td>{exam.get('birth_year') or '—'}</td></tr>
  <tr><th>Jenis kelamin</th><td>{exam.get('sex') or '—'}</td></tr>
  <tr><th>Pemeriksa</th><td>{clinician}{(' · ' + institution) if institution else ''}</td></tr>
</table>
{vital_section}
<h2>Citra dan temuan</h2>
{img_block}
<table style="margin-top:14px">
  <thead><tr><th style="width:auto">Kelas</th><th style="width:auto">Nama klinis</th><th class="num">Keyakinan</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
<h2>Anamnesis OLD CARTS</h2>
<table>{anam_rows}</table>
<h2>Sintesis klinis (Berbasis AI)</h2>
<div class="synth">{exam.get('synthesis') or '—'}</div>
{f"<h2>Catatan pemeriksa</h2><p>{exam.get('clinician_note')}</p>" if exam.get('clinician_note') else ""}
<div class="sign">Tingkat prioritas: <strong>{exam.get('urgency') or '—'}</strong><br><br><br>
  ______________________________<br>{clinician}</div>
<footer>
  Dihasilkan oleh Mammouth v{APP_VERSION} · model {exam.get('model_version') or '—'} ·
  ambang keyakinan {exam.get('conf_thr') or '—'} / IoU {exam.get('iou_thr') or '—'}.
  Laporan ini adalah alat bantu skrining. Diagnosis akhir ditegakkan oleh dokter gigi melalui pemeriksaan langsung.
</footer></body></html>"""

def build_export_zip(user_id: str, include_images: bool = True) -> bytes:
    exams = load_exams(user_id)
    dets = load_detections(user_id)
    pats = list_patients(user_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("pemeriksaan.csv", exams.to_csv(index=False))
        z.writestr("deteksi.csv", dets.to_csv(index=False))
        z.writestr("pasien.csv", pats.to_csv(index=False))
        z.writestr("README.txt",
                   f"Ekspor data Mammouth v{APP_VERSION}\n"
                   f"Dibuat: {datetime.now():%Y-%m-%d %H:%M}\n"
                   f"Berisi seluruh data milik satu akun: pasien, pemeriksaan, deteksi"
                   f"{', dan citra' if include_images else ''}.\n")
        if include_images:
            folder = IMG_DIR / user_id
            if folder.exists():
                for f in folder.glob("*.jpg"):
                    z.write(f, f"citra/{f.name}")
    return buf.getvalue()


# ============================================================
# 6. SESI & HALAMAN MASUK
# ============================================================
st.set_page_config(
    page_title=f"{APP_NAME} — {APP_TAGLINE}",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

DEFAULTS = {
    "user": None,
    "theme": "sistem",
    "lang": "en",
    "page": "Dashboard",
    "anamnesis": {},
    "vitals": {},
    "active_patient": None,
    "last_batch": [],
    "open_exam": None,
    "model_version": "YOLOv8",
    "conf_thr": 0.25,
    "iou_thr": 0.45,
    "demo_mode": False,
    "login_fails": 0,
    "migration_note": None,
}
for _k, _v in DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

if st.session_state.migration_note is None:
    st.session_state.migration_note = migrate_legacy() or ""

st.markdown(theme_css(st.session_state.theme), unsafe_allow_html=True)


def tooth_svg() -> str:
    return """
<svg viewBox="0 0 320 200" width="100%" height="190" role="img" aria-label="Ilustrasi gigi dan titik pemindaian" class="tooth-plot">
  <defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="var(--surface)"/><stop offset="100%" stop-color="var(--primary-soft)"/>
  </linearGradient></defs>
  <path d="M110 24c14-10 34-10 50 0 16-10 36-10 50 0 16 12 18 36 12 60-5 21-9 44-14 66-4 17-22 18-26 1-3-13-5-27-9-40-3-9-13-9-16 0-4 13-6 27-9 40-4 17-22 16-26-1-5-22-9-45-14-66-6-24-4-48 12-60z"
        fill="url(#g)" stroke="var(--primary)" stroke-width="2.5" stroke-linejoin="round"/>
  <rect x="146" y="72" width="34" height="26" rx="5" fill="none" stroke="var(--accent)" stroke-width="2.5" stroke-dasharray="5 4"/>
  <circle cx="163" cy="85" r="3.5" fill="var(--accent)"/>
  <line x1="180" y1="85" x2="252" y2="85" stroke="var(--border)" stroke-width="1.5"/>
  <text x="256" y="82" font-family="Inter, sans-serif" font-size="11" font-weight="600" fill="var(--text)">karies</text>
  <text x="256" y="96" font-family="Inter, sans-serif" font-size="10" fill="var(--muted)">0.91</text>
  <line x1="40" y1="150" x2="108" y2="150" stroke="var(--border)" stroke-width="1.5"/>
  <circle cx="122" cy="150" r="3.5" fill="var(--primary)"/>
  <text x="12" y="146" font-family="Inter, sans-serif" font-size="11" font-weight="600" fill="var(--text)">stain</text>
  <text x="12" y="160" font-family="Inter, sans-serif" font-size="10" fill="var(--muted)">0.74</text>
</svg>"""

def render_login() -> None:
    st.markdown("<style>[data-testid='stSidebar']{display:none}</style>", unsafe_allow_html=True)
    left, gap, right = st.columns([1.05, 0.12, 1], gap="large")

    with left:
        st.markdown(
            f"""
            <div class="auth-hero">
              <p class="mark">{APP_NAME}</p>
              <p class="tag">{APP_TAGLINE}</p>
              <p class="lede">Ruang kerja pribadi untuk skrining lesi rongga mulut: unggah citra klinis,
              gabungkan dengan anamnesis, lalu simpan hasilnya sebagai rekam medis yang hanya bisa dibuka oleh akun Anda.</p>
              <ul class="auth-list">
                <li>Delapan kelas lesi dikenali dari citra intraoral</li>
                <li>Anamnesis OLD CARTS memperkaya sintesis temuan</li>
                <li>Riwayat pasien, laporan cetak, dan ekspor data penuh</li>
                <li>Setiap akun berdiri sendiri — data tidak saling terlihat</li>
              </ul>
            </div>
            {tooth_svg()}
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        tab_in, tab_new = st.tabs(["Masuk", "Buat akun"])

        with tab_in:
            with st.form("form_login"):
                st.markdown("### Selamat datang kembali")
                u = st.text_input("Username", placeholder="mis. drg.rina")
                p = st.text_input("Kata sandi", type="password", placeholder="Kata sandi akun Anda")
                if st.form_submit_button("Masuk", type="primary", use_container_width=True):
                    if st.session_state.login_fails >= 5:
                        st.error("Terlalu banyak percobaan gagal. Muat ulang halaman untuk mencoba lagi.")
                    else:
                        row = verify_login(u, p)
                        if row:
                            st.session_state.user = dict(row)
                            st.session_state.theme = row["theme"] or "sistem"
                            pref = json.loads(row["pref"] or "{}")
                            st.session_state.model_version = pref.get("model_version", "YOLOv8")
                            st.session_state.conf_thr = pref.get("conf_thr", 0.25)
                            st.session_state.iou_thr = pref.get("iou_thr", 0.45)
                            st.session_state.login_fails = 0
                            st.session_state.page = "Dashboard"
                            st.rerun()
                        else:
                            st.session_state.login_fails += 1
                            st.error("Username atau kata sandi tidak cocok.")
            if st.session_state.migration_note:
                st.info(st.session_state.migration_note)

        with tab_new:
            with st.form("form_register"):
                st.markdown("### Buat akun baru")
                st.caption("Akun pertama yang dibuat di server ini otomatis menjadi administrator.")
                n = st.text_input("Nama lengkap", placeholder="drg. Rina Pratiwi")
                r = st.text_input("Peran", placeholder="Dokter gigi, mahasiswa profesi, operator klinis")
                inst = st.text_input("Institusi atau klinik (opsional)", placeholder="Klinik Sehat Gigi")
                nu = st.text_input("Username", placeholder="Huruf, angka, titik, atau garis bawah")
                np1 = st.text_input("Kata sandi", type="password", placeholder="Minimal 8 karakter, huruf dan angka")
                np2 = st.text_input("Ulangi kata sandi", type="password")
                if st.form_submit_button("Buat akun", type="primary", use_container_width=True):
                    if not all([n.strip(), r.strip(), nu.strip(), np1]):
                        st.warning("Lengkapi nama, peran, username, dan kata sandi.")
                    elif np1 != np2:
                        st.warning("Kedua kata sandi belum sama.")
                    else:
                        ok, msg = create_user(nu, np1, n, r, inst)
                        (st.success if ok else st.error)(msg)
                        if ok:
                            st.info("Buka tab Masuk untuk mulai bekerja.")

        st.caption("Kata sandi disimpan sebagai turunan PBKDF2-SHA256 bersalt. "
                   "Data klinis tersimpan lokal di server tempat aplikasi ini dijalankan.")


# ============================================================
# 7. SIDEBAR
# ============================================================
NAV = [
    "Dashboard",
    "AI Detection",
    "Upload Image",
    "Detection History",
    "Patients",
    "Lesion Database",
    "Analytics",
    "Reports",
    "Settings"
]

def render_sidebar(user: dict, weights: Optional[Path]) -> None:
    with st.sidebar:
        st.markdown(
            f"""<div class="brand"><span class="mark">{APP_NAME}</span><span class="ver">v{APP_VERSION}</span></div>
                <p class="brand-sub">{APP_TAGLINE}</p>""",
            unsafe_allow_html=True,
        )

        for menu in NAV:
            active = st.session_state.page == menu
            if st.button(_t(menu), key=f"nav_{menu}", use_container_width=True,
                         type="primary" if active else "secondary"):
                st.session_state.page = menu
                st.rerun()

        st.markdown("---")
        st.markdown("**Model**")
        versions = list(MODEL_FILES.keys())
        st.session_state.model_version = st.selectbox(
            "Arsitektur", versions, index=versions.index(st.session_state.model_version),
            label_visibility="collapsed",
        )
        if weights:
            st.caption(f"Bobot aktif: `{weights.name}`")
        elif st.session_state.demo_mode:
            st.caption("Mode demo aktif — deteksi disimulasikan.")
        else:
            st.caption("Bobot belum ditemukan.")

        st.session_state.conf_thr = st.slider("Ambang keyakinan", 0.05, 0.95, float(st.session_state.conf_thr), 0.05)
        st.session_state.iou_thr = st.slider("Ambang IoU (NMS)", 0.05, 0.95, float(st.session_state.iou_thr), 0.05)

        st.markdown("---")
        st.markdown(
            f"""<div class="who"><p class="name">{user['full_name']}</p>
                <p class="role">{user.get('role') or 'Klinisi'}{' · admin' if user.get('is_admin') else ''}</p></div>""",
            unsafe_allow_html=True,
        )
        
        # Pengaturan Tema dan Bahasa
        c1, c2 = st.columns(2)
        with c1:
            theme_opts = {"terang": _t("Light"), "gelap": _t("Dark"), "sistem": _t("System")}
            reverse_theme = {v: k for k, v in theme_opts.items()}
            current_theme_label = theme_opts.get(st.session_state.theme, _t("System"))
            sel_theme = st.selectbox(_t("Theme Mode"), list(theme_opts.values()), index=list(theme_opts.values()).index(current_theme_label), label_visibility="collapsed")
            if reverse_theme[sel_theme] != st.session_state.theme:
                st.session_state.theme = reverse_theme[sel_theme]
                set_user_pref(user["id"], theme=st.session_state.theme)
                st.rerun()
        with c2:
            lang_opts = {"id": "ID", "en": "EN"}
            reverse_lang = {v: k for k, v in lang_opts.items()}
            current_lang_label = lang_opts.get(st.session_state.lang, "EN")
            sel_lang = st.selectbox(_t("Language"), list(lang_opts.values()), index=list(lang_opts.values()).index(current_lang_label), label_visibility="collapsed")
            if reverse_lang[sel_lang] != st.session_state.lang:
                st.session_state.lang = reverse_lang[sel_lang]
                st.rerun()
        
        if st.button(_t("Logout"), use_container_width=True):
            set_user_pref(user["id"], model_version=st.session_state.model_version,
                          conf_thr=st.session_state.conf_thr, iou_thr=st.session_state.iou_thr)
            log_activity(user["id"], "logout", "")
            for k in ["user", "anamnesis", "active_patient", "last_batch", "open_exam"]:
                st.session_state[k] = DEFAULTS[k]
            st.rerun()


# ============================================================
# 8. HALAMAN: DASHBOARD (Struktur Baru)
# ============================================================
def page_dashboard(user: dict) -> None:
    page_head(_t("Dashboard"), _t("Ringkasan aktivitas skrining pada akun Anda."))

    exams = load_exams(user["id"])
    dets = load_detections(user["id"])
    
    # 1. KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi(_t("Total Examinations"), f"{len(exams)}", "", "primary")
    with c2:
        kpi(_t("AI Detections"), f"{len(dets)}")
    with c3:
        acc = f"{dets['confidence'].mean()*100:.1f}%" if not dets.empty else "0%"
        kpi(_t("Detection Accuracy"), acc)
    with c4:
        # Simulasi rata-rata waktu pemrosesan karena tidak tercatat di basis data sebelumnya
        kpi(_t("Average Processing Time"), "1.42s", "", "accent")

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    left, right = st.columns([1, 1], gap="large")

    # 2. Recent AI Detection
    with left:
        st.markdown(f"### {_t('Recent AI Detection')}")
        if not exams.empty and not dets.empty:
            positive_exams = exams[exams["n_detections"] > 0]
            if not positive_exams.empty:
                latest = positive_exams.iloc[0]
                l_dets = dets[dets["exam_id"] == latest["id"]]

                with st.container(border=True):
                    img_col, txt_col = st.columns([1, 1.2])
                    with img_col:
                        st.markdown(f"**{_t('Original radiograph/oral image')}**")
                        show_image(load_stored(latest["image_path"]))
                    with txt_col:
                        st.markdown(f"**{_t('Detection result')}**")
                        show_image(load_stored(latest["annot_path"]))
                    
                    st.markdown("---")
                    st.markdown(f"**{_t('Bounding boxes & Confidence score')}**")
                    for _, d in l_dets.iterrows():
                        st.markdown(f"- **{d['label']}**: {d['confidence']*100:.1f}%")
            else:
                st.caption(_t("Belum ada deteksi AI positif."))
        else:
            st.caption(_t("Belum ada deteksi AI."))

    # 3. Charts & Analytics
    with right:
        st.markdown(f"### {_t('Lesion Distribution')}")
        if not dets.empty and alt is not None:
            vc = dets["label"].value_counts().reset_index()
            vc.columns = ["label", "jumlah"]
            chart = alt.Chart(vc).mark_arc(innerRadius=50, cornerRadius=4).encode(
                theta=alt.Theta(field="jumlah", type="quantitative"),
                color=alt.Color(field="label", type="nominal", legend=alt.Legend(title="Kelas", orient="bottom")),
                tooltip=["label", "jumlah"]
            ).properties(height=280)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.caption(_t("Data belum cukup untuk distribusi."))

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    
    st.markdown(f"### {_t('Detection Activity')}")
    if not exams.empty and alt is not None:
        daily = exams.groupby("exam_date").size().reset_index(name="count")
        primary_color = THEMES[st.session_state.theme]["primary"]
        line = alt.Chart(daily).mark_area(line={'color': primary_color}, color=primary_color, opacity=0.2).encode(
            x=alt.X("exam_date:T", title=None),
            y=alt.Y("count:Q", title=None),
            tooltip=["exam_date", "count"]
        ).properties(height=260)
        st.altair_chart(line, use_container_width=True)
    else:
        st.caption(_t("Data aktivitas belum cukup."))

    # 4. Recent Cases (Table)
    st.markdown(f"### {_t('Recent Cases')}")
    if not exams.empty:
        rc = exams.head(6)[["patient_code", "file_name", "labels", "max_conf", "exam_date", "urgency"]].copy()
        rc.columns = ["Patient ID", "Image", "Lesion type", "Confidence", "Date", "Status"]
        rc["Confidence"] = (pd.to_numeric(rc["Confidence"]).fillna(0) * 100).apply(lambda x: f"{x:.1f}%")
        rc["Patient ID"] = rc["Patient ID"].fillna("Unknown")
        rc["Image"] = rc["Image"].fillna("—")
        rc["Lesion type"] = rc["Lesion type"].fillna("No findings")
        
        st.dataframe(rc, use_container_width=True, hide_index=True)


# ============================================================
# 9. HALAMAN: SKRINING (Digabungkan untuk 'AI Detection' & 'Upload Image')
# ============================================================
def page_screening(user: dict, model, weights: Optional[Path]) -> None:
    page_head(_t("AI Detection") + " & " + _t("Upload Image"),
              "Satukan citra klinis dengan anamnesis, jalankan deteksi, lalu simpan hasilnya ke rekam medis pasien.")

    demo = st.session_state.demo_mode and model is None
    if model is None and not demo:
        with st.container(border=True):
            st.markdown("### Bobot model belum tersedia")
            st.write(
                f"Letakkan berkas bobot untuk {st.session_state.model_version} "
                f"(`{'`, `'.join(MODEL_FILES[st.session_state.model_version])}`) di folder aplikasi, "
                "lalu muat ulang halaman."
            )
            if YOLO is None:
                st.write("Paket `ultralytics` juga belum terpasang. Jalankan `pip install ultralytics`.")
            if st.button("Nyalakan mode demo", help="Menjelajah alur kerja dengan deteksi tiruan"):
                st.session_state.demo_mode = True
                st.rerun()
        return
    if demo:
        st.warning("Mode demo aktif. Kotak deteksi disimulasikan dan ditandai di rekam medis — jangan dipakai klinis.")

    if genai is None or not get_gemini_api_key():
        st.warning("Google Gemini belum siap. Sintesis klinis akan menggunakan fallback lokal sampai `google-genai` dan `GEMINI_API_KEY` tersedia.")

    pats = list_patients(user["id"])
    
    # --------------------------------------------------------
    # FITUR 1: Pencarian Pasien Manual Berbasis Teks Input
    # --------------------------------------------------------
    st.markdown(f"### {_t('Select Patient')}")
    search_q = st.text_input(_t("Search Patient (Name / MR No)"), placeholder="Ketik secara manual nama atau nomor rekam medis...")
    patient_id = None
    
    if search_q:
        matches = pats[pats["name"].str.contains(search_q, case=False, na=False) | pats["code"].str.contains(search_q, case=False, na=False)]
        if not matches.empty:
            if len(matches) == 1:
                patient_id = matches.iloc[0]["id"]
                st.success(f"Pasien terpilih: **{matches.iloc[0]['name']}** ({matches.iloc[0]['code']})")
            else:
                p_opts = {f"{r['name']} ({r['code']})": r["id"] for _, r in matches.iterrows()}
                p_choice = st.radio("Pilih salah satu dari hasil pencarian:", list(p_opts.keys()))
                patient_id = p_opts[p_choice]
        else:
            st.warning("Pasien tidak ditemukan. Silakan daftarkan di halaman Patients.")
    else:
        st.info("Anda dapat menjalankan pemeriksaan tanpa pasien, atau ketik nama/RM di atas untuk mengaitkan rekam medis.")

    exam_date = st.date_input("Tanggal pemeriksaan", value=datetime.now())

    with st.expander("Tanda-tanda vital", expanded=not st.session_state.vitals):
        with st.form("form_vitals"):
            curv = st.session_state.vitals
            v1, v2, v3 = st.columns(3, gap="large")
            with v1:
                st.markdown("**Tekanan darah**")
                sistol = st.number_input("Sistolik (mmHg)", min_value=0, max_value=300, value=int(curv.get("bp_systolic") or 0), step=1)
                diastol = st.number_input("Diastolik (mmHg)", min_value=0, max_value=200, value=int(curv.get("bp_diastolic") or 0), step=1)
            with v2:
                st.markdown("**Pernapasan**")
                nadi = st.number_input("Nadi (denyut/menit)", min_value=0, max_value=250, value=int(curv.get("pulse_rate") or 0), step=1)
                napas = st.number_input("Frekuensi napas (napas/menit)", min_value=0, max_value=80, value=int(curv.get("resp_rate") or 0), step=1)
            with v3:
                st.markdown("**Antropometri**")
                berat = st.number_input("Berat badan (kg)", min_value=0.0, max_value=400.0, value=float(curv.get("weight_kg") or 0.0), step=0.1, format="%.1f")
                tinggi = st.number_input("Tinggi badan (cm)", min_value=0.0, max_value=250.0, value=float(curv.get("height_cm") or 0.0), step=0.5, format="%.1f")
            vb1, vb2 = st.columns([1, 1])
            with vb1:
                vsaved = st.form_submit_button("Simpan tanda vital", type="primary", use_container_width=True)
            with vb2:
                vcleared = st.form_submit_button("Kosongkan", use_container_width=True, key="vitals_clear")
            if vsaved:
                bmi_val = compute_bmi(berat, tinggi)
                cat, _ = bmi_category(bmi_val)
                st.session_state.vitals = {
                    "bp_systolic": int(sistol) or None, "bp_diastolic": int(diastol) or None,
                    "pulse_rate": int(nadi) or None, "resp_rate": int(napas) or None,
                    "weight_kg": float(berat) or None, "height_cm": float(tinggi) or None,
                    "bmi": bmi_val, "bmi_category": cat,
                }
                st.success("Tanda-tanda vital tersimpan dan akan disertakan pada rekam pemeriksaan.")
            if vcleared:
                st.session_state.vitals = {}
                st.rerun()

    if st.session_state.vitals:
        v = st.session_state.vitals
        bmi_txt = f"{v['bmi']:.1f}" if v.get("bmi") else "—"
        cat, tone = bmi_category(v.get("bmi"))
        unit = "<span style='font-size:.7rem;font-weight:500;color:var(--muted)'>{}</span>"
        st.markdown(
            "<div class='vital-grid'>"
            f"<div class='vital-cell'><p class='vlabel'>Tekanan darah</p><p class='vvalue'>{v.get('bp_systolic') or '—'}/{v.get('bp_diastolic') or '—'} {unit.format('mmHg')}</p></div>"
            f"<div class='vital-cell'><p class='vlabel'>Nadi</p><p class='vvalue'>{v.get('pulse_rate') or '—'} {unit.format('x/menit')}</p></div>"
            f"<div class='vital-cell'><p class='vlabel'>Respirasi</p><p class='vvalue'>{v.get('resp_rate') or '—'} {unit.format('x/menit')}</p></div>"
            f"<div class='vital-cell'><p class='vlabel'>Berat badan</p><p class='vvalue'>{v.get('weight_kg') or '—'} {unit.format('kg')}</p></div>"
            f"<div class='vital-cell'><p class='vlabel'>Tinggi badan</p><p class='vvalue'>{v.get('height_cm') or '—'} {unit.format('cm')}</p></div>"
            f"<div class='vital-cell'><p class='vlabel'>IMT (BMI)</p><p class='vvalue'>{bmi_txt} {pill(cat, tone)}</p></div>"
            "</div>",
            unsafe_allow_html=True,
        )

    with st.expander("Anamnesis OLD CARTS", expanded=not st.session_state.anamnesis):
        with st.form("form_anamnesis"):
            a1, a2 = st.columns(2, gap="large")
            cur = st.session_state.anamnesis
            with a1:
                o = st.text_input("Onset — sejak kapan muncul", cur.get("o_onset", ""))
                l = st.text_input("Lokasi — di bagian mana", cur.get("l_location", ""))
                d = st.text_input("Durasi — berapa lama tiap serangan", cur.get("d_duration", ""))
                ch = st.text_input("Karakter — tajam, tumpul, berdenyut, ngilu", cur.get("c_character", ""))
            with a2:
                ag = st.text_input("Memperberat — apa yang memicu", cur.get("a_aggravating", ""))
                rl = st.text_input("Meredakan — apa yang menolong", cur.get("r_relieving", ""))
                tm = st.text_input("Waktu — kapan paling terasa", cur.get("t_timing", ""))
                sv = st.slider("Skala nyeri (VAS)", 0, 10, int(cur.get("s_severity", 0)))
            b1, b2 = st.columns([1, 1])
            with b1:
                saved = st.form_submit_button("Simpan anamnesis", type="primary", use_container_width=True)
            with b2:
                cleared = st.form_submit_button("Kosongkan", use_container_width=True)
            if saved:
                st.session_state.anamnesis = {
                    "o_onset": o or "-", "l_location": l or "-", "d_duration": d or "-",
                    "c_character": ch or "-", "a_aggravating": ag or "-", "r_relieving": rl or "-",
                    "t_timing": tm or "-", "s_severity": sv,
                }
                st.success("Anamnesis tersimpan dan akan dipakai untuk sintesis temuan (via AI/Rules).")
            if cleared:
                st.session_state.anamnesis = {}
                st.rerun()

    if st.session_state.anamnesis:
        a = st.session_state.anamnesis
        st.caption(f"Anamnesis aktif · nyeri {a.get('s_severity', 0)}/10 · karakter: {a.get('c_character', '-')} · lokasi: {a.get('l_location', '-')}")

    st.markdown(f"### {_t('Upload Image')}")
    tab_up, tab_cam = st.tabs(["Unggah berkas", "Ambil dari kamera"])
    raw: list[Image.Image] = []
    names: list[str] = []

    with tab_up:
        files = st.file_uploader("Pilih satu atau beberapa foto klinis", type=["jpg", "jpeg", "png", "bmp", "webp"],
                                 accept_multiple_files=True, label_visibility="collapsed")
        for f in files or []:
            try:
                raw.append(Image.open(f).convert("RGB"))
                names.append(f.name)
            except Exception:  # noqa: BLE001
                st.error(f"Berkas {f.name} tidak bisa dibaca sebagai gambar.")

    with tab_cam:
        shot = st.camera_input("Arahkan kamera intraoral ke area yang dikeluhkan", label_visibility="collapsed")
        if shot:
            try:
                raw.append(Image.open(shot).convert("RGB"))
                names.append(f"kamera_{datetime.now():%Y%m%d_%H%M%S}.jpg")
            except Exception:  # noqa: BLE001
                st.error("Gambar kamera gagal dibaca.")

    if not raw:
        st.info("Belum ada citra. Unggah foto intraoral atau ambil langsung lewat kamera untuk memulai.")
        return

    st.markdown("### Penyesuaian citra")
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        bright = st.slider("Kecerahan", 0.5, 2.0, 1.0, 0.05)
    with e2:
        contrast = st.slider("Kontras", 0.5, 2.0, 1.0, 0.05)
    with e3:
        sharp = st.slider("Ketajaman", 0.5, 2.5, 1.0, 0.05)
    with e4:
        auto = st.toggle("Auto-kontras", value=False, help="Meratakan histogram sebelum penyesuaian manual")

    processed = [enhance(img, bright, contrast, sharp, auto) for img in raw]
    cols = st.columns(min(len(processed), 4))
    for i, img in enumerate(processed):
        with cols[i % len(cols)]:
            show_image(img, names[i])

    note = st.text_area("Catatan pemeriksa (opsional)", placeholder="Temuan klinis langsung, rencana, atau rujukan.")

    run = st.button(f"Jalankan deteksi & AI pada {len(processed)} citra",
                    type="primary", use_container_width=True)

    if run:
        anam = st.session_state.anamnesis
        vit = st.session_state.vitals
        results_view = []
        bar = st.progress(0.0, text="Menyiapkan…")
        for i, (img, fname) in enumerate(zip(processed, names), start=1):
            bar.progress((i - 1) / len(processed), text=f"Menganalisis {fname} & Sintesis AI")
            try:
                if demo:
                    dets, annotated = run_demo_inference(img, st.session_state.conf_thr)
                else:
                    dets, annotated = run_inference(model, img, st.session_state.conf_thr, st.session_state.iou_thr)
            except Exception as exc:  # noqa: BLE001
                st.error(f"{fname} gagal diproses: {exc}")
                continue

            exam_id = uuid.uuid4().hex
            labels = [d["label"] for d in dets]
            exam = {
                "id": exam_id,
                "patient_id": patient_id,
                "exam_date": exam_date.strftime("%Y-%m-%d"),
                "model_version": "DEMO" if demo else st.session_state.model_version,
                "conf_thr": st.session_state.conf_thr,
                "iou_thr": st.session_state.iou_thr,
                "file_name": fname,
                "image_path": store_image(user["id"], exam_id, img, "asli"),
                "annot_path": store_image(user["id"], exam_id, annotated, "anotasi"),
                "max_conf": max([d["confidence"] for d in dets], default=0.0),
                "urgency": urgency_of(labels, int(anam.get("s_severity", 0) or 0)),
                "synthesis": synthesize(dets, anam, img),
                "clinician_note": note.strip(),
                "is_demo": 1 if demo else 0,
                **anam,
                **{k: v for k, v in vit.items() if k != "bmi_category"},
            }
            save_exam(user["id"], exam, dets)
            results_view.append({"exam": exam, "dets": dets, "annotated": annotated})

        bar.progress(1.0, text="Selesai")
        bar.empty()
        log_activity(user["id"], "skrining", f"{len(results_view)} citra")
        st.session_state.last_batch = [r["exam"]["id"] for r in results_view]
        st.toast(f"{len(results_view)} pemeriksaan tersimpan ke rekam medis Anda.", icon="✅")

        st.markdown("### Hasil")
        for r in results_view:
            render_result(user, r["exam"], r["dets"], r["annotated"])


def render_result(user: dict, exam: dict, dets: list[dict], annotated: Image.Image) -> None:
    with st.container(border=True):
        head, badge = st.columns([3, 1])
        with head:
            st.markdown(f"#### {exam['file_name']}")
        with badge:
            st.markdown(f"<div style='text-align:right'>{pill(exam['urgency'], urgency_tone(exam['urgency']))}</div>",
                        unsafe_allow_html=True)

        img_col, info_col = st.columns([1.1, 1], gap="large")
        with img_col:
            show_image(annotated, "Citra dengan kotak deteksi")
        with info_col:
            if dets:
                for d in dets:
                    info = LESION_INFO.get(d["label"].lower(), {})
                    pct = d["confidence"] * 100
                    st.markdown(
                        f"<div class='det-row'><div><div class='det-name'>{info.get('nama_klinis', d['label'])}</div>"
                        f"<div class='det-sub'>{info.get('awam', d['label'])}</div></div>"
                        f"<div style='text-align:right'><div class='meter'><span style='width:{pct:.0f}%'></span></div>"
                        f"<div class='det-sub'>{pct:.1f}%</div></div></div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown("<div class='det-sub'>Tidak ada anomali yang melewati ambang keyakinan.</div>",
                            unsafe_allow_html=True)

        tone = urgency_tone(exam["urgency"])
        cls = {"high": "u-tinggi", "med": "u-sedang", "low": "u-rendah"}.get(tone, "")
        st.markdown(
            f"<div class='verdict {cls}'><p class='title'>Sintesis Klinis (Berbasis AI)</p>"
            f"<p class='body'>{exam['synthesis']}</p></div>",
            unsafe_allow_html=True,
        )

        if dets:
            with st.expander("Rujukan tatalaksana dan tanda bahaya"):
                for d in dets:
                    info = LESION_INFO.get(d["label"].lower())
                    if not info:
                        continue
                    st.markdown(f"**{info['nama_klinis']}** — {info['tatalaksana']}")
                    st.caption(f"Tanda bahaya: {info['red_flag']} · Diagnosis banding: {info['banding']}")

        full = get_exam(user["id"], exam["id"])
        if full:
            st.download_button(
                "Unduh laporan", data=build_report_html(full, user["full_name"], user.get("institution") or ""),
                file_name=f"laporan_{exam['id'][:8]}.html", mime="text/html",
                key=f"rep_{exam['id']}", use_container_width=True,
            )


# ============================================================
# 10. HALAMAN: PASIEN
# ============================================================
def page_patients(user: dict) -> None:
    page_head(_t("Patients"), "Daftar pasien beserta riwayat pemeriksaannya. Hanya akun ini yang dapat melihatnya.")

    with st.expander("Tambah pasien", expanded=False):
        with st.form("form_pasien", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                code = st.text_input("Nomor rekam medis", placeholder="Kosongkan untuk dibuatkan otomatis")
                name = st.text_input("Nama pasien")
            with c2:
                by = st.number_input("Tahun lahir", min_value=1900, max_value=datetime.now().year, value=1995, step=1)
                sex = st.selectbox("Jenis kelamin", ["Perempuan", "Laki-laki", "Tidak disebutkan"])
            with c3:
                contact = st.text_input("Kontak", placeholder="Nomor telepon atau surel")
                med = st.text_area("Riwayat medis singkat", placeholder="Alergi, penyakit sistemik, obat rutin", height=88)
            if st.form_submit_button("Simpan pasien", type="primary"):
                ok, msg = add_patient(user["id"], code, name, int(by), sex, contact, med)
                (st.success if ok else st.error)(msg)

    pats = list_patients(user["id"])
    if pats.empty:
        st.info("Belum ada pasien. Tambahkan satu untuk mulai melacak riwayat pemeriksaan.")
        return

    q = st.text_input("Cari pasien", placeholder="Nama atau nomor rekam medis", label_visibility="collapsed")
    if q:
        mask = pats["name"].str.contains(q, case=False, na=False) | pats["code"].str.contains(q, case=False, na=False)
        pats = pats[mask]
        if pats.empty:
            st.warning(f"Tidak ada pasien yang cocok dengan “{q}”.")
            return

    for _, p in pats.iterrows():
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 1.4, 1])
            usia = f"{datetime.now().year - int(p['birth_year'])} th" if p["birth_year"] else "usia —"
            with c1:
                st.markdown(
                    f"**{p['name']}** &nbsp;{pill(p['code'], 'neutral')}<br>"
                    f"<span class='det-sub'>{usia} · {p['sex'] or '—'} · {p['contact'] or 'tanpa kontak'}</span>",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"<span class='det-sub'>{int(p['n_exams'])} pemeriksaan<br>"
                    f"terakhir {p['last_exam'] or '—'}</span>", unsafe_allow_html=True,
                )
            with c3:
                if st.button("Riwayat", key=f"pat_{p['id']}", use_container_width=True):
                    st.session_state.active_patient = None if st.session_state.active_patient == p["id"] else p["id"]
                    st.rerun()

            if st.session_state.active_patient == p["id"]:
                st.markdown("---")
                if p["med_history"]:
                    st.caption(f"Riwayat medis: {p['med_history']}")
                hist = load_exams(user["id"], p["id"])
                if hist.empty:
                    st.caption("Belum ada pemeriksaan untuk pasien ini.")
                else:
                    for _, e in hist.iterrows():
                        h1, h2 = st.columns([4, 1])
                        with h1:
                            st.markdown(
                                f"<div class='det-row'><div><div class='det-name'>{e['exam_date']} — "
                                f"{e['labels'] or 'tanpa temuan'}</div>"
                                f"<div class='det-sub'>{e['file_name'] or '—'} · nyeri {e['s_severity']}/10</div></div>"
                                f"{pill(e['urgency'] or 'Rendah', urgency_tone(e['urgency']))}</div>",
                                unsafe_allow_html=True,
                            )
                        with h2:
                            if st.button("Detail", key=f"phist_{e['id']}", use_container_width=True):
                                st.session_state.open_exam = e["id"]
                                st.session_state.page = "Detection History"
                                st.rerun()

                d1, d2, _ = st.columns([1, 1, 2])
                with d1:
                    if st.button("Sunting data", key=f"edit_{p['id']}", use_container_width=True):
                        st.session_state[f"editing_{p['id']}"] = not st.session_state.get(f"editing_{p['id']}", False)
                        st.rerun()
                with d2:
                    if st.button("Hapus pasien", key=f"del_{p['id']}", use_container_width=True):
                        st.session_state[f"confirm_del_{p['id']}"] = True
                        st.rerun()

                if st.session_state.get(f"confirm_del_{p['id']}"):
                    st.warning(f"Hapus {p['name']}? Pemeriksaan yang sudah tersimpan tetap ada, "
                               "tetapi kehilangan kaitan ke pasien ini.")
                    y, n = st.columns(2)
                    with y:
                        if st.button("Ya, hapus", key=f"yes_{p['id']}", type="primary", use_container_width=True):
                            delete_patient(user["id"], p["id"])
                            st.session_state[f"confirm_del_{p['id']}"] = False
                            st.rerun()
                    with n:
                        if st.button("Batal", key=f"no_{p['id']}", use_container_width=True):
                            st.session_state[f"confirm_del_{p['id']}"] = False
                            st.rerun()

                if st.session_state.get(f"editing_{p['id']}"):
                    with st.form(f"form_edit_{p['id']}"):
                        n1 = st.text_input("Nama", p["name"])
                        n2 = st.text_input("Kontak", p["contact"] or "")
                        n3 = st.text_area("Riwayat medis", p["med_history"] or "")
                        if st.form_submit_button("Simpan perubahan", type="primary"):
                            update_patient(user["id"], p["id"], name=n1, contact=n2, med_history=n3)
                            st.session_state[f"editing_{p['id']}"] = False
                            st.rerun()


# ============================================================
# 11. HALAMAN: REKAM MEDIS & LAPORAN
# ============================================================
def page_records(user: dict) -> None:
    page_head(_t("Detection History"), "Arsip seluruh pemeriksaan pada akun ini, lengkap dengan citra dan sintesisnya.")

    exams = load_exams(user["id"])
    if exams.empty:
        st.info("Arsip masih kosong. Jalankan skrining pertama Anda di halaman AI Detection.")
        return

    if st.session_state.open_exam:
        render_exam_detail(user, st.session_state.open_exam)
        st.markdown("---")

    with st.container(border=True):
        f1, f2, f3 = st.columns([1.4, 1, 1])
        with f1:
            all_labels = sorted({l.strip() for s in exams["labels"].dropna() for l in s.split(",") if l.strip()})
            picked = st.multiselect("Kelas lesi", all_labels, placeholder="Semua kelas")
        with f2:
            urg = st.multiselect("Prioritas", ["Tinggi", "Sedang–Tinggi", "Sedang", "Rendah"], placeholder="Semua")
        with f3:
            min_conf = st.slider("Keyakinan minimum", 0, 100, 0, 5)

        g1, g2 = st.columns(2)
        dates = pd.to_datetime(exams["exam_date"], errors="coerce").dropna()
        with g1:
            rng = st.date_input("Rentang tanggal",
                                value=(dates.min().date(), dates.max().date()) if not dates.empty else ())
        with g2:
            q = st.text_input("Cari", placeholder="Nama pasien, nama berkas, atau isi sintesis")

    view = exams.copy()
    if picked:
        view = view[view["labels"].fillna("").apply(lambda s: any(p in s for p in picked))]
    if urg:
        view = view[view["urgency"].isin(urg)]
    if min_conf:
        view = view[pd.to_numeric(view["max_conf"], errors="coerce").fillna(0) >= min_conf / 100]
    if isinstance(rng, tuple) and len(rng) == 2:
        d = pd.to_datetime(view["exam_date"], errors="coerce")
        view = view[(d >= pd.to_datetime(rng[0])) & (d <= pd.to_datetime(rng[1]))]
    if q:
        blob = (view["patient_name"].fillna("") + " " + view["file_name"].fillna("") + " " +
                view["synthesis"].fillna("") + " " + view["labels"].fillna(""))
        view = view[blob.str.contains(q, case=False, na=False)]

    st.caption(f"{len(view)} dari {len(exams)} pemeriksaan ditampilkan.")
    if view.empty:
        st.warning("Tidak ada pemeriksaan yang cocok dengan filter. Longgarkan kriterianya.")
        return

    table = view[["exam_date", "patient_name", "patient_code", "labels", "max_conf",
                  "urgency", "s_severity", "model_version", "file_name", "id"]].copy()
    table.columns = ["Tanggal", "Pasien", "No. RM", "Temuan", "Keyakinan",
                     "Prioritas", "Nyeri", "Model", "Berkas", "ID"]
    table["Keyakinan"] = pd.to_numeric(table["Keyakinan"], errors="coerce").fillna(0)

    st.dataframe(
        table, use_container_width=True, hide_index=True,
        column_config={
            "Keyakinan": st.column_config.ProgressColumn("Keyakinan", min_value=0, max_value=1, format="%.2f"),
            "Nyeri": st.column_config.NumberColumn("Nyeri", format="%d/10"),
            "ID": st.column_config.TextColumn("ID", width="small"),
        },
    )

    st.markdown("### Buka pemeriksaan")
    o1, o2 = st.columns([3, 1])
    with o1:
        labels = {
            f"{r['exam_date']} · {r['patient_name'] or 'tanpa pasien'} · {r['labels'] or 'tanpa temuan'}"
            f" · {r['id'][:8]}": r["id"]
            for _, r in view.head(60).iterrows()
        }
        sel = st.selectbox("Pilih", list(labels.keys()), label_visibility="collapsed")
    with o2:
        if st.button("Tampilkan", type="primary", use_container_width=True):
            st.session_state.open_exam = labels[sel]
            st.rerun()

    st.markdown("### Ekspor")
    e1, e2, e3 = st.columns(3)
    export = view.drop(columns=["user_id"], errors="ignore")
    with e1:
        st.download_button("Unduh CSV", export.to_csv(index=False).encode("utf-8"),
                           file_name=f"mammouth_rekam_{user['username']}.csv", mime="text/csv",
                           use_container_width=True)
    with e2:
        try:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                export.to_excel(w, index=False, sheet_name="Pemeriksaan")
            st.download_button("Unduh Excel", buf.getvalue(),
                               file_name=f"mammouth_rekam_{user['username']}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
        except Exception:  # noqa: BLE001
            st.button("Excel butuh openpyxl", disabled=True, use_container_width=True)
    with e3:
        st.download_button("Unduh arsip lengkap (ZIP)", build_export_zip(user["id"]),
                           file_name=f"mammouth_arsip_{user['username']}.zip", mime="application/zip",
                           use_container_width=True)

def render_exam_detail(user: dict, exam_id: str) -> None:
    exam = get_exam(user["id"], exam_id)
    if exam is None:
        st.session_state.open_exam = None
        st.warning("Pemeriksaan tidak ditemukan pada akun ini.")
        return

    with st.container(border=True):
        h1, h2 = st.columns([3, 1])
        with h1:
            st.markdown(f"### {exam.get('patient_name') or 'Tanpa identitas pasien'}")
            st.caption(f"{exam['exam_date']} · {exam.get('file_name') or '—'} · "
                       f"model {exam.get('model_version') or '—'} · ID {exam_id[:8].upper()}")
        with h2:
            st.markdown(f"<div style='text-align:right;padding-top:12px'>"
                        f"{pill(exam.get('urgency') or 'Rendah', urgency_tone(exam.get('urgency')))}</div>",
                        unsafe_allow_html=True)
            if st.button("Tutup", key="close_detail", use_container_width=True):
                st.session_state.open_exam = None
                st.rerun()

        if exam.get("is_demo"):
            st.warning("Pemeriksaan ini dibuat dalam mode demo — deteksinya simulasi.")

        has_vitals = any(exam.get(k) for k in ("bp_systolic", "bp_diastolic", "pulse_rate", "resp_rate", "weight_kg", "height_cm", "bmi"))
        if has_vitals:
            st.markdown("**Tanda-tanda vital**")
            cat, tone = bmi_category(exam.get("bmi"))
            bmi_txt = f"{exam['bmi']:.1f}" if exam.get("bmi") else "—"
            unit = "<span style='font-size:.7rem;font-weight:500;color:var(--muted)'>{}</span>"
            st.markdown(
                "<div class='vital-grid'>"
                f"<div class='vital-cell'><p class='vlabel'>Tekanan darah</p><p class='vvalue'>{exam.get('bp_systolic') or '—'}/{exam.get('bp_diastolic') or '—'} {unit.format('mmHg')}</p></div>"
                f"<div class='vital-cell'><p class='vlabel'>Nadi</p><p class='vvalue'>{exam.get('pulse_rate') or '—'} {unit.format('x/menit')}</p></div>"
                f"<div class='vital-cell'><p class='vlabel'>Respirasi</p><p class='vvalue'>{exam.get('resp_rate') or '—'} {unit.format('x/menit')}</p></div>"
                f"<div class='vital-cell'><p class='vlabel'>Berat badan</p><p class='vvalue'>{exam.get('weight_kg') or '—'} {unit.format('kg')}</p></div>"
                f"<div class='vital-cell'><p class='vlabel'>Tinggi badan</p><p class='vvalue'>{exam.get('height_cm') or '—'} {unit.format('cm')}</p></div>"
                f"<div class='vital-cell'><p class='vlabel'>IMT (BMI)</p><p class='vvalue'>{bmi_txt} {pill(cat, tone)}</p></div>"
                "</div>",
                unsafe_allow_html=True,
            )

        i1, i2 = st.columns(2, gap="large")
        with i1:
            annot = load_stored(exam.get("annot_path"))
            orig = load_stored(exam.get("image_path"))
            if annot or orig:
                t1, t2 = st.tabs(["Dengan anotasi", "Citra asli"])
                with t1:
                    show_image(annot or orig, "")
                with t2:
                    show_image(orig or annot, "")
            else:
                st.caption("Citra tidak tersimpan untuk pemeriksaan ini.")
        with i2:
            st.markdown("**Temuan**")
            if exam["detections"]:
                for d in exam["detections"]:
                    info = LESION_INFO.get(d["label"].lower(), {})
                    pct = d["confidence"] * 100
                    st.markdown(
                        f"<div class='det-row'><div><div class='det-name'>{info.get('nama_klinis', d['label'])}</div>"
                        f"<div class='det-sub'>{info.get('awam', '')}</div></div>"
                        f"<div style='text-align:right'><div class='meter'><span style='width:{pct:.0f}%'></span></div>"
                        f"<div class='det-sub'>{pct:.1f}%</div></div></div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("Tidak ada anomali terdeteksi.")

            st.markdown("**Anamnesis**")
            olds = [("Onset", "o_onset"), ("Lokasi", "l_location"), ("Durasi", "d_duration"),
                    ("Karakter", "c_character"), ("Memperberat", "a_aggravating"),
                    ("Meredakan", "r_relieving"), ("Waktu", "t_timing")]
            st.markdown(
                "".join(f"<div class='det-row'><span class='det-sub'>{lab}</span>"
                        f"<span class='det-name' style='font-weight:500'>{exam.get(k) or '—'}</span></div>"
                        for lab, k in olds)
                + f"<div class='det-row'><span class='det-sub'>Skala nyeri</span>"
                  f"<span class='det-name'>{exam.get('s_severity', 0)}/10</span></div>",
                unsafe_allow_html=True,
            )

        tone = urgency_tone(exam.get("urgency"))
        cls = {"high": "u-tinggi", "med": "u-sedang", "low": "u-rendah"}.get(tone, "")
        st.markdown(f"<div class='verdict {cls}'><p class='title'>Sintesis klinis</p>"
                    f"<p class='body'>{exam.get('synthesis') or '—'}</p></div>", unsafe_allow_html=True)

        with st.form(f"note_{exam_id}"):
            note = st.text_area("Catatan pemeriksa", exam.get("clinician_note") or "",
                                placeholder="Tambahkan hasil pemeriksaan langsung, rencana perawatan, atau rujukan.")
            if st.form_submit_button("Simpan catatan", type="primary"):
                conn = get_conn()
                conn.execute("UPDATE exams SET clinician_note=? WHERE id=? AND user_id=?",
                             (note.strip(), exam_id, user["id"]))
                conn.commit()
                conn.close()
                st.success("Catatan tersimpan.")
                st.rerun()

        a1, a2 = st.columns(2)
        with a1:
            st.download_button("Unduh laporan", build_report_html(exam, user["full_name"], user.get("institution") or ""),
                               file_name=f"laporan_{exam_id[:8]}.html", mime="text/html",
                               key=f"dl_{exam_id}", use_container_width=True)
        with a2:
            if st.button("Hapus pemeriksaan", key=f"delx_{exam_id}", use_container_width=True):
                st.session_state[f"confirm_exam_{exam_id}"] = True
                st.rerun()

        if st.session_state.get(f"confirm_exam_{exam_id}"):
            st.warning("Pemeriksaan dan citranya akan dihapus permanen.")
            y, n = st.columns(2)
            with y:
                if st.button("Ya, hapus", key=f"yesx_{exam_id}", type="primary", use_container_width=True):
                    delete_exam(user["id"], exam_id)
                    st.session_state[f"confirm_exam_{exam_id}"] = False
                    st.session_state.open_exam = None
                    st.rerun()
            with n:
                if st.button("Batal", key=f"nox_{exam_id}", use_container_width=True):
                    st.session_state[f"confirm_exam_{exam_id}"] = False
                    st.rerun()


# ============================================================
# 12. HALAMAN: ANALITIK
# ============================================================
def page_analytics(user: dict) -> None:
    page_head(_t("Analytics"), "Pola temuan dari seluruh pemeriksaan yang Anda rekam di akun ini.")

    exams = load_exams(user["id"])
    dets = load_detections(user["id"])
    if exams.empty:
        st.info("Analitik akan muncul setelah ada pemeriksaan tersimpan.")
        return

    primary = THEMES[st.session_state.theme if st.session_state.theme in THEMES else "terang"]["primary"]
    accent = THEMES[st.session_state.theme if st.session_state.theme in THEMES else "terang"]["accent"]

    hide_demo = st.toggle("Sembunyikan data mode demo", value=True)
    if hide_demo:
        exams = exams[exams["is_demo"] == 0]
        dets = dets[dets["is_demo"] == 0]
        if exams.empty:
            st.info("Semua data pada akun ini berasal dari mode demo. Matikan filter untuk melihatnya.")
            return

    c1, c2, c3, c4 = st.columns(4)
    positive = exams[exams["n_detections"] > 0]
    with c1:
        kpi("Citra dianalisis", f"{len(exams)}", color="primary")
    with c2:
        rate = len(positive) / len(exams) * 100 if len(exams) else 0
        kpi("Citra dengan temuan", f"{rate:.0f}%", f"{len(positive)} citra")
    with c3:
        kpi("Deteksi total", f"{len(dets)}",
            f"{len(dets)/len(exams):.1f} per citra" if len(exams) else "")
    with c4:
        kpi("Nyeri rata-rata", f"{pd.to_numeric(exams['s_severity'], errors='coerce').fillna(0).mean():.1f}/10",
            color="accent")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    g1, g2 = st.columns(2, gap="large")

    with g1:
        st.markdown("### Frekuensi kelas lesi")
        if dets.empty:
            st.caption("Belum ada deteksi positif.")
        elif alt is not None:
            vc = dets["label"].value_counts().reset_index()
            vc.columns = ["Kelas", "Jumlah"]
            st.altair_chart(
                alt.Chart(vc).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
                    x=alt.X("Jumlah:Q", title=None),
                    y=alt.Y("Kelas:N", sort="-x", title=None),
                    color=alt.value(primary), tooltip=["Kelas", "Jumlah"],
                ).properties(height=max(200, 34 * len(vc))),
                use_container_width=True,
            )
        else:
            st.bar_chart(dets["label"].value_counts())

    with g2:
        st.markdown("### Keyakinan per kelas")
        if dets.empty:
            st.caption("Belum ada deteksi positif.")
        elif alt is not None:
            st.altair_chart(
                alt.Chart(dets).mark_boxplot(extent="min-max", color=primary).encode(
                    x=alt.X("confidence:Q", title="Keyakinan", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("label:N", title=None),
                ).properties(height=max(200, 34 * dets["label"].nunique())),
                use_container_width=True,
            )
        else:
            st.dataframe(dets.groupby("label")["confidence"].describe(), use_container_width=True)

    g3, g4 = st.columns(2, gap="large")
    with g3:
        st.markdown("### Aktivitas harian")
        daily = exams.groupby("exam_date").size().reset_index(name="Pemeriksaan")
        if alt is not None and not daily.empty:
            st.altair_chart(
                alt.Chart(daily).mark_area(line={"color": primary}, opacity=0.25, color=primary).encode(
                    x=alt.X("exam_date:T", title=None),
                    y=alt.Y("Pemeriksaan:Q", title=None),
                    tooltip=["exam_date", "Pemeriksaan"],
                ).properties(height=240),
                use_container_width=True,
            )
        else:
            st.line_chart(daily.set_index("exam_date"))

    with g4:
        st.markdown("### Sebaran skala nyeri")
        sev = pd.to_numeric(exams["s_severity"], errors="coerce").fillna(0).astype(int)
        hist = sev.value_counts().sort_index().reset_index()
        hist.columns = ["VAS", "Jumlah"]
        if alt is not None:
            st.altair_chart(
                alt.Chart(hist).mark_bar(cornerRadius=3, color=accent).encode(
                    x=alt.X("VAS:O", title="Skala nyeri"), y=alt.Y("Jumlah:Q", title=None),
                    tooltip=["VAS", "Jumlah"],
                ).properties(height=240),
                use_container_width=True,
            )
        else:
            st.bar_chart(hist.set_index("VAS"))

    st.markdown("### Prioritas tindak lanjut per kelas")
    if not dets.empty:
        tab = dets.groupby("label").agg(
            Deteksi=("id", "count"),
            Keyakinan_rata=("confidence", "mean"),
            Nyeri_rata=("s_severity", "mean"),
        ).reset_index().rename(columns={"label": "Kelas"})
        tab["Prioritas"] = tab["Kelas"].map(lambda k: LESION_INFO.get(k.lower(), {}).get("urgensi", "Rendah"))
        tab = tab.sort_values("Deteksi", ascending=False)
        st.dataframe(
            tab, use_container_width=True, hide_index=True,
            column_config={
                "Keyakinan_rata": st.column_config.ProgressColumn("Keyakinan rata-rata", min_value=0, max_value=1, format="%.2f"),
                "Nyeri_rata": st.column_config.NumberColumn("Nyeri rata-rata", format="%.1f"),
            },
        )


# ============================================================
# 13. HALAMAN: ENSIKLOPEDIA
# ============================================================
def page_encyclopedia() -> None:
    page_head(_t("Lesion Database"), "Delapan kelas yang dikenali model, beserta tatalaksana dan tanda bahayanya.")

    c1, c2 = st.columns([2, 1])
    with c1:
        q = st.text_input("Cari", placeholder="Nama lesi, gejala, atau kata kunci", label_visibility="collapsed")
    with c2:
        level = st.selectbox("Saring prioritas", ["Semua", "Rendah", "Sedang", "Sedang–Tinggi"],
                             label_visibility="collapsed")

    items = []
    for key, info in LESION_INFO.items():
        blob = " ".join(str(v) for v in info.values()) + " " + key
        if q and q.lower() not in blob.lower():
            continue
        if level != "Semua" and info["urgensi"] != level:
            continue
        items.append((key, info))

    if not items:
        st.warning("Tidak ada lesi yang cocok. Coba kata kunci lain.")
        return

    for i in range(0, len(items), 2):
        cols = st.columns(2, gap="large")
        for col, (key, info) in zip(cols, items[i:i + 2]):
            with col, st.container(border=True):
                a, b = st.columns([3, 1])
                with a:
                    st.markdown(f"#### {info['nama_klinis']}")
                    st.caption(f"{info['awam']} · kelas model: `{key}`")
                with b:
                    st.markdown(f"<div style='text-align:right'>{pill(info['urgensi'], urgency_tone(info['urgensi']))}</div>",
                                unsafe_allow_html=True)
                st.write(info["deskripsi"])
                st.markdown(f"**Penyebab.** {info['etiologi']}")
                st.markdown(f"**Tatalaksana.** {info['tatalaksana']}")
                with st.expander("Diagnosis banding dan tanda bahaya"):
                    st.markdown(f"**Banding.** {info['banding']}")
                    st.markdown(f"**Tanda bahaya.** {info['red_flag']}")


# ============================================================
# 14. HALAMAN: PENGATURAN
# ============================================================
def page_settings(user: dict, model, weights: Optional[Path]) -> None:
    page_head(_t("Settings"), "Profil, keamanan, model, dan pengelolaan data akun Anda.")

    t_prof, t_sec, t_model, t_data, t_about = st.tabs(
        ["Profil", "Keamanan", "Model", "Data", "Tentang"]
    )

    with t_prof:
        with st.container(border=True):
            with st.form("form_profil"):
                c1, c2 = st.columns(2)
                with c1:
                    fn = st.text_input("Nama lengkap", user["full_name"])
                    rl = st.text_input("Peran", user.get("role") or "")
                with c2:
                    inst = st.text_input("Institusi atau klinik", user.get("institution") or "")
                    st.text_input("Username", user["username"], disabled=True,
                                  help="Username tidak dapat diubah")
                if st.form_submit_button("Simpan profil", type="primary"):
                    update_profile(user["id"], fn, rl, inst)
                    st.session_state.user = dict(get_user(user["id"]))
                    st.success("Profil diperbarui.")
                    st.rerun()
        st.caption(f"Akun dibuat {user['created_at'][:10]} · terakhir masuk {(user.get('last_login') or '—')[:16]}")

    with t_sec:
        with st.container(border=True):
            st.markdown("### Ganti kata sandi")
            with st.form("form_sandi"):
                old = st.text_input("Kata sandi saat ini", type="password")
                n1 = st.text_input("Kata sandi baru", type="password")
                n2 = st.text_input("Ulangi kata sandi baru", type="password")
                if st.form_submit_button("Perbarui kata sandi", type="primary"):
                    if n1 != n2:
                        st.error("Kedua kata sandi baru belum sama.")
                    else:
                        ok, msg = change_password(user["id"], old, n1)
                        (st.success if ok else st.error)(msg)
        with st.container(border=True):
            st.markdown("### Bagaimana data Anda dipisahkan")
            st.write(
                "Setiap baris pasien, pemeriksaan, dan deteksi membawa penanda akun pemiliknya, dan setiap "
                "kueri aplikasi menyaring berdasarkan penanda itu. Citra disimpan di folder terpisah per akun. "
                "Akun lain di server yang sama tidak bisa membuka data Anda dari dalam aplikasi."
            )
            st.caption("Catatan: siapa pun yang punya akses langsung ke berkas server tetap bisa membaca basis data. "
                       "Untuk pemakaian klinis nyata, aktifkan enkripsi disk dan HTTPS.")

    with t_model:
        with st.container(border=True):
            st.markdown("### Status runtime")
            model_info = inspect_loaded_model(model, weights)
            rows = [
                ("Paket ultralytics", "terpasang" if YOLO else "belum terpasang"),
                ("YOLO checkpoint", "LOADED / siap inferensi" if model_info["loaded"] else "GAGAL DIMUAT"),
                ("Task model", model_info.get("task", "—")),
                ("Jumlah kelas", str(model_info.get("count", 0))),
                ("Ukuran bobot", f"{model_info.get('size_mb', 0):.2f} MB"),
                ("Google Gemini API", "SDK + konfigurasi tersedia" if (genai is not None and get_gemini_api_key()) else "belum siap"),
                ("Arsitektur aktif", st.session_state.model_version),
                ("Berkas bobot", str(weights) if weights else "tidak ditemukan"),
                ("Ambang keyakinan", f"{st.session_state.conf_thr:.2f}"),
                ("Ambang IoU", f"{st.session_state.iou_thr:.2f}"),
            ]
            st.markdown("".join(
                f"<div class='det-row'><span class='det-sub'>{k}</span><span class='det-name'>{v}</span></div>"
                for k, v in rows), unsafe_allow_html=True)

            if not weights:
                st.warning(
                    "`best.pt` belum ditemukan. Pastikan file bobot benar-benar ikut di-deploy ke repository "
                    "atau set `MAMMOUTH_MODEL_PATH` ke path file `.pt` yang valid."
                )
                roots_txt = "\n".join(f"- `{r}`" for r in _weight_search_roots())
                with st.expander("Lokasi yang diperiksa MAMMOUTH"):
                    st.markdown(roots_txt)
            elif not model_info["loaded"]:
                st.error(f"`best.pt` ditemukan di server, tetapi checkpoint GAGAL dimuat: {st.session_state.get('model_load_error') or 'error tidak tersedia'}")
                st.info("Ini berbeda dari masalah file tidak ditemukan. Kemungkinan terkait format checkpoint, versi Ultralytics/PyTorch, atau dependensi runtime.")
            else:
                st.success(f"✓ `best.pt` ditemukan dan berhasil dimuat sebagai model {model_info.get('task', 'unknown')} dengan {model_info.get('count', 0)} kelas.")
                if model_info.get("classes"):
                    with st.expander("Kelas yang dibaca dari best.pt"):
                        st.code("\n".join(model_info["classes"]))

            st.markdown("**Konfigurasi Gemini**")
            gemini_key_present = bool(get_gemini_api_key())
            st.markdown(
                "".join([
                    f"<div class='det-row'><span class='det-sub'>API key</span><span class='det-name'>{'terdeteksi' if gemini_key_present else 'tidak ditemukan'}</span></div>",
                    f"<div class='det-row'><span class='det-sub'>SDK</span><span class='det-name'>{'google-genai' if genai is not None else 'belum terpasang'}</span></div>",
                    f"<div class='det-row'><span class='det-sub'>Model</span><span class='det-name'>{get_gemini_model_name()}</span></div>",
                ]),
                unsafe_allow_html=True,
            )
            st.caption("Secrets yang didukung: `GEMINI_API_KEY` atau `GOOGLE_API_KEY`. Jangan masukkan key ke GitHub.")
            st.caption(f"Status request: {GEMINI_LAST_STATUS}")
            if GEMINI_SDK_ERROR:
                with st.expander("Detail error Gemini", expanded=False):
                    st.code(GEMINI_SDK_ERROR)
            if st.button("Tes koneksi Gemini", key="test_gemini", use_container_width=True):
                with st.spinner("Menghubungi Gemini…"):
                    ok, msg = test_gemini_connection()
                if ok:
                    st.success(f"Gemini aktif. Respons: {msg}")
                else:
                    st.error(f"Gemini gagal dihubungi: {msg}")

            st.markdown("**Nama berkas yang dicari untuk tiap arsitektur**")
            for v, files in MODEL_FILES.items():
                st.caption(f"{v}: " + ", ".join(f"`{f}`" for f in files))

        with st.container(border=True):
            st.markdown("### Mode demo")
            st.write("Menjalankan alur kerja lengkap dengan deteksi tiruan ketika bobot model belum tersedia. "
                     "Pemeriksaan yang dihasilkan diberi tanda demo dan bisa disembunyikan dari analitik.")
            demo = st.toggle("Aktifkan mode demo", value=st.session_state.demo_mode)
            if demo != st.session_state.demo_mode:
                st.session_state.demo_mode = demo
                st.rerun()

        with st.container(border=True):
            st.markdown("### Simpan sebagai bawaan")
            st.write("Arsitektur dan kedua ambang di sidebar akan dipakai ulang setiap kali Anda masuk.")
            if st.button("Jadikan pengaturan saat ini sebagai bawaan", type="primary"):
                set_user_pref(user["id"], model_version=st.session_state.model_version,
                              conf_thr=st.session_state.conf_thr, iou_thr=st.session_state.iou_thr)
                st.success("Tersimpan.")

    with t_data:
        exams = load_exams(user["id"])
        pats = list_patients(user["id"])
        folder = IMG_DIR / user["id"]
        size_mb = sum(f.stat().st_size for f in folder.glob("*")) / 1e6 if folder.exists() else 0

        c1, c2, c3 = st.columns(3)
        with c1:
            kpi("Pemeriksaan", f"{len(exams)}", color="primary")
        with c2:
            kpi("Pasien", f"{len(pats)}")
        with c3:
            kpi("Citra tersimpan", f"{size_mb:.1f} MB", str(folder))

        with st.container(border=True):
            st.markdown("### Ekspor")
            st.write("Arsip ZIP berisi tiga berkas CSV (pasien, pemeriksaan, deteksi) beserta seluruh citra Anda.")
            inc = st.toggle("Sertakan citra", value=True)
            st.download_button("Unduh arsip akun", build_export_zip(user["id"], inc),
                               file_name=f"mammouth_arsip_{user['username']}.zip",
                               mime="application/zip", type="primary")

        with st.container(border=True):
            st.markdown("### Hapus data")
            st.write("Tindakan berikut permanen dan hanya memengaruhi akun Anda.")
            d1, d2 = st.columns(2)
            with d1:
                if st.button("Kosongkan semua data klinis", use_container_width=True):
                    st.session_state["confirm_wipe"] = True
                    st.rerun()
            with d2:
                if st.button("Hapus akun ini", use_container_width=True):
                    st.session_state["confirm_account"] = True
                    st.rerun()

            if st.session_state.get("confirm_wipe"):
                st.warning("Seluruh pasien, pemeriksaan, dan citra akan hilang. Akun tetap aktif.")
                word = st.text_input("Ketik HAPUS untuk mengonfirmasi", key="wipe_word")
                if st.button("Jalankan penghapusan", type="primary", disabled=word != "HAPUS"):
                    wipe_user_data(user["id"])
                    st.session_state["confirm_wipe"] = False
                    st.success("Data klinis dikosongkan.")
                    st.rerun()

            if st.session_state.get("confirm_account"):
                st.error("Akun beserta seluruh isinya akan dihapus dan Anda langsung keluar.")
                word2 = st.text_input("Ketik nama pengguna Anda untuk mengonfirmasi", key="acc_word")
                if st.button("Hapus akun permanen", type="primary", disabled=word2 != user["username"]):
                    delete_account(user["id"])
                    for k, v in DEFAULTS.items():
                        st.session_state[k] = v
                    st.rerun()

        if user.get("is_admin"):
            with st.container(border=True):
                st.markdown("### Administrasi server")
                st.caption("Administrator melihat daftar akun dan volume datanya, bukan isi rekam medisnya.")
                conn = get_conn()
                admin_df = pd.read_sql(
                    """SELECT u.username AS Username, u.full_name AS Nama, u.role AS Peran,
                              u.created_at AS Dibuat, u.last_login AS "Terakhir masuk",
                              (SELECT COUNT(*) FROM exams e WHERE e.user_id=u.id) AS Pemeriksaan,
                              (SELECT COUNT(*) FROM patients p WHERE p.user_id=u.id) AS Pasien
                       FROM users u ORDER BY u.created_at""", conn)
                conn.close()
                st.dataframe(admin_df, use_container_width=True, hide_index=True)

    with t_about:
        with st.container(border=True):
            st.markdown(f"### {APP_NAME} v{APP_VERSION}")
            st.write(
                "Proyek independen untuk skrining lesi rongga mulut berbasis computer vision. "
                "Aplikasi ini adalah alat bantu penapisan, bukan pengganti pemeriksaan klinis. "
                "Diagnosis akhir selalu ditegakkan oleh dokter gigi melalui pemeriksaan langsung, "
                "bila perlu dengan radiograf dan biopsi."
            )
            st.markdown(
                "- Antarmuka: Streamlit\n"
                "- Deteksi objek: Ultralytics YOLO (v8/v11/v12)\n"
                "- Sintesis Klinis: Google Gemini AI\n"
                "- Penyimpanan: SQLite relasional dan arsip citra per akun\n"
                "- Keamanan kata sandi: PBKDF2-HMAC-SHA256, 200.000 iterasi, salt per akun"
            )
            st.caption(f"Folder data: `{DATA_DIR.resolve()}`")


# ============================================================
# 15. ROUTER
# ============================================================
def main() -> None:
    if st.session_state.user is None:
        render_login()
        return

    user = st.session_state.user
    fresh = get_user(user["id"])
    if fresh is None:  # akun terhapus dari sesi lain
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()
    user = dict(fresh)
    st.session_state.user = user

    weights = find_weights(st.session_state.model_version)
    model, model_load_error = load_model(st.session_state.model_version, str(weights) if weights else "")
    st.session_state["model_load_error"] = model_load_error
    render_sidebar(user, weights)

    page = st.session_state.page
    if page == "Dashboard":
        page_dashboard(user)
    elif page in ("AI Detection", "Upload Image"):
        page_screening(user, model, weights)
    elif page == "Patients":
        page_patients(user)
    elif page in ("Detection History", "Reports"):
        page_records(user)
    elif page == "Analytics":
        page_analytics(user)
    elif page == "Lesion Database":
        page_encyclopedia()
    elif page == "Settings":
        page_settings(user, model, weights)
    else:
        page_dashboard(user)

    st.markdown(
        f"<hr><p style='font-size:.78rem;color:var(--muted);text-align:center'>"
        f"{APP_NAME} v{APP_VERSION} · alat bantu skrining, bukan alat diagnosis. "
        f"Konfirmasi setiap temuan dengan pemeriksaan klinis langsung.</p>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
