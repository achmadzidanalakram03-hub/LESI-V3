"""
MAMMOUTH — My Assistant in Mouth Health
=======================================
Platform skrining kesehatan rongga mulut berbasis computer vision.

v6.1 — Proyek independen.
Fitur utama:
  • Akun mandiri (registrasi terbuka) dengan isolasi data penuh per pengguna
  • Penyimpanan permanen: SQLite relasional + arsip citra per akun di disk
  • Rekam medis pasien (bukan sekadar log gambar): pasien → pemeriksaan → deteksi
  • Anamnesis OLD CARTS + tanda-tanda vital (TD, nadi, napas, BB/TB, IMT otomatis)
  • Anamnesis dan tanda vital terintegrasi dengan sintesis klinis berbasis aturan
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


# ------------------------------------------------------------
# 0. KOMPATIBILITAS API STREAMLIT
# ------------------------------------------------------------
# Streamlit >=1.50 mengganti `use_container_width` dengan `width`. Shim ini
# menerjemahkan argumen lama ke argumen baru saat aplikasi dijalankan, sehingga
# berkas yang sama tetap jalan di versi lama maupun versi terbaru.
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
            continue  # versi lama, argumen asli masih diterima
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
LEGACY_DB = Path("mammouth.db")  # basis data versi lama (v5) bila ada

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

@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
footer, #MainMenu { visibility: hidden; }
"""


def theme_css(theme_name: str) -> str:
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
    theme        TEXT NOT NULL DEFAULT 'terang',
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
    """Menambahkan kolom tanda vital pada basis data lama (dibuat sebelum v6.1)."""
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
                "terang", "{}", datetime.now().isoformat(timespec="seconds"),
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
    if row["iterations"] == 0:  # akun warisan v5 (SHA-256 polos)
        ok = hmac.compare_digest(row["pw_hash"], hashlib.sha256(password.encode()).hexdigest())
        if ok:  # naikkan ke PBKDF2 saat login berhasil
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
    """Memindahkan tabel users/emr_logs versi lama ke skema baru. Dijalankan sekali."""
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
                   VALUES(?,?,?,?,0,?,?,'',0,'terang','{}',?)""",
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
# 4. MESIN AI & SINTESIS KLINIS
# ============================================================
def find_weights(version: str) -> Optional[Path]:
    for cand in MODEL_FILES.get(version, ["best.pt"]):
        p = Path(cand)
        if p.exists():
            return p
    return None


@st.cache_resource(show_spinner=False)
def load_model(version: str, weight_path: str):
    if YOLO is None or not weight_path:
        return None
    try:
        return YOLO(weight_path)
    except Exception:  # noqa: BLE001
        return None


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
    """Deteksi tiruan untuk menjelajah aplikasi tanpa bobot model. Bukan hasil AI."""
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


def synthesize(detections: list[dict], anam: dict) -> str:
    """Menggabungkan temuan visual dengan anamnesis OLD CARTS menjadi kalimat kerja klinis."""
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
    """Menghitung indeks massa tubuh (IMT) = berat badan (kg) / tinggi badan (m)^2."""
    try:
        w, h = float(weight_kg or 0), float(height_cm or 0)
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return w / ((h / 100) ** 2)


def bmi_category(bmi: Optional[float]) -> tuple[str, str]:
    """Klasifikasi IMT dewasa menurut acuan Asia-Pasifik (rujukan Kemenkes RI).
    Mengembalikan (label, tone) — tone dipakai untuk pewarnaan lencana (pill)."""
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
<h2>Sintesis klinis</h2>
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
    "theme": "terang",
    "page": "Ringkasan",
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
    """Grafik gigi sederhana untuk halaman masuk — satu aksen visual, tanpa animasi."""
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
                            st.session_state.theme = row["theme"] or "terang"
                            pref = json.loads(row["pref"] or "{}")
                            st.session_state.model_version = pref.get("model_version", "YOLOv8")
                            st.session_state.conf_thr = pref.get("conf_thr", 0.25)
                            st.session_state.iou_thr = pref.get("iou_thr", 0.45)
                            st.session_state.login_fails = 0
                            st.session_state.page = "Ringkasan"
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
    ("Ringkasan", "Ringkasan"),
    ("Skrining", "Skrining"),
    ("Pasien", "Pasien"),
    ("Rekam medis", "Rekam medis"),
    ("Analitik", "Analitik"),
    ("Ensiklopedia", "Ensiklopedia"),
    ("Pengaturan", "Pengaturan"),
]


def render_sidebar(user: dict, weights: Optional[Path]) -> None:
    with st.sidebar:
        st.markdown(
            f"""<div class="brand"><span class="mark">{APP_NAME}</span><span class="ver">v{APP_VERSION}</span></div>
                <p class="brand-sub">{APP_TAGLINE}</p>""",
            unsafe_allow_html=True,
        )

        for key, label in NAV:
            active = st.session_state.page == key
            if st.button(label, key=f"nav_{key}", use_container_width=True,
                         type="primary" if active else "secondary"):
                st.session_state.page = key
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
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Tema", use_container_width=True, help="Ganti antara tampilan terang dan gelap"):
                st.session_state.theme = "gelap" if st.session_state.theme == "terang" else "terang"
                set_user_pref(user["id"], theme=st.session_state.theme)
                st.rerun()
        with c2:
            if st.button("Keluar", use_container_width=True):
                set_user_pref(user["id"], model_version=st.session_state.model_version,
                              conf_thr=st.session_state.conf_thr, iou_thr=st.session_state.iou_thr)
                log_activity(user["id"], "logout", "")
                for k in ["user", "anamnesis", "active_patient", "last_batch", "open_exam"]:
                    st.session_state[k] = DEFAULTS[k]
                st.rerun()


# ============================================================
# 8. HALAMAN: RINGKASAN
# ============================================================
def page_overview(user: dict) -> None:
    page_head(f"Halo, {user['full_name'].split()[-1]}",
              "Ringkasan aktivitas skrining pada akun Anda.")

    exams = load_exams(user["id"])
    dets = load_detections(user["id"])
    pats = list_patients(user["id"])

    if exams.empty:
        with st.container(border=True):
            st.markdown("### Belum ada pemeriksaan")
            st.write("Mulai dengan mendaftarkan pasien, lalu unggah citra klinis pertama untuk dianalisis.")
            a, b, _ = st.columns([1, 1, 2])
            with a:
                if st.button("Daftarkan pasien", type="primary", use_container_width=True):
                    st.session_state.page = "Pasien"
                    st.rerun()
            with b:
                if st.button("Buka skrining", use_container_width=True):
                    st.session_state.page = "Skrining"
                    st.rerun()
        return

    week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    recent = exams[exams["exam_date"] >= week_ago]
    urgent = exams[exams["urgency"].isin(["Tinggi", "Sedang–Tinggi"])]
    avg_conf = dets["confidence"].mean() * 100 if not dets.empty else 0
    top = dets["label"].mode()[0].title() if not dets.empty else "—"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Pemeriksaan", f"{len(exams)}", f"{len(recent)} dalam 7 hari terakhir", "primary")
    with c2:
        kpi("Pasien terdaftar", f"{len(pats)}", "Tersimpan di akun ini")
    with c3:
        kpi("Perlu tindak lanjut", f"{len(urgent)}", "Prioritas sedang-tinggi ke atas",
            "danger" if len(urgent) else "text")
    with c4:
        kpi("Rata-rata keyakinan", f"{avg_conf:.0f}%", f"Temuan terbanyak: {top}", "accent")

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.4, 1], gap="large")

    with left:
        st.markdown("### Pemeriksaan terakhir")
        for _, row in exams.head(6).iterrows():
            with st.container(border=True):
                a, b = st.columns([3, 1])
                with a:
                    who = row["patient_name"] or "Tanpa identitas pasien"
                    labels = row["labels"] or "Tidak ada temuan"
                    st.markdown(
                        f"**{who}** &nbsp; {pill(row['urgency'] or 'Rendah', urgency_tone(row['urgency']))}<br>"
                        f"<span class='det-sub'>{row['exam_date']} · {labels} · {row['file_name'] or '—'}</span>",
                        unsafe_allow_html=True,
                    )
                with b:
                    if st.button("Buka", key=f"ov_{row['id']}", use_container_width=True):
                        st.session_state.open_exam = row["id"]
                        st.session_state.page = "Rekam medis"
                        st.rerun()

    with right:
        st.markdown("### Sebaran temuan")
        if not dets.empty and alt is not None:
            vc = dets["label"].value_counts().reset_index()
            vc.columns = ["label", "jumlah"]
            chart = (
                alt.Chart(vc).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                .encode(
                    x=alt.X("jumlah:Q", title=None),
                    y=alt.Y("label:N", sort="-x", title=None),
                    color=alt.value(THEMES[st.session_state.theme]["primary"]),
                    tooltip=["label", "jumlah"],
                )
                .properties(height=max(180, 32 * len(vc)))
            )
            st.altair_chart(chart, use_container_width=True)
        elif not dets.empty:
            st.bar_chart(dets["label"].value_counts())
        else:
            st.caption("Belum ada temuan positif yang tercatat.")

        st.markdown("### Prioritas")
        counts = exams["urgency"].fillna("Rendah").value_counts()
        for level in ["Tinggi", "Sedang–Tinggi", "Sedang", "Rendah"]:
            if level in counts:
                st.markdown(
                    f"<div class='det-row'><span class='det-name'>{level}</span>"
                    f"{pill(str(counts[level]) + ' pemeriksaan', urgency_tone(level))}</div>",
                    unsafe_allow_html=True,
                )


# ============================================================
# 9. HALAMAN: SKRINING
# ============================================================
def page_screening(user: dict, model, weights: Optional[Path]) -> None:
    page_head("Skrining citra",
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

    pats = list_patients(user["id"])
    opts = {"— Tanpa identitas pasien —": None}
    opts.update({f"{r['name']} · {r['code']}": r["id"] for _, r in pats.iterrows()})

    c_pat, c_date = st.columns([2, 1])
    with c_pat:
        choice = st.selectbox("Pasien", list(opts.keys()),
                              help="Kaitkan pemeriksaan ini dengan pasien agar riwayatnya terlacak.")
        patient_id = opts[choice]
    with c_date:
        exam_date = st.date_input("Tanggal pemeriksaan", value=datetime.now())

    with st.expander("Tanda-tanda vital", expanded=not st.session_state.vitals):
        with st.form("form_vitals"):
            curv = st.session_state.vitals
            v1, v2, v3 = st.columns(3, gap="large")
            with v1:
                st.markdown("**Tekanan darah**")
                sistol = st.number_input("Sistolik (mmHg)", min_value=0, max_value=300,
                                         value=int(curv.get("bp_systolic") or 0), step=1)
                diastol = st.number_input("Diastolik (mmHg)", min_value=0, max_value=200,
                                          value=int(curv.get("bp_diastolic") or 0), step=1)
            with v2:
                st.markdown("**Pernapasan**")
                nadi = st.number_input("Nadi (denyut/menit)", min_value=0, max_value=250,
                                       value=int(curv.get("pulse_rate") or 0), step=1)
                napas = st.number_input("Frekuensi napas (napas/menit)", min_value=0, max_value=80,
                                        value=int(curv.get("resp_rate") or 0), step=1)
            with v3:
                st.markdown("**Antropometri**")
                berat = st.number_input("Berat badan (kg)", min_value=0.0, max_value=400.0,
                                        value=float(curv.get("weight_kg") or 0.0), step=0.1, format="%.1f")
                tinggi = st.number_input("Tinggi badan (cm)", min_value=0.0, max_value=250.0,
                                         value=float(curv.get("height_cm") or 0.0), step=0.5, format="%.1f")
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
            f"<div class='vital-cell'><p class='vlabel'>Tekanan darah</p>"
            f"<p class='vvalue'>{v.get('bp_systolic') or '—'}/{v.get('bp_diastolic') or '—'} {unit.format('mmHg')}</p></div>"
            f"<div class='vital-cell'><p class='vlabel'>Nadi</p>"
            f"<p class='vvalue'>{v.get('pulse_rate') or '—'} {unit.format('x/menit')}</p></div>"
            f"<div class='vital-cell'><p class='vlabel'>Respirasi</p>"
            f"<p class='vvalue'>{v.get('resp_rate') or '—'} {unit.format('x/menit')}</p></div>"
            f"<div class='vital-cell'><p class='vlabel'>Berat badan</p>"
            f"<p class='vvalue'>{v.get('weight_kg') or '—'} {unit.format('kg')}</p></div>"
            f"<div class='vital-cell'><p class='vlabel'>Tinggi badan</p>"
            f"<p class='vvalue'>{v.get('height_cm') or '—'} {unit.format('cm')}</p></div>"
            f"<div class='vital-cell'><p class='vlabel'>IMT (BMI)</p>"
            f"<p class='vvalue'>{bmi_txt} {pill(cat, tone)}</p></div>"
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
                st.success("Anamnesis tersimpan dan akan dipakai untuk sintesis temuan.")
            if cleared:
                st.session_state.anamnesis = {}
                st.rerun()

    if st.session_state.anamnesis:
        a = st.session_state.anamnesis
        st.caption(f"Anamnesis aktif · nyeri {a.get('s_severity', 0)}/10 · "
                   f"karakter: {a.get('c_character', '-')} · lokasi: {a.get('l_location', '-')}")

    st.markdown("### Citra")
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

    run = st.button(f"Jalankan deteksi pada {len(processed)} citra",
                    type="primary", use_container_width=True)

    if run:
        anam = st.session_state.anamnesis
        vit = st.session_state.vitals
        results_view = []
        bar = st.progress(0.0, text="Menyiapkan…")
        for i, (img, fname) in enumerate(zip(processed, names), start=1):
            bar.progress((i - 1) / len(processed), text=f"Menganalisis {fname}")
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
                "synthesis": synthesize(dets, anam),
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
            f"<div class='verdict {cls}'><p class='title'>Sintesis klinis</p>"
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
    page_head("Pasien", "Daftar pasien beserta riwayat pemeriksaannya. Hanya akun ini yang dapat melihatnya.")

    with st.expander("Tambah pasien", expanded=False):
        with st.form("form_pasien", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                code = st.text_input("Nomor rekam medis", placeholder="Kosongkan untuk dibuatkan otomatis")
                name = st.text_input("Nama pasien")
            with c2:
                by = st.number_input("Tahun lahir", min_value=1900, max_value=datetime.now().year,
                                     value=1995, step=1)
                sex = st.selectbox("Jenis kelamin", ["Perempuan", "Laki-laki", "Tidak disebutkan"])
            with c3:
                contact = st.text_input("Kontak", placeholder="Nomor telepon atau surel")
                med = st.text_area("Riwayat medis singkat", placeholder="Alergi, penyakit sistemik, obat rutin",
                                   height=88)
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
                                st.session_state.page = "Rekam medis"
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
# 11. HALAMAN: REKAM MEDIS
# ============================================================
def page_records(user: dict) -> None:
    page_head("Rekam medis", "Arsip seluruh pemeriksaan pada akun ini, lengkap dengan citra dan sintesisnya.")

    exams = load_exams(user["id"])
    if exams.empty:
        st.info("Arsip masih kosong. Jalankan skrining pertama Anda di halaman Skrining.")
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

        has_vitals = any(exam.get(k) for k in
                         ("bp_systolic", "bp_diastolic", "pulse_rate", "resp_rate", "weight_kg", "height_cm", "bmi"))
        if has_vitals:
            st.markdown("**Tanda-tanda vital**")
            cat, tone = bmi_category(exam.get("bmi"))
            bmi_txt = f"{exam['bmi']:.1f}" if exam.get("bmi") else "—"
            unit = "<span style='font-size:.7rem;font-weight:500;color:var(--muted)'>{}</span>"
            st.markdown(
                "<div class='vital-grid'>"
                f"<div class='vital-cell'><p class='vlabel'>Tekanan darah</p>"
                f"<p class='vvalue'>{exam.get('bp_systolic') or '—'}/{exam.get('bp_diastolic') or '—'} {unit.format('mmHg')}</p></div>"
                f"<div class='vital-cell'><p class='vlabel'>Nadi</p>"
                f"<p class='vvalue'>{exam.get('pulse_rate') or '—'} {unit.format('x/menit')}</p></div>"
                f"<div class='vital-cell'><p class='vlabel'>Respirasi</p>"
                f"<p class='vvalue'>{exam.get('resp_rate') or '—'} {unit.format('x/menit')}</p></div>"
                f"<div class='vital-cell'><p class='vlabel'>Berat badan</p>"
                f"<p class='vvalue'>{exam.get('weight_kg') or '—'} {unit.format('kg')}</p></div>"
                f"<div class='vital-cell'><p class='vlabel'>Tinggi badan</p>"
                f"<p class='vvalue'>{exam.get('height_cm') or '—'} {unit.format('cm')}</p></div>"
                f"<div class='vital-cell'><p class='vlabel'>IMT (BMI)</p>"
                f"<p class='vvalue'>{bmi_txt} {pill(cat, tone)}</p></div>"
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
    page_head("Analitik", "Pola temuan dari seluruh pemeriksaan yang Anda rekam di akun ini.")

    exams = load_exams(user["id"])
    dets = load_detections(user["id"])
    if exams.empty:
        st.info("Analitik akan muncul setelah ada pemeriksaan tersimpan.")
        return

    primary = THEMES[st.session_state.theme]["primary"]
    accent = THEMES[st.session_state.theme]["accent"]

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
                "Keyakinan_rata": st.column_config.ProgressColumn("Keyakinan rata-rata", min_value=0, max_value=1,
                                                                  format="%.2f"),
                "Nyeri_rata": st.column_config.NumberColumn("Nyeri rata-rata", format="%.1f"),
            },
        )


# ============================================================
# 13. HALAMAN: ENSIKLOPEDIA
# ============================================================
def page_encyclopedia() -> None:
    page_head("Ensiklopedia lesi", "Delapan kelas yang dikenali model, beserta tatalaksana dan tanda bahayanya.")

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
def page_settings(user: dict, weights: Optional[Path]) -> None:
    page_head("Pengaturan", "Profil, keamanan, model, dan pengelolaan data akun Anda.")

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
            rows = [
                ("Paket ultralytics", "terpasang" if YOLO else "belum terpasang"),
                ("Arsitektur aktif", st.session_state.model_version),
                ("Berkas bobot", weights.name if weights else "tidak ditemukan"),
                ("Ambang keyakinan", f"{st.session_state.conf_thr:.2f}"),
                ("Ambang IoU", f"{st.session_state.iou_thr:.2f}"),
            ]
            st.markdown("".join(
                f"<div class='det-row'><span class='det-sub'>{k}</span><span class='det-name'>{v}</span></div>"
                for k, v in rows), unsafe_allow_html=True)

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
    model = load_model(st.session_state.model_version, str(weights) if weights else "")
    render_sidebar(user, weights)

    page = st.session_state.page
    if page == "Ringkasan":
        page_overview(user)
    elif page == "Skrining":
        page_screening(user, model, weights)
    elif page == "Pasien":
        page_patients(user)
    elif page == "Rekam medis":
        page_records(user)
    elif page == "Analitik":
        page_analytics(user)
    elif page == "Ensiklopedia":
        page_encyclopedia()
    elif page == "Pengaturan":
        page_settings(user, weights)
    else:
        page_overview(user)

    st.markdown(
        f"<hr><p style='font-size:.78rem;color:var(--muted);text-align:center'>"
        f"{APP_NAME} v{APP_VERSION} · alat bantu skrining, bukan alat diagnosis. "
        f"Konfirmasi setiap temuan dengan pemeriksaan klinis langsung.</p>",
        unsafe_allow_html=True,
    )


main()
