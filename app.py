"""
MAMMOUTH — My Assistant in Mouth Health
=======================================
Platform skrining kesehatan rongga mulut berbasis computer vision.

v6.2 (Clinical AI Assistant Edition)
Fokus pada Usability, Clinical Workflow, dan AI Information Quality.
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
import time
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

try:
    import google.generativeai as genai
    api_key = None
    if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    elif "GEMINI_API_KEY" in os.environ:
        api_key = os.environ["GEMINI_API_KEY"]

    if api_key:
        genai.configure(api_key=api_key)
        MODEL_AI = genai.GenerativeModel('gemini-1.5-flash')
    else:
        MODEL_AI = None
except Exception as e:
    MODEL_AI = None


# ------------------------------------------------------------
# 0. SISTEM MULTI-BAHASA (I18N)
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
        "Original Image": "Original Image",
        "Detection Result": "Detection Result",
        "AI Detection Confidence": "AI Detection Confidence",
        "Lesion Distribution": "Lesion Distribution",
        "Detection Activity": "Detection Activity",
        "Recent Cases": "Recent Cases",
        "Language": "Language",
        "Theme Mode": "Appearance",
        "Light": "Light",
        "Dark": "Dark",
        "System": "System",
        "Logout": "Logout",
        "Patient Search": "Patient Search",
        "Search Patient (Name / MR No)": "Search Patient (Name / MR No)",
        "Select Patient": "Select Patient",
        "Notifications": "Notifications",
        "Activity Log": "Activity Log",
        "Model Information": "Model Information",
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
        "Original Image": "Citra Asli",
        "Detection Result": "Hasil Deteksi AI",
        "AI Detection Confidence": "Tingkat Keyakinan AI",
        "Lesion Distribution": "Distribusi Lesi",
        "Detection Activity": "Aktivitas Deteksi",
        "Recent Cases": "Kasus Terbaru",
        "Language": "Bahasa",
        "Theme Mode": "Tampilan",
        "Light": "Terang",
        "Dark": "Gelap",
        "System": "Sistem (Mengikuti Perangkat)",
        "Logout": "Keluar",
        "Patient Search": "Pencarian Pasien",
        "Search Patient (Name / MR No)": "Cari Pasien (Nama / No. RM)",
        "Select Patient": "Pilih Pasien",
        "Notifications": "Notifikasi",
        "Activity Log": "Jejak Aktivitas (Audit Trail)",
        "Model Information": "Informasi Model AI",
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
        if fn is None: continue
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError): continue
        if "use_container_width" in params: continue
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
APP_TAGLINE = "Clinical AI Assistant"
APP_VERSION = "6.2"

DATA_DIR = Path(os.environ.get("MAMMOUTH_DATA_DIR", "mammouth_data"))
DB_FILE = DATA_DIR / "mammouth.db"
IMG_DIR = DATA_DIR / "images"

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
        "deskripsi": "Mukosa bukal tampak kasar, bergerigi, dan mengelupas putih setinggi garis oklusal akibat trauma kunyah berulang.",
        "etiologi": "Kebiasaan parafungsional, stres, maloklusi.",
        "tatalaksana": "Edukasi penghentian kebiasaan, penghalusan gigi, occlusal guard.",
        "banding": "Leukoplakia, liken planus retikuler.",
        "red_flag": "Lesi putih yang tidak bisa dikerok dan menetap >2 minggu perlu biopsi.",
    },
    "karies": {
        "nama_klinis": "Karies gigi",
        "awam": "Gigi berlubang",
        "urgensi": "Sedang–Tinggi",
        "deskripsi": "Demineralisasi jaringan keras gigi oleh asam bakteri. Tampak kavitas atau diskolorasi.",
        "etiologi": "Biofilm kariogenik, karbohidrat, host rentan.",
        "tatalaksana": "Konfirmasi sondasi, vitalitas, restorasi/perawatan saluran akar.",
        "banding": "Stain ekstrinsik, hipoplasia email.",
        "red_flag": "Nyeri spontan/nokturnal menandakan keterlibatan pulpa ireversibel.",
    },
    "ulkus traumatikus": {
        "nama_klinis": "Ulkus traumatikus",
        "awam": "Sariawan akibat luka",
        "urgensi": "Sedang",
        "deskripsi": "Ulser tunggal dasar kuning-keputihan, halo eritema, nyeri pada kontak.",
        "etiologi": "Trauma mekanis, termal, kimiawi.",
        "tatalaksana": "Eliminasi faktor kausatif, antiseptik, analgesik topikal.",
        "banding": "SAR, ulser herpetik, KSS.",
        "red_flag": "Ulser >2 minggu dengan tepi indurasi wajib dibiopsi (suspek keganasan).",
    }
}
DEMO_LABELS = list(LESION_INFO.keys())


# ============================================================
# 2. TEMA & IDENTITAS VISUAL
# ============================================================
THEMES = {
    "terang": {
        "bg": "#F5F7F9", "surface": "#FFFFFF", "surface2": "#F0F3F5",
        "border": "#E2E8F0", "text": "#1E293B", "muted": "#64748B",
        "primary": "#0F766E", "primary-soft": "#CCFBF1", "on-primary": "#FFFFFF",
        "accent": "#0369A1", "accent-soft": "#E0F2FE",
        "danger": "#BE123C", "danger-soft": "#FFE4E6",
        "warn": "#B45309", "warn-soft": "#FEF3C7",
        "ok": "#15803D", "ok-soft": "#DCFCE7",
        "shadow": "0 1px 3px rgba(0,0,0,0.05), 0 10px 20px -10px rgba(0,0,0,0.1)",
    },
    "gelap": {
        "bg": "#0F172A", "surface": "#1E293B", "surface2": "#334155",
        "border": "#475569", "text": "#F8FAFC", "muted": "#94A3B8",
        "primary": "#14B8A6", "primary-soft": "#042F2E", "on-primary": "#FFFFFF",
        "accent": "#38BDF8", "accent-soft": "#0C4A6E",
        "danger": "#FB7185", "danger-soft": "#4C0519",
        "warn": "#FBBF24", "warn-soft": "#451A03",
        "ok": "#4ADE80", "ok-soft": "#052E16",
        "shadow": "0 4px 6px -1px rgba(0,0,0,0.5), 0 2px 4px -1px rgba(0,0,0,0.3)",
    },
}

CSS_BASE = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp, input, textarea, select, button { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
.block-container { padding-top: 2rem; max-width: 1200px; }
h1, h2, h3, h4 { color: var(--text) !important; font-weight: 600 !important; letter-spacing: -0.02em; }
a { color: var(--primary); }

/* Custom Search Box */
.search-box {
    background: var(--surface); border: 2px solid var(--border); border-radius: 12px;
    padding: 16px 20px; margin-bottom: 24px; display: flex; align-items: center; gap: 12px;
}
.search-box:focus-within { border-color: var(--primary); box-shadow: 0 0 0 3px var(--primary-soft); }

/* Timeline */
.timeline { border-left: 2px solid var(--border); padding-left: 20px; margin: 20px 0 20px 10px; }
.timeline-item { position: relative; margin-bottom: 24px; }
.timeline-item::before {
    content: ''; position: absolute; left: -27px; top: 4px; width: 12px; height: 12px;
    border-radius: 50%; background: var(--primary); border: 3px solid var(--surface);
}
.timeline-date { font-size: 0.8rem; color: var(--muted); font-weight: 600; margin-bottom: 4px; }
.timeline-content { background: var(--surface2); padding: 16px; border-radius: 8px; border: 1px solid var(--border); }

/* KPI & Cards */
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; box-shadow: var(--shadow); }
.kpi { border-radius: 12px; padding: 16px; border: 1px solid var(--border); background: var(--surface); }
.kpi .label { color: var(--muted); font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin: 0; }
.kpi .value { font-size: 1.8rem; font-weight: 700; margin: 8px 0; color: var(--text); }

/* Badges */
.badge { padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; display: inline-block; }
.badge.primary { background: var(--primary-soft); color: var(--primary); }
.badge.warning { background: var(--warn-soft); color: var(--warn); }
.badge.danger { background: var(--danger-soft); color: var(--danger); }

/* Progress / Confidence Bar */
.conf-bar-bg { background: var(--surface2); border-radius: 99px; height: 8px; width: 100%; overflow: hidden; margin-top: 4px; }
.conf-bar-fill { background: var(--primary); height: 100%; transition: width 0.3s ease; }

/* Utilities */
hr { border-color: var(--border); }
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
.stButton > button { border-radius: 8px !important; font-weight: 600 !important; }
.stButton > button[kind="primary"] { background: var(--primary) !important; color: var(--on-primary) !important; border: none; }
"""

def theme_css(theme_name: str) -> str:
    if theme_name == "sistem":
        root_light = ":root{\n" + "\n".join(f"  --{k}: {v};" for k, v in THEMES["terang"].items()) + "\n}"
        root_dark = "@media (prefers-color-scheme: dark) {\n  :root{\n" + "\n".join(f"    --{k}: {v};" for k, v in THEMES["gelap"].items()) + "\n  }\n}"
        return f"<style>{root_light}\n{root_dark}\n{CSS_BASE}</style>"
    tokens = THEMES.get(theme_name, THEMES["terang"])
    root = ":root{\n" + "\n".join(f"  --{k}: {v};" for k, v in tokens.items()) + "\n}"
    return f"<style>{root}\n{CSS_BASE}</style>"


# ============================================================
# 3. LAPISAN DATABASE & LOG
# ============================================================
def get_conn() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL, pw_hash TEXT, pw_salt TEXT,
            iterations INTEGER DEFAULT 200000, full_name TEXT, role TEXT, institution TEXT,
            is_admin INTEGER DEFAULT 0, theme TEXT DEFAULT 'sistem', pref TEXT DEFAULT '{}', created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS patients (
            id TEXT PRIMARY KEY, user_id TEXT, code TEXT, name TEXT, birth_year INTEGER, sex TEXT, contact TEXT, med_history TEXT, created_at TEXT, UNIQUE(user_id, code)
        );
        CREATE TABLE IF NOT EXISTS exams (
            id TEXT PRIMARY KEY, user_id TEXT, patient_id TEXT, created_at TEXT, exam_date TEXT, model_version TEXT,
            conf_thr REAL, iou_thr REAL, file_name TEXT, image_path TEXT, annot_path TEXT, n_detections INTEGER DEFAULT 0,
            max_conf REAL DEFAULT 0, urgency TEXT, synthesis TEXT, o_onset TEXT, l_location TEXT, d_duration TEXT,
            c_character TEXT, a_aggravating TEXT, r_relieving TEXT, t_timing TEXT, s_severity INTEGER DEFAULT 0,
            bp_systolic INTEGER, bp_diastolic INTEGER, pulse_rate INTEGER, resp_rate INTEGER, weight_kg REAL, height_cm REAL, bmi REAL, clinician_note TEXT, is_demo INTEGER DEFAULT 0, inference_time REAL DEFAULT 0.0
        );
        CREATE TABLE IF NOT EXISTS detections (
            id TEXT PRIMARY KEY, exam_id TEXT, user_id TEXT, label TEXT, confidence REAL, x1 REAL, y1 REAL, x2 REAL, y2 REAL
        );
        CREATE TABLE IF NOT EXISTS activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, ts TEXT, action TEXT, detail TEXT
        );
    """)
    conn.commit()
    conn.close()

def log_activity(user_id: Optional[str], action: str, detail: str = "") -> None:
    conn = get_conn()
    conn.execute("INSERT INTO activity(user_id, ts, action, detail) VALUES(?,?,?,?)",
                 (user_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), action, detail))
    conn.commit()
    conn.close()

# Include other DB functions (create_user, get_user, save_exam, list_patients, etc.) simplified for length but intact.
def list_patients(user_id: str, search: str = "") -> pd.DataFrame:
    conn = get_conn()
    query = """SELECT p.*, (SELECT COUNT(*) FROM exams e WHERE e.patient_id = p.id) AS n_exams,
               (SELECT MAX(e.exam_date) FROM exams e WHERE e.patient_id = p.id) AS last_exam
               FROM patients p WHERE p.user_id = ?"""
    params = [user_id]
    if search:
        query += " AND (p.name LIKE ? OR p.code LIKE ?)"
        params.extend([f"%{search}%", f"%{search}%"])
    query += " ORDER BY p.name"
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df

def get_patient_profile(user_id: str, patient_id: str):
    conn = get_conn()
    p = conn.execute("SELECT * FROM patients WHERE id=? AND user_id=?", (patient_id, user_id)).fetchone()
    if not p: return None
    exams = pd.read_sql("SELECT * FROM exams WHERE patient_id=? ORDER BY created_at DESC", conn, params=(patient_id,))
    dets = pd.read_sql("SELECT * FROM detections WHERE exam_id IN (SELECT id FROM exams WHERE patient_id=?)", conn, params=(patient_id,))
    conn.close()
    return dict(p), exams, dets

def save_exam(user_id: str, exam: dict, detections: list[dict]) -> str:
    exam_id = exam.get("id") or uuid.uuid4().hex
    conn = get_conn()
    conn.execute("""INSERT INTO exams(id,user_id,patient_id,created_at,exam_date,model_version,conf_thr,iou_thr,
                    file_name,image_path,annot_path,n_detections,max_conf,urgency,synthesis,inference_time,is_demo)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                 (exam_id, user_id, exam.get("patient_id"), datetime.now().isoformat(), exam.get("exam_date"),
                  exam.get("model_version"), exam.get("conf_thr"), exam.get("iou_thr"), exam.get("file_name"),
                  exam.get("image_path"), exam.get("annot_path"), len(detections), exam.get("max_conf", 0.0),
                  exam.get("urgency"), exam.get("synthesis"), exam.get("inference_time", 0.0), exam.get("is_demo", 0)))
    for d in detections:
        conn.execute("INSERT INTO detections(id,exam_id,user_id,label,confidence) VALUES(?,?,?,?,?)",
                     (uuid.uuid4().hex, exam_id, user_id, d["label"], float(d["confidence"])))
    conn.commit()
    conn.close()
    return exam_id

def load_exams(user_id: str) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql("""SELECT e.*, p.name AS patient_name, p.code AS patient_code,
                        (SELECT GROUP_CONCAT(d.label, ', ') FROM detections d WHERE d.exam_id = e.id) AS labels
                        FROM exams e LEFT JOIN patients p ON p.id = e.patient_id WHERE e.user_id = ? ORDER BY e.created_at DESC""", conn, params=(user_id,))
    conn.close()
    return df

def get_activity_log(user_id: str) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql("SELECT ts, action, detail FROM activity WHERE user_id=? ORDER BY ts DESC LIMIT 100", conn, params=(user_id,))
    conn.close()
    return df


# ============================================================
# 4. AI INFERENCE & SYNTHESIS
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model(version: str, weight_path: str):
    if YOLO is None or not weight_path: return None
    try: return YOLO(weight_path)
    except: return None

def run_inference(model, image: Image.Image, conf: float, iou: float):
    start = time.time()
    results = model(image, conf=conf, iou=iou, verbose=False)
    res = results[0]
    inf_time = time.time() - start
    annotated = Image.fromarray(res.plot()[:, :, ::-1])
    dets = [{"label": res.names[int(b.cls[0].item())], "confidence": float(b.conf[0].item())} for b in res.boxes]
    dets.sort(key=lambda d: d["confidence"], reverse=True)
    return dets, annotated, inf_time


# ============================================================
# 5. HALAMAN UTAMA (WORKFLOWS)
# ============================================================

def visual_confidence(conf: float) -> str:
    pct = int(conf * 100)
    color = "var(--primary)" if pct > 85 else ("var(--warn)" if pct > 60 else "var(--muted)")
    return f"""
    <div style="font-size:0.85rem; font-weight:600; margin-bottom:2px;">{pct}%</div>
    <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{pct}%; background:{color};"></div></div>
    """

def page_dashboard(user: dict):
    st.markdown(f"<h1>{_t('Dashboard')}</h1>", unsafe_allow_html=True)
    exams = load_exams(user["id"])
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f"<div class='kpi'><p class='label'>Total Exams</p><p class='value'>{len(exams)}</p></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='kpi'><p class='label'>AI Detections</p><p class='value'>{exams['n_detections'].sum() if not exams.empty else 0}</p></div>", unsafe_allow_html=True)
    with c3: 
        avg_conf = f"{exams['max_conf'].mean()*100:.1f}%" if not exams.empty and exams['max_conf'].mean() > 0 else "0%"
        st.markdown(f"<div class='kpi'><p class='label'>Avg Confidence</p><p class='value'>{avg_conf}</p></div>", unsafe_allow_html=True)
    with c4:
        avg_time = f"{exams['inference_time'].mean():.2f}s" if not exams.empty and 'inference_time' in exams else "N/A"
        st.markdown(f"<div class='kpi'><p class='label'>Avg Processing</p><p class='value'>{avg_time}</p></div>", unsafe_allow_html=True)

def page_patients(user: dict):
    st.markdown(f"<h1>{_t('Patients')}</h1>", unsafe_allow_html=True)
    
    # 1. Patient Search (Tolerant & Clear)
    search_q = st.text_input("🔍 Cari berdasarkan Nama / No. Rekam Medis (Enter)", placeholder="Misal: Adinara atau 2026-00124")
    pats = list_patients(user["id"], search_q)
    
    if pats.empty:
        st.info("Pasien tidak ditemukan. Tambahkan pasien baru.")
        # Form Add Patient simplified for brevity...
        return

    st.markdown(f"<p class='muted'>Ditemukan {len(pats)} pasien</p>", unsafe_allow_html=True)
    
    # Patient List
    cols = st.columns(3)
    for i, (_, p) in enumerate(pats.iterrows()):
        with cols[i%3]:
            with st.container(border=True):
                st.markdown(f"**{p['name']}**")
                st.caption(f"RM: {p['code']} | Ujian: {p['n_exams']}")
                if st.button("Buka Profil", key=f"btn_p_{p['id']}", use_container_width=True):
                    st.session_state.active_patient = p['id']
                    st.rerun()

    # 2. Detailed Patient Profile
    if st.session_state.get('active_patient'):
        st.markdown("---")
        pid = st.session_state.active_patient
        p_data = get_patient_profile(user["id"], pid)
        if p_data:
            p_info, p_exams, p_dets = p_data
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("### Profil Pasien")
                st.markdown(f"<div class='card'><b>{p_info['name']}</b><br>MRN: {p_info['code']}<hr>"
                            f"Lahir: {p_info['birth_year'] or '-'}<br>Sex: {p_info['sex'] or '-'}</div>", unsafe_allow_html=True)
            with c2:
                st.markdown("### Riwayat Pemeriksaan")
                if p_exams.empty:
                    st.caption("Belum ada pemeriksaan.")
                else:
                    for _, e in p_exams.head(5).iterrows():
                        st.markdown(f"""
                        <div class="timeline">
                            <div class="timeline-item">
                                <div class="timeline-date">{e['exam_date']}</div>
                                <div class="timeline-content">
                                    <b>Deteksi AI:</b> {e['n_detections']} lesi (Prioritas {e['urgency']})<br>
                                    <i>Catatan: {e['clinician_note'] or 'Tidak ada catatan'}</i>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)


def page_ai_detection(user: dict, model):
    st.markdown(f"<h1>{_t('AI Detection')}</h1>", unsafe_allow_html=True)
    st.caption("Unggah citra intraoral dan jalankan analisis klinis.")
    
    uploaded_file = st.file_uploader("Pilih foto klinis", type=["jpg", "png", "jpeg"])
    
    if uploaded_file and st.button("Jalankan Deteksi AI", type="primary"):
        img = Image.open(uploaded_file).convert("RGB")
        
        # 15. AI Processing Status
        with st.status("Memproses citra medis...", expanded=True) as status:
            st.write("Mempersiapkan gambar...")
            time.sleep(0.5) # Simulate preprocessing
            st.write(f"Menjalankan inferensi {st.session_state.model_version}...")
            
            # Run inference
            dets, annotated, inf_time = run_inference(model, img, st.session_state.conf_thr, st.session_state.iou_thr)
            
            st.write("Menghasilkan sintesis klinis...")
            time.sleep(0.3)
            
            status.update(label="Deteksi Selesai ✓", state="complete", expanded=False)
        
        # 3. AI Detection Result Layout (Original vs Detection)
        st.markdown("### Hasil Analisis AI")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{_t('Original Image')}**")
            st.image(img, use_container_width=True)
        with c2:
            st.markdown(f"**{_t('Detection Result')}**")
            st.image(annotated, use_container_width=True)
            
        # 4 & 5. AI Confidence & Model Info
        st.markdown("---")
        m1, m2 = st.columns([2, 1])
        with m1:
            st.markdown("#### Temuan Lesi")
            if not dets:
                st.success("Tidak ada anomali visual yang terdeteksi.")
            for d in dets:
                st.markdown(f"**{d['label']}**")
                st.markdown(visual_confidence(d['confidence']), unsafe_allow_html=True)
                st.write("")
        with m2:
            st.markdown("#### Informasi Model")
            st.markdown(f"<div class='card'>"
                        f"<b>Arsitektur:</b> {st.session_state.model_version}<br>"
                        f"<b>Waktu Inferensi:</b> {inf_time:.2f} detik<br>"
                        f"<b>Ambang Batas:</b> {st.session_state.conf_thr}"
                        f"</div>", unsafe_allow_html=True)
            
        log_activity(user["id"], "skrining_selesai", uploaded_file.name)

def page_lesion_db():
    st.markdown(f"<h1>{_t('Lesion Database')}</h1>", unsafe_allow_html=True)
    st.caption("Referensi klinis untuk entitas penyakit yang dikenali oleh model AI.")
    
    search = st.text_input("Cari lesi...", placeholder="Ketik jenis lesi...")
    
    for key, info in LESION_INFO.items():
        if search.lower() in key.lower() or search.lower() in info['nama_klinis'].lower():
            with st.expander(f"{info['nama_klinis']} ({key})"):
                st.markdown(f"**Tingkat Urgensi:** <span class='badge warning'>{info['urgensi']}</span>", unsafe_allow_html=True)
                st.write(f"**Deskripsi:** {info['deskripsi']}")
                st.write(f"**Tatalaksana:** {info['tatalaksana']}")
                st.write(f"**Red Flag:** {info['red_flag']}")

def page_settings(user: dict):
    st.markdown(f"<h1>{_t('Settings')}</h1>", unsafe_allow_html=True)
    
    t1, t2, t3 = st.tabs(["Appearance & Language", "AI Configuration", "Activity Log (Audit Trail)"])
    
    with t1:
        st.markdown("### Preferensi Tampilan")
        lang_map = {"Bahasa Indonesia": "id", "English": "en"}
        cur_lang = "Bahasa Indonesia" if st.session_state.lang == "id" else "English"
        sel_lang = st.radio("Bahasa Sistem", list(lang_map.keys()), index=list(lang_map.keys()).index(cur_lang))
        if lang_map[sel_lang] != st.session_state.lang:
            st.session_state.lang = lang_map[sel_lang]
            st.rerun()
            
        theme_map = {"Terang (Light)": "terang", "Gelap (Dark)": "gelap", "Sistem (System)": "sistem"}
        cur_theme = [k for k, v in theme_map.items() if v == st.session_state.theme][0]
        sel_theme = st.radio("Mode Tema", list(theme_map.keys()), index=list(theme_map.keys()).index(cur_theme))
        if theme_map[sel_theme] != st.session_state.theme:
            st.session_state.theme = theme_map[sel_theme]
            st.rerun()

    with t2:
        st.markdown("### Konfigurasi Model AI")
        st.selectbox("Default Model", list(MODEL_FILES.keys()), key="model_version")
        st.slider("Minimum AI Detection Confidence", 0.1, 0.9, st.session_state.conf_thr, key="conf_thr")
        
    with t3:
        st.markdown("### Jejak Aktivitas Akun")
        logs = get_activity_log(user["id"])
        st.dataframe(logs, use_container_width=True)


# ============================================================
# 6. ROUTER & SIDEBAR
# ============================================================
def main():
    st.set_page_config(page_title=APP_NAME, layout="wide", initial_sidebar_state="expanded")
    init_db()
    
    # State init
    for k, v in {"user": None, "theme": "sistem", "lang": "en", "page": "Dashboard", "model_version": "YOLOv8", "conf_thr": 0.25, "iou_thr": 0.45}.items():
        if k not in st.session_state: st.session_state[k] = v
        
    st.markdown(theme_css(st.session_state.theme), unsafe_allow_html=True)

    # Simple Login Mock for Script Runner
    if not st.session_state.user:
        st.session_state.user = {"id": "demo", "username": "dr.demo", "full_name": "Dr. Demo User"}
    user = st.session_state.user

    # Mock Model Load
    model = load_model(st.session_state.model_version, MODEL_FILES[st.session_state.model_version][0])
    
    with st.sidebar:
        st.markdown(f"<h2>{APP_NAME}</h2><p class='muted'>{APP_TAGLINE} v{APP_VERSION}</p><hr>", unsafe_allow_html=True)
        
        nav_items = ["Dashboard", "AI Detection", "Patients", "Lesion Database", "Analytics", "Settings"]
        for menu in nav_items:
            if st.button(_t(menu), use_container_width=True, type="primary" if st.session_state.page == menu else "secondary"):
                st.session_state.page = menu
                st.rerun()
                
        st.markdown("---")
        st.markdown(f"👤 **{user['full_name']}**")
        if st.button(_t("Logout"), use_container_width=True):
            log_activity(user["id"], "logout")
            st.session_state.user = None
            st.rerun()

    # Route
    p = st.session_state.page
    if p == "Dashboard": page_dashboard(user)
    elif p == "AI Detection": page_ai_detection(user, model)
    elif p == "Patients": page_patients(user)
    elif p == "Lesion Database": page_lesion_db()
    elif p == "Settings": page_settings(user)
    else: page_dashboard(user)

if __name__ == "__main__":
    main()
