
"""
Klinik AI RSGM — Integrated Oral Lesion Screening & OLD CARTS
==============================================================
Gabungan engine fungsional Streamlit + UI Frost modern dari dua versi sumber.

Fitur utama:
- Login operator
- Anamnesis OLD CARTS
- Batch upload + camera capture
- Real YOLO/Ultralytics inference bila best.pt tersedia
- Overlay bounding box + confidence
- Sintesis pendukung berbasis temuan AI + OLD CARTS
- Simpan hasil ke EMR lokal SQLite
- Filter EMR + ekspor CSV/Excel/JSON
- Dashboard analitik
- Referensi klinis edukatif
- Model status, threshold, dan session reset
- UI responsive bergaya Frost / Medical SaaS

CATATAN KLINIS:
Aplikasi ini adalah prototipe clinical decision support/screening.
Hasil AI bukan diagnosis final dan harus diverifikasi oleh dokter gigi.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import sqlite3
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ============================================================
# CONFIG
# ============================================================

APP_VERSION = "6.0 Fusion"
CLINIC_NAME = "RSGM Unjani"
USER_ROLE = "Clinical Clerkship (Koas Aktif)"

MODEL_PATH = Path(os.getenv("YOLO_MODEL_PATH", "best.pt"))
DB_FILE = Path(os.getenv("KLINIK_AI_DB", "klinik_ai_rsgm.sqlite3"))

DEFAULT_OPERATOR_ID = os.getenv("KLINIK_AI_OPERATOR_ID", "adinara savero")
DEFAULT_OPERATOR_PASSWORD = os.getenv("KLINIK_AI_OPERATOR_PASSWORD", "2560171013")

DB_COLUMNS = [
    "record_id",
    "case_id",
    "tanggal",
    "waktu",
    "nama_file",
    "lesi_terdeteksi",
    "confidence",
    "onset",
    "location",
    "duration",
    "character",
    "aggravating",
    "relieving",
    "timing",
    "severity",
    "suspek_diagnosis",
    "operator_id",
    "model_file",
]

LESION_INFO: dict[str, dict[str, str]] = {
    "cheek biting": {
        "nama_klinis": "Morsicatio Buccarum",
        "deskripsi": "Lesi traumatik akibat kebiasaan menggigit mukosa pipi berulang.",
        "rekomendasi": "Identifikasi dan eliminasi faktor trauma/parafungsi; evaluasi bila menetap.",
        "urgensi": "Rendah",
    },
    "coated tongue": {
        "nama_klinis": "Coated Tongue",
        "deskripsi": "Lapisan pada dorsum lidah yang dapat berkaitan dengan debris, higiene, atau faktor lokal.",
        "rekomendasi": "Optimalkan pembersihan lidah dan higiene oral; evaluasi bila menetap.",
        "urgensi": "Rendah",
    },
    "karies": {
        "nama_klinis": "Karies Gigi",
        "deskripsi": "Kerusakan jaringan keras gigi yang memerlukan konfirmasi pemeriksaan klinis sesuai indikasi.",
        "rekomendasi": "Lakukan pemeriksaan klinis dan penunjang sesuai indikasi untuk menentukan rencana perawatan.",
        "urgensi": "Sedang-Tinggi",
    },
    "linea alba": {
        "nama_klinis": "Linea Alba Buccalis",
        "deskripsi": "Garis putih pada mukosa bukal yang dapat berkaitan dengan tekanan atau friksi lokal.",
        "rekomendasi": "Identifikasi faktor friksi/parafungsi dan lakukan observasi klinis.",
        "urgensi": "Rendah",
    },
    "lingual varicosities": {
        "nama_klinis": "Lingual Varicosities",
        "deskripsi": "Pelebaran vena yang dapat tampak pada permukaan ventral lidah.",
        "rekomendasi": "Dokumentasikan dan evaluasi bila terdapat perubahan mendadak, nyeri, atau perdarahan.",
        "urgensi": "Rendah",
    },
    "stain calculus": {
        "nama_klinis": "Stain & Kalkulus",
        "deskripsi": "Noda ekstrinsik dan deposit kalkulus pada permukaan gigi.",
        "rekomendasi": "Pertimbangkan scaling profesional dan edukasi kebersihan mulut.",
        "urgensi": "Sedang",
    },
    "torus": {
        "nama_klinis": "Torus",
        "deskripsi": "Eksostosis tulang jinak yang umumnya asimtomatik.",
        "rekomendasi": "Observasi; evaluasi/rujuk bila mengganggu fungsi, trauma, atau rencana prostetik.",
        "urgensi": "Rendah",
    },
    "ulkus traumatikus": {
        "nama_klinis": "Ulkus Traumatikus",
        "deskripsi": "Ulkus mukosa yang dapat berkaitan dengan trauma lokal.",
        "rekomendasi": "Hilangkan faktor trauma dan evaluasi ulang; lesi persisten memerlukan pemeriksaan lanjutan.",
        "urgensi": "Sedang",
    },
    "olp": {
        "nama_klinis": "Oral Lichen Planus",
        "deskripsi": "Kondisi inflamasi mukosa oral yang memerlukan penilaian klinis menyeluruh.",
        "rekomendasi": "Pertimbangkan evaluasi dokter gigi/spesialis penyakit mulut untuk konfirmasi dan pemantauan.",
        "urgensi": "Tinggi",
    },
}


# ============================================================
# PAGE / SESSION
# ============================================================

st.set_page_config(
    page_title="Klinik AI RSGM",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)

SESSION_DEFAULTS = {
    "logged_in": False,
    "anamnesis_data": None,
    "current_case_id": None,
    "analysis_results": [],
    "analysis_images": {},
    "last_analysis_ts": None,
    "saved_result_keys": set(),
}

for key, value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CSS
# ============================================================

def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Work+Sans:wght@400;500;600;700;800&display=swap');

        :root {
            --violet: #A163F7;
            --blue: #6F88FC;
            --cyan: #45E3FF;
            --ink: #1E293B;
            --muted: #64748B;
            --paper: #F4F7F9;
            --side: #1A1D2D;
            --border: #E8EDF2;
            --success: #059669;
            --warning: #D97706;
            --danger: #DC2626;
        }

        html, body, [class*="css"] {
            font-family: "Work Sans", sans-serif !important;
        }

        .stApp {
            background: var(--paper);
        }

        #MainMenu, footer, header {
            visibility: hidden;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }

        [data-testid="stSidebar"] {
            background: var(--side) !important;
        }

        [data-testid="stSidebar"] * {
            color: #CBD5E1 !important;
        }

        [data-testid="stSidebar"] .stMarkdown h1,
        [data-testid="stSidebar"] .stMarkdown h2,
        [data-testid="stSidebar"] .stMarkdown h3,
        [data-testid="stSidebar"] strong {
            color: white !important;
        }

        .fade-up {
            animation: fadeUp .45s ease both;
        }

        @keyframes fadeUp {
            from {opacity: 0; transform: translateY(12px);}
            to {opacity: 1; transform: translateY(0);}
        }

        .clinical-card {
            background: #fff;
            border: 1px solid var(--border);
            border-radius: 20px;
            box-shadow: 0 8px 28px rgba(30,41,59,.06);
            padding: 22px;
            margin-bottom: 18px;
        }

        .hero-card {
            background:
                radial-gradient(circle at 88% 16%, rgba(69,227,255,.18), transparent 28%),
                radial-gradient(circle at 72% 78%, rgba(161,99,247,.17), transparent 34%),
                linear-gradient(135deg, #1A1D2D 0%, #3E4B81 52%, #5B55A4 100%);
            color: white;
            border-radius: 24px;
            padding: 28px;
            box-shadow: 0 16px 40px rgba(30,41,59,.15);
            margin-bottom: 20px;
        }

        .hero-title {
            color: white;
            font-size: 2rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -.03em;
        }

        .hero-subtitle {
            color: rgba(255,255,255,.82);
            margin: 8px 0 0 0;
            line-height: 1.6;
        }

        .section-kicker {
            color: var(--blue);
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: .1em;
            font-size: .78rem;
            margin-bottom: 4px;
        }

        .section-title {
            color: var(--ink);
            font-weight: 800;
            letter-spacing: -.025em;
            margin: 0;
            font-size: 2rem;
        }

        .section-subtitle {
            color: var(--muted);
            margin: 6px 0 18px 0;
        }

        .metric-card {
            background: #fff;
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 8px 24px rgba(30,41,59,.045);
        }

        .metric-label {
            color: var(--muted);
            font-size: .8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .06em;
        }

        .metric-value {
            color: var(--ink);
            font-size: 2rem;
            font-weight: 800;
            margin-top: 5px;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 6px 10px;
            border-radius: 999px;
            font-size: .73rem;
            font-weight: 800;
            white-space: nowrap;
        }

        .badge-low { background: #DCFCE7; color: #166534; }
        .badge-med { background: #FEF3C7; color: #92400E; }
        .badge-high { background: #FEE2E2; color: #991B1B; }
        .badge-neutral { background: #EEF2FF; color: #4338CA; }

        .notice {
            border-radius: 16px;
            padding: 15px 18px;
            margin: 0 0 18px 0;
            line-height: 1.5;
        }

        .notice-warning {
            background: #FFFBEB;
            border: 1px solid #FDE68A;
            color: #78350F;
        }

        .notice-info {
            background: #EFF6FF;
            border: 1px solid #BFDBFE;
            color: #1E3A8A;
        }

        .notice-success {
            background: #ECFDF5;
            border: 1px solid #A7F3D0;
            color: #065F46;
        }

        .model-ok {
            border-left: 4px solid #10B981;
            background: #ECFDF5;
            padding: 12px 14px;
            border-radius: 12px;
        }

        .model-bad {
            border-left: 4px solid #F59E0B;
            background: #FFFBEB;
            padding: 12px 14px;
            border-radius: 12px;
        }

        .result-title {
            color: var(--ink);
            font-size: 1.15rem;
            font-weight: 800;
        }

        .small-muted {
            color: var(--muted);
            font-size: .8rem;
        }

        .dropzone {
            border: 2px dashed #CFD9FF;
            border-radius: 18px;
            background: linear-gradient(180deg, #FCFDFF, #F7F9FF);
            padding: 20px;
            text-align: center;
            margin-top: 8px;
        }

        .divider {
            height: 1px;
            background: var(--border);
            margin: 16px 0;
        }

        .login-shell {
            min-height: 84vh;
        }

        .login-left {
            background:
              radial-gradient(circle at 83% 14%, rgba(255,255,255,.18), transparent 22%),
              radial-gradient(circle at 8% 87%, rgba(255,255,255,.11), transparent 24%),
              linear-gradient(145deg, #5477ED 0%, #8D65EE 58%, #45DFF7 145%);
            border-radius: 28px;
            padding: 40px;
            color: white;
            position: relative;
            overflow: hidden;
        }

        .login-glass {
            background: rgba(255,255,255,.14);
            border: 1px solid rgba(255,255,255,.2);
            backdrop-filter: blur(12px);
            border-radius: 18px;
            padding: 16px;
        }

        .stButton > button {
            border-radius: 12px !important;
            border: 0 !important;
            font-weight: 700 !important;
            min-height: 44px;
        }

        .stDownloadButton > button {
            border-radius: 12px !important;
            font-weight: 700 !important;
            min-height: 44px;
        }

        div[data-testid="stForm"] {
            border: 0 !important;
            padding: 0 !important;
            background: transparent !important;
        }

        [data-testid="stMetric"] {
            background: white;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 12px;
        }

        .sticky-note {
            background: #F6F4FF;
            border-radius: 14px;
            padding: 14px;
            border: 1px solid #E8DEFE;
        }

        @media (max-width: 900px) {
            .login-left { padding: 28px; }
            .hero-title { font-size: 1.55rem; }
            .section-title { font-size: 1.6rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# ============================================================
# AUTH
# ============================================================

def verify_login(operator_id: str, password: str) -> bool:
    return (
        hmac.compare_digest(operator_id.strip().lower(), DEFAULT_OPERATOR_ID.strip().lower())
        and hmac.compare_digest(password, DEFAULT_OPERATOR_PASSWORD)
    )


def render_login() -> None:
    st.markdown('<div class="fade-up">', unsafe_allow_html=True)
    left, right = st.columns([1.02, .98], gap="large")

    with left:
        st.markdown(
            f"""
            <div class="login-left login-shell">
                <div style="display:flex;align-items:center;gap:10px;font-weight:800;letter-spacing:.08em;">
                    <div class="login-glass" style="width:44px;height:44px;display:flex;align-items:center;justify-content:center;">✦</div>
                    <span>{CLINIC_NAME.upper()}</span>
                </div>

                <div style="margin-top:18vh;max-width:620px;">
                    <div style="font-size:.8rem;font-weight:800;letter-spacing:.18em;text-transform:uppercase;opacity:.92;">
                        SISTEM KLINIS TERINTEGRASI
                    </div>
                    <div style="font-size:2.55rem;font-weight:800;line-height:1.05;margin-top:10px;">
                        Hello RSGM Unjani!
                    </div>
                    <p style="font-size:1.03rem;line-height:1.65;color:rgba(255,255,255,.88);margin-top:18px;">
                        Satu ruang kerja untuk menyatukan anamnesis OLD CARTS,
                        dokumentasi visual, dan skrining lesi oral berbasis AI.
                    </p>

                    <div class="login-glass" style="margin-top:24px;">
                        <div style="font-weight:800;margin-bottom:5px;">🛡️ Clinical Decision Support</div>
                        <div style="font-size:.88rem;line-height:1.55;color:rgba(255,255,255,.84);">
                            Dirancang sebagai pendukung keputusan klinis. Setiap temuan
                            harus diverifikasi melalui pemeriksaan dan pertimbangan dokter gigi.
                        </div>
                    </div>
                </div>

                <div style="position:absolute;bottom:28px;left:40px;color:rgba(255,255,255,.72);font-size:.8rem;">
                    Klinik AI RSGM · {APP_VERSION}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("<div style='height:10vh'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="max-width:520px;margin:0 auto;">
                <div class="section-kicker">SELAMAT DATANG</div>
                <h1 class="section-title" style="font-size:1.9rem;">Masuk ke ruang klinis Anda</h1>
                <p class="section-subtitle">Gunakan identitas operator untuk mengakses dashboard skrining.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form", clear_on_submit=False):
            operator_id = st.text_input("Email / ID Operator", placeholder="contoh: adinara savero")
            password = st.text_input("Password", type="password", placeholder="••••••••")

            submit = st.form_submit_button("Masuk ke Dashboard", use_container_width=True)

            if submit:
                if verify_login(operator_id, password):
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Kredensial tidak valid. Periksa kembali ID operator dan password.")

        with st.expander("Akses Demo"):
            st.caption("Untuk prototipe lokal saja. Saat deployment nyata, pindahkan kredensial ke st.secrets atau environment variable.")
            st.code(f"ID: {DEFAULT_OPERATOR_ID}\nPassword: {DEFAULT_OPERATOR_PASSWORD}")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# DATABASE
# ============================================================

def db_connect() -> sqlite3.Connection:
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_FILE, timeout=20)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with db_connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS emr_records (
                record_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                tanggal TEXT NOT NULL,
                waktu TEXT NOT NULL,
                nama_file TEXT NOT NULL,
                lesi_terdeteksi TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0,
                onset TEXT,
                location TEXT,
                duration TEXT,
                character TEXT,
                aggravating TEXT,
                relieving TEXT,
                timing TEXT,
                severity INTEGER NOT NULL DEFAULT 0,
                suspek_diagnosis TEXT,
                operator_id TEXT,
                model_file TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_emr_date ON emr_records(tanggal)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_emr_lesion ON emr_records(lesi_terdeteksi)"
        )
        conn.commit()


def load_log() -> pd.DataFrame:
    init_db()
    query = """
        SELECT
            record_id AS "ID",
            case_id AS "Case ID",
            tanggal AS "Tanggal",
            waktu AS "Waktu",
            nama_file AS "Nama File",
            lesi_terdeteksi AS "Lesi Terdeteksi",
            confidence AS "Confidence",
            onset AS "O_Onset",
            location AS "L_Location",
            duration AS "D_Duration",
            character AS "C_Character",
            aggravating AS "A_Aggravating",
            relieving AS "R_Relieving",
            timing AS "T_Timing",
            severity AS "S_Severity",
            suspek_diagnosis AS "Suspek Diagnosis",
            operator_id AS "Operator",
            model_file AS "Model"
        FROM emr_records
        ORDER BY tanggal DESC, waktu DESC
    """
    with db_connect() as conn:
        df = pd.read_sql_query(query, conn)

    if "Confidence" in df.columns:
        df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce").fillna(0.0)
    if "S_Severity" in df.columns:
        df["S_Severity"] = pd.to_numeric(df["S_Severity"], errors="coerce").fillna(0).astype(int)
    return df


def append_records(records: list[dict[str, Any]]) -> int:
    if not records:
        return 0

    init_db()
    query = """
        INSERT INTO emr_records (
            record_id, case_id, tanggal, waktu, nama_file,
            lesi_terdeteksi, confidence, onset, location, duration,
            character, aggravating, relieving, timing, severity,
            suspek_diagnosis, operator_id, model_file
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    params = []
    for r in records:
        params.append(
            (
                r["record_id"],
                r["case_id"],
                r["tanggal"],
                r["waktu"],
                r["nama_file"],
                r["lesi_terdeteksi"],
                float(r.get("confidence", 0.0)),
                r.get("onset", ""),
                r.get("location", ""),
                r.get("duration", ""),
                r.get("character", ""),
                r.get("aggravating", ""),
                r.get("relieving", ""),
                r.get("timing", ""),
                int(r.get("severity", 0)),
                r.get("suspek_diagnosis", ""),
                r.get("operator_id", ""),
                r.get("model_file", ""),
            )
        )

    with db_connect() as conn:
        conn.executemany(query, params)
        conn.commit()

    return len(params)


# ============================================================
# MODEL HELPERS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    if YOLO is None:
        return None
    path = Path(model_path)
    if not path.exists():
        return None
    try:
        return YOLO(str(path))
    except Exception:
        return None


def model_name(model: Any) -> str:
    if model is None:
        return "Tidak tersedia"

    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return f"{len(names)} kelas terdeteksi"
    if isinstance(names, list):
        return f"{len(names)} kelas terdeteksi"
    return "YOLO custom model"


def class_name_from_model(model: Any, class_id: int) -> str:
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        return str(names.get(class_id, f"class_{class_id}"))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return f"class_{class_id}"


def plot_to_pil(result: Any) -> Image.Image:
    plotted = result.plot()
    if plotted is None:
        raise RuntimeError("Ultralytics tidak menghasilkan overlay gambar.")
    if isinstance(plotted, np.ndarray):
        # Ultralytics plot umumnya BGR; balik kanal ke RGB untuk Streamlit.
        return Image.fromarray(plotted[..., ::-1])
    return plotted


def extract_detections(result: Any, model: Any, conf_threshold: float) -> list[tuple[str, float]]:
    detections: list[tuple[str, float]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections

    for box in boxes:
        try:
            c_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
        except Exception:
            continue

        if conf >= conf_threshold:
            detections.append((class_name_from_model(model, c_id), conf))

    return sorted(detections, key=lambda x: x[1], reverse=True)


def get_lesion_info(name: str) -> dict[str, str]:
    key = str(name).strip().lower()
    if key in LESION_INFO:
        return LESION_INFO[key]

    return {
        "nama_klinis": str(name).title(),
        "deskripsi": "Informasi klinis belum tersedia pada katalog proyek.",
        "rekomendasi": "Lakukan evaluasi klinis mendalam oleh dokter gigi.",
        "urgensi": "Tidak Diketahui",
    }


# ============================================================
# CLINICAL SYNTHESIS
# ============================================================

def synthesize_clinical_diagnosis(
    detections: list[tuple[str, float]],
    anamnesis: dict[str, Any] | None,
) -> str:
    if not detections:
        if anamnesis:
            return (
                "Tidak ada temuan visual di atas threshold. "
                "Keluhan OLD CARTS tetap perlu dikorelasikan dengan pemeriksaan klinis."
            )
        return (
            "Tidak ada temuan visual di atas threshold. "
            "Lakukan evaluasi klinis sesuai keluhan pasien."
        )

    anamnesis = anamnesis or {}
    severity = int(anamnesis.get("severity", 0))
    character = str(anamnesis.get("character", "")).lower()
    timing = str(anamnesis.get("timing", "")).lower()

    results: list[str] = []

    for lesion_name, conf in detections:
        key = lesion_name.strip().lower()

        if key == "karies":
            if severity >= 7 or "denyut" in character or "malam" in timing:
                results.append(
                    "Suspek pulpitis berdasarkan temuan karies dan pola nyeri yang perlu dikonfirmasi."
                )
            elif severity >= 4 or "ngilu" in character:
                results.append(
                    "Suspek keterlibatan pulpa awal berdasarkan temuan karies dan keluhan nyeri."
                )
            else:
                results.append(
                    "Karies terdeteksi secara visual; status simptomatik perlu dikonfirmasi klinis."
                )

        elif key in {"ulkus traumatikus", "cheek biting", "linea alba"}:
            if severity >= 5:
                results.append(
                    f"Temuan reaktif/traumatik dengan keluhan nyeri {severity}/10; "
                    "faktor etiologi perlu dieliminasi."
                )
            else:
                results.append(
                    "Temuan reaktif/traumatik tampak tanpa keluhan nyeri berat berdasarkan anamnesis yang diisi."
                )

        elif key == "stain calculus":
            if "berdarah" in character or "gusi berdarah" in character:
                results.append(
                    "Stain/kalkulus dengan keluhan perdarahan; evaluasi periodontal diperlukan."
                )
            else:
                results.append(
                    "Stain/kalkulus terdeteksi; pertimbangkan evaluasi periodontal dan higiene oral."
                )

        else:
            info = get_lesion_info(lesion_name)
            results.append(f"Temuan visual: {info['nama_klinis']}; konfirmasi dengan pemeriksaan klinis.")

    return " ".join(results)


# ============================================================
# UI HELPERS
# ============================================================

def urgency_badge(urgency: str) -> str:
    if "Tinggi" in urgency:
        cls = "badge-high"
    elif "Sedang" in urgency:
        cls = "badge-med"
    elif urgency == "Rendah":
        cls = "badge-low"
    else:
        cls = "badge-neutral"
    return f'<span class="badge {cls}">Urgensi: {urgency}</span>'


def confidence_badge(conf: float) -> str:
    pct = conf * 100
    if pct >= 75:
        cls = "badge-low"
    elif pct >= 50:
        cls = "badge-med"
    else:
        cls = "badge-high"
    return f'<span class="badge {cls}">Confidence {pct:.1f}%</span>'


def page_header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="fade-up">
            <div class="section-kicker">{kicker}</div>
            <h1 class="section-title">{title}</h1>
            <p class="section-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def severity_band(sev: int) -> str:
    if sev <= 2:
        return "Ringan"
    if sev <= 5:
        return "Sedang"
    if sev <= 8:
        return "Berat"
    return "Sangat berat"


def reset_screening() -> None:
    st.session_state.anamnesis_data = None
    st.session_state.current_case_id = None
    st.session_state.analysis_results = []
    st.session_state.analysis_images = {}
    st.session_state.last_analysis_ts = None
    st.session_state.saved_result_keys = set()


# ============================================================
# LOGIN GATE
# ============================================================

if not st.session_state.logged_in:
    render_login()
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:12px;padding:4px 6px 20px 6px;border-bottom:1px solid rgba(255,255,255,.10);">
            <div style="width:42px;height:42px;border-radius:13px;background:linear-gradient(135deg,#A163F7,#45E3FF);
                        display:flex;align-items:center;justify-content:center;color:white;font-weight:800;">✦</div>
            <div>
                <div style="color:white;font-weight:800;">{CLINIC_NAME}</div>
                <div style="font-size:.72rem;color:#94A3B8;">Clinical Intelligence Workspace</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    menu = st.radio(
        "Navigasi",
        [
            "Dashboard Skrining",
            "Rekam Medis (EMR)",
            "Analitik Kinerja",
            "Referensi Klinis",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        """
        <div style="border-top:1px solid rgba(255,255,255,.10);margin-top:14px;padding-top:18px;">
            <div style="font-size:.72rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#94A3B8!important;">
                Pengaturan Model
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.55,
        step=0.05,
        format="%.2f",
    )

    iou_threshold = st.slider(
        "IoU Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        format="%.2f",
    )

    st.caption(f"Model: {MODEL_PATH.name}")
    st.caption(f"Status: {'Ultralytics tersedia' if YOLO is not None else 'Ultralytics belum terpasang'}")

    st.markdown(
        """
        <div style="margin-top:18px;padding-top:18px;border-top:1px solid rgba(255,255,255,.10);">
            <div style="display:flex;align-items:center;gap:10px;">
                <div style="width:40px;height:40px;border-radius:50%;background:#6F88FC;color:white;display:flex;align-items:center;justify-content:center;font-weight:800;">AS</div>
                <div>
                    <div style="color:white;font-weight:700;">Adinara Savero</div>
                    <div style="font-size:.72rem;color:#94A3B8;">Clinical Clerkship (Koas Aktif)</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Log out", use_container_width=True):
        for key, value in SESSION_DEFAULTS.items():
            st.session_state[key] = value
        st.rerun()


# ============================================================
# MODEL LOAD
# ============================================================

model = load_model(str(MODEL_PATH))


# ============================================================
# DASHBOARD SCREENING
# ============================================================

if menu == "Dashboard Skrining":
    page_header(
        "DASHBOARD SKRINING",
        "Skrining Lesi Oral",
        "Gabungan anamnesis OLD CARTS dan inferensi visual sebagai pendukung keputusan klinis.",
    )

    st.markdown(
        """
        <div class="notice notice-warning">
            <strong>⚠️ Peringatan klinis:</strong>
            sistem ini merupakan alat skrining/prototipe clinical decision support.
            Hasil AI bukan diagnosis final. Validasi melalui pemeriksaan langsung dan
            pertimbangan dokter gigi tetap diperlukan.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hero
    model_status = (
        f"Model aktif · {model_name(model)}"
        if model is not None
        else "Model belum siap · tambahkan best.pt dan instal ultralytics"
    )
    model_class = "model-ok" if model is not None else "model-bad"

    st.markdown(
        f"""
        <div class="hero-card">
            <div style="font-size:.76rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;opacity:.85;">
                {CLINIC_NAME} · AI ORAL SCREENING
            </div>
            <div class="hero-title">Clinical Intelligence Workspace</div>
            <div class="hero-subtitle">
                Dokumentasikan keluhan, analisis citra secara batch, lihat confidence,
                lalu simpan hasil terpilih ke EMR.
            </div>
            <div class="{model_class}" style="margin-top:18px;color:#164E63!important;">
                <strong style="color:#065F46!important;">Status Model:</strong>
                <span style="color:#334155!important;">{model_status}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # OLD CARTS
    left, right = st.columns([1.15, .85], gap="large")

    with left:
        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:18px;">
                <div style="width:38px;height:38px;border-radius:12px;background:#F0ECFF;color:#855AE5;display:flex;align-items:center;justify-content:center;">📋</div>
                <div>
                    <div class="result-title">Anamnesis OLD CARTS</div>
                    <div class="small-muted">Lengkapi keluhan subjektif pasien sebelum analisis.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        saved = st.session_state.anamnesis_data or {}

        with st.form("old_carts_form"):
            c1, c2 = st.columns(2, gap="large")
            with c1:
                onset = st.text_input(
                    "Onset (O)",
                    value=saved.get("onset", ""),
                    placeholder="Misal: 3 hari lalu, bertahap",
                )
                location = st.text_input(
                    "Location (L)",
                    value=saved.get("location", ""),
                    placeholder="Misal: mukosa bukal kanan",
                )
                duration = st.text_input(
                    "Duration (D)",
                    value=saved.get("duration", ""),
                    placeholder="Misal: terus-menerus / hilang timbul",
                )
                character = st.text_input(
                    "Character (C)",
                    value=saved.get("character", ""),
                    placeholder="Misal: tajam, tumpul, ngilu, berdarah",
                )

            with c2:
                aggravating = st.text_input(
                    "Aggravating (A)",
                    value=saved.get("aggravating", ""),
                    placeholder="Misal: makanan pedas",
                )
                relieving = st.text_input(
                    "Relieving (R)",
                    value=saved.get("relieving", ""),
                    placeholder="Misal: air dingin",
                )
                timing = st.text_input(
                    "Timing (T)",
                    value=saved.get("timing", ""),
                    placeholder="Misal: malam hari",
                )
                severity = st.slider(
                    "Severity (S) — Skala nyeri 0–10",
                    min_value=0,
                    max_value=10,
                    value=int(saved.get("severity", 0)),
                )

            save_anamnesis = st.form_submit_button(
                "Simpan Parameter Anamnesis",
                use_container_width=False,
            )

        if save_anamnesis:
            st.session_state.anamnesis_data = {
                "onset": onset.strip() or "Tidak diisi",
                "location": location.strip() or "Tidak diisi",
                "duration": duration.strip() or "Tidak diisi",
                "character": character.strip() or "Tidak diisi",
                "aggravating": aggravating.strip() or "Tidak diisi",
                "relieving": relieving.strip() or "Tidak diisi",
                "timing": timing.strip() or "Tidak diisi",
                "severity": severity,
            }
            st.success(f"Parameter tersimpan sementara · Severity: {severity}/10 ({severity_band(severity)})")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)

        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:18px;">
                <div style="width:38px;height:38px;border-radius:12px;background:#E7FAFF;color:#1688A4;display:flex;align-items:center;justify-content:center;">📸</div>
                <div>
                    <div class="result-title">Akuisisi Visual</div>
                    <div class="small-muted">JPG/PNG · batch upload atau kamera perangkat.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        upload_tab, camera_tab = st.tabs(["Unggah Batch", "Kamera Perangkat"])

        images: list[Image.Image] = []
        file_names: list[str] = []

        with upload_tab:
            uploaded_files = st.file_uploader(
                "Pilih satu atau beberapa citra",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                help="Pilih citra klinis yang cukup jelas dan dapat diinterpretasikan.",
            )
            if uploaded_files:
                for f in uploaded_files:
                    try:
                        images.append(Image.open(f).convert("RGB"))
                        file_names.append(f.name)
                    except Exception:
                        st.warning(f"Gagal membaca {f.name}.")

        with camera_tab:
            camera_file = st.camera_input("Ambil foto klinis")
            if camera_file is not None:
                try:
                    images.append(Image.open(camera_file).convert("RGB"))
                    file_names.append(f"Camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                except Exception:
                    st.warning("Citra dari kamera tidak dapat dibaca.")

        if images:
            st.caption(f"{len(images)} citra siap dianalisis.")
            cols = st.columns(min(4, len(images)))
            for idx, (img, fname) in enumerate(zip(images, file_names)):
                with cols[idx % len(cols)]:
                    st.image(img, caption=fname, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Analysis action
    current_case_id = st.session_state.current_case_id or f"CASE-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.metric("Case ID", current_case_id)
    with c2:
        st.metric("Threshold", f"{conf_threshold*100:.0f}%")
    with c3:
        st.caption(
            "IoU digunakan saat inferensi untuk mengendalikan tumpang tindih bounding box."
        )

    analyze_disabled = model is None or not images
    if st.button(
        "🚀 Jalankan Analisis AI & Sintesis Klinis",
        use_container_width=True,
        disabled=analyze_disabled,
    ):
        anamnesis = st.session_state.anamnesis_data

        if not anamnesis:
            st.warning(
                "Anamnesis belum disimpan. Analisis visual tetap dapat dijalankan, "
                "tetapi sintesis OLD CARTS tidak akan lengkap."
            )
            anamnesis_for_ai = {}
        else:
            anamnesis_for_ai = anamnesis

        st.session_state.current_case_id = current_case_id

        progress = st.progress(0, text="Menginisialisasi model...")
        results_payload: list[dict[str, Any]] = []
        image_store: dict[str, bytes] = {}

        for idx, (img, fname) in enumerate(zip(images, file_names)):
            try:
                progress.progress(
                    (idx + 1) / max(len(images), 1),
                    text=f"Menganalisis {fname}...",
                )

                yolo_results = model(
                    img,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False,
                )
                result = yolo_results[0]
                detections = extract_detections(result, model, conf_threshold)
                overlay = plot_to_pil(result)

                buffer_original = io.BytesIO()
                img.save(buffer_original, format="JPEG", quality=92)

                buffer_overlay = io.BytesIO()
                overlay.save(buffer_overlay, format="JPEG", quality=92)

                image_store[fname + "::original"] = buffer_original.getvalue()
                image_store[fname + "::overlay"] = buffer_overlay.getvalue()

                synthesis = synthesize_clinical_diagnosis(
                    detections,
                    anamnesis_for_ai,
                )

                results_payload.append(
                    {
                        "file_name": fname,
                        "detections": detections,
                        "synthesis": synthesis,
                        "model_file": MODEL_PATH.name,
                        "case_id": current_case_id,
                    }
                )

            except Exception as exc:
                results_payload.append(
                    {
                        "file_name": fname,
                        "detections": [],
                        "synthesis": f"Analisis gagal untuk citra ini: {exc}",
                        "model_file": MODEL_PATH.name,
                        "case_id": current_case_id,
                        "error": True,
                    }
                )

        st.session_state.analysis_results = results_payload
        st.session_state.analysis_images = image_store
        st.session_state.last_analysis_ts = datetime.now().isoformat(timespec="seconds")
        progress.progress(1.0, text="Analisis selesai.")
        st.rerun()

    if model is None:
        st.markdown(
            """
            <div class="notice notice-info">
                <strong>Model AI belum aktif.</strong>
                Letakkan file <code>best.pt</code> di folder yang sama dengan aplikasi
                dan instal dependensi <code>ultralytics</code>. Antarmuka dan EMR tetap dapat dibuka.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Results
    results_payload = st.session_state.analysis_results
    if results_payload:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        positives = sum(bool(r["detections"]) for r in results_payload)
        total_detections = sum(len(r["detections"]) for r in results_payload)
        st.markdown(
            f"""
            <div class="clinical-card">
                <div style="display:flex;justify-content:space-between;align-items:end;gap:12px;flex-wrap:wrap;">
                    <div>
                        <div class="section-kicker">HASIL ANALISIS</div>
                        <div class="result-title" style="font-size:1.45rem;">Skrining Visual & Sintesis</div>
                        <div class="small-muted">
                            {len(results_payload)} citra · {positives} citra dengan temuan · {total_detections} deteksi
                        </div>
                    </div>
                    <div class="badge badge-neutral">Demo AI / Decision Support</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for idx, payload in enumerate(results_payload):
            fname = payload["file_name"]
            detections = payload["detections"]
            synthesis = payload["synthesis"]
            error_state = payload.get("error", False)

            st.markdown('<div class="clinical-card fade-up">', unsafe_allow_html=True)

            h1, h2 = st.columns([3, 1])
            with h1:
                st.markdown(
                    f"<div class='result-title'>Hasil {idx+1} · {fname}</div>",
                    unsafe_allow_html=True,
                )
                st.caption(f"Case: {payload['case_id']} · Model: {payload['model_file']}")
            with h2:
                if detections:
                    best_conf = max(c for _, c in detections)
                    st.markdown(
                        f"<div style='text-align:right'>{confidence_badge(best_conf)}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div style='text-align:right'><span class='badge badge-neutral'>Tidak ada deteksi di atas threshold</span></div>",
                        unsafe_allow_html=True,
                    )

            c1, c2 = st.columns(2, gap="large")
            original_bytes = st.session_state.analysis_images.get(fname + "::original")
            overlay_bytes = st.session_state.analysis_images.get(fname + "::overlay")

            with c1:
                st.markdown("<div class='small-muted' style='font-weight:800;text-transform:uppercase;'>Citra Asli</div>", unsafe_allow_html=True)
                if original_bytes:
                    st.image(original_bytes, use_container_width=True)

            with c2:
                st.markdown("<div class='small-muted' style='font-weight:800;text-transform:uppercase;'>Overlay Deteksi</div>", unsafe_allow_html=True)
                if overlay_bytes:
                    st.image(overlay_bytes, use_container_width=True)

            if error_state:
                st.error(synthesis)
            else:
                if detections:
                    for lesion_name, conf in detections:
                        info = get_lesion_info(lesion_name)
                        st.markdown(
                            f"""
                            <div style="padding:14px 0;border-top:1px solid #E8EDF2;">
                                <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                                    {urgency_badge(info['urgensi'])}
                                    <strong style="font-size:1rem;color:#1E293B;">{info['nama_klinis']}</strong>
                                    {confidence_badge(conf)}
                                </div>
                                <div style="margin-top:8px;color:#64748B;line-height:1.55;font-size:.9rem;">
                                    {info['deskripsi']}
                                </div>
                                <div class="sticky-note" style="margin-top:10px;">
                                    <div style="font-size:.72rem;font-weight:800;color:#6D43CA;text-transform:uppercase;">Orientasi tindakan</div>
                                    <div style="margin-top:4px;color:#475569;font-size:.86rem;">{info['rekomendasi']}</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("Tidak ada objek yang melewati confidence threshold saat ini.")

                st.markdown(
                    f"""
                    <div class="sticky-note" style="margin-top:14px;">
                        <div style="font-size:.72rem;font-weight:800;color:#6D43CA;text-transform:uppercase;">
                            Sintesis Suspek Diagnosis
                        </div>
                        <div style="margin-top:5px;color:#334155;line-height:1.55;">
                            {synthesis}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                save_col, dl_col = st.columns([1.2, 1])

                save_key = f"save_result_{current_case_id}_{idx}"
                with save_col:
                    if st.button(
                        "💾 Simpan hasil ke EMR",
                        key=save_key,
                        use_container_width=True,
                        disabled=(save_key in st.session_state.saved_result_keys),
                    ):
                        now = datetime.now()
                        anam = st.session_state.anamnesis_data or {
                            "onset": "Tidak diisi",
                            "location": "Tidak diisi",
                            "duration": "Tidak diisi",
                            "character": "Tidak diisi",
                            "aggravating": "Tidak diisi",
                            "relieving": "Tidak diisi",
                            "timing": "Tidak diisi",
                            "severity": 0,
                        }

                        rows_to_save = []
                        if detections:
                            for lesion_name, conf in detections:
                                rows_to_save.append(
                                    {
                                        "record_id": f"EMR-{uuid.uuid4().hex[:10].upper()}",
                                        "case_id": current_case_id,
                                        "tanggal": now.strftime("%Y-%m-%d"),
                                        "waktu": now.strftime("%H:%M:%S"),
                                        "nama_file": fname,
                                        "lesi_terdeteksi": lesion_name,
                                        "confidence": round(conf, 4),
                                        "onset": anam["onset"],
                                        "location": anam["location"],
                                        "duration": anam["duration"],
                                        "character": anam["character"],
                                        "aggravating": anam["aggravating"],
                                        "relieving": anam["relieving"],
                                        "timing": anam["timing"],
                                        "severity": anam["severity"],
                                        "suspek_diagnosis": synthesis,
                                        "operator_id": DEFAULT_OPERATOR_ID,
                                        "model_file": MODEL_PATH.name,
                                    }
                                )
                        else:
                            rows_to_save.append(
                                {
                                    "record_id": f"EMR-{uuid.uuid4().hex[:10].upper()}",
                                    "case_id": current_case_id,
                                    "tanggal": now.strftime("%Y-%m-%d"),
                                    "waktu": now.strftime("%H:%M:%S"),
                                    "nama_file": fname,
                                    "lesi_terdeteksi": "Tidak terdeteksi",
                                    "confidence": 0.0,
                                    "onset": anam["onset"],
                                    "location": anam["location"],
                                    "duration": anam["duration"],
                                    "character": anam["character"],
                                    "aggravating": anam["aggravating"],
                                    "relieving": anam["relieving"],
                                    "timing": anam["timing"],
                                    "severity": anam["severity"],
                                    "suspek_diagnosis": synthesis,
                                    "operator_id": DEFAULT_OPERATOR_ID,
                                    "model_file": MODEL_PATH.name,
                                }
                            )

                        saved_count = append_records(rows_to_save)
                        st.session_state.saved_result_keys.add(save_key)
                        st.success(f"{saved_count} record tersimpan ke EMR.")
                with dl_col:
                    if overlay_bytes:
                        st.download_button(
                            "⬇️ Unduh overlay",
                            data=overlay_bytes,
                            file_name=f"overlay_{Path(fname).stem}.jpg",
                            mime="image/jpeg",
                            key=f"download_overlay_{current_case_id}_{idx}",
                            use_container_width=True,
                        )

            st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Detail teknis inferensi"):
            st.json(
                {
                    "case_id": st.session_state.current_case_id,
                    "model": MODEL_PATH.name,
                    "confidence_threshold": conf_threshold,
                    "iou_threshold": iou_threshold,
                    "analysis_timestamp": st.session_state.last_analysis_ts,
                    "files": [r["file_name"] for r in results_payload],
                }
            )

    if st.button("↩️ Reset Skrining", use_container_width=False):
        reset_screening()
        st.rerun()


# ============================================================
# EMR
# ============================================================

elif menu == "Rekam Medis (EMR)":
    page_header(
        "REKAM MEDIS",
        "Rekam Medis (EMR)",
        "Riwayat hasil skrining yang secara eksplisit telah disimpan ke basis data lokal.",
    )

    df = load_log()

    if df.empty:
        st.markdown(
            '<div class="notice notice-info"><strong>Belum ada record EMR.</strong> Simpan hasil analisis dari halaman Dashboard Skrining untuk mulai membangun riwayat.</div>',
            unsafe_allow_html=True,
        )
    else:
        positive = df[df["Lesi Terdeteksi"] != "Tidak terdeteksi"]
        avg_conf = positive["Confidence"].mean() * 100 if not positive.empty else 0

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Record", len(df))
        with m2:
            st.metric("Case ID Unik", df["Case ID"].nunique())
        with m3:
            st.metric("Record Positif", len(positive))
        with m4:
            st.metric("Rata-rata Confidence", f"{avg_conf:.1f}%")

        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
        f1, f2, f3, f4 = st.columns(4)

        with f1:
            lesion_options = sorted(df["Lesi Terdeteksi"].dropna().unique().tolist())
            selected_lesions = st.multiselect(
                "Temuan",
                lesion_options,
                default=lesion_options,
            )

        with f2:
            date_min = pd.to_datetime(df["Tanggal"], errors="coerce").min().date()
            start_date = st.date_input("Tanggal mulai", value=date_min)

        with f3:
            date_max = pd.to_datetime(df["Tanggal"], errors="coerce").max().date()
            end_date = st.date_input("Tanggal akhir", value=date_max)

        with f4:
            min_conf_pct = st.slider(
                "Confidence minimum (%)",
                0,
                100,
                0,
                5,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        mask = (
            df["Lesi Terdeteksi"].isin(selected_lesions)
            & (pd.to_datetime(df["Tanggal"], errors="coerce").dt.date >= start_date)
            & (pd.to_datetime(df["Tanggal"], errors="coerce").dt.date <= end_date)
            & (df["Confidence"].fillna(0) * 100 >= min_conf_pct)
        )
        filtered_df = df.loc[mask].copy()

        st.markdown(f"**{len(filtered_df)} record** sesuai filter.")
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
        )

        export_df = filtered_df.copy()

        csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")

        excel_buffer = io.BytesIO()
        try:
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                export_df.to_excel(writer, index=False, sheet_name="EMR")
            excel_bytes = excel_buffer.getvalue()
        except Exception:
            excel_bytes = None

        json_bytes = export_df.to_json(
            orient="records",
            force_ascii=False,
            date_format="iso",
        ).encode("utf-8")

        e1, e2, e3 = st.columns(3)
        with e1:
            st.download_button(
                "📥 Unduh CSV",
                data=csv_bytes,
                file_name="EMR_Klinik_AI_RSGM.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with e2:
            if excel_bytes:
                st.download_button(
                    "📥 Unduh Excel",
                    data=excel_bytes,
                    file_name="EMR_Klinik_AI_RSGM.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            else:
                st.warning("openpyxl belum tersedia.")
        with e3:
            st.download_button(
                "📥 Unduh JSON",
                data=json_bytes,
                file_name="EMR_Klinik_AI_RSGM.json",
                mime="application/json",
                use_container_width=True,
            )

        with st.expander("Ringkasan record terpilih"):
            st.dataframe(
                filtered_df.groupby("Lesi Terdeteksi")
                .agg(
                    Jumlah=("ID", "count"),
                    Rata_rata_Confidence=("Confidence", "mean"),
                    Rata_rata_Severity=("S_Severity", "mean"),
                )
                .reset_index()
                .sort_values("Jumlah", ascending=False),
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# ANALYTICS
# ============================================================

elif menu == "Analitik Kinerja":
    page_header(
        "ANALITIK KINERJA",
        "Analitik Kinerja",
        "Ringkasan distribusi hasil yang berasal dari record EMR yang tersimpan.",
    )

    df = load_log()

    if df.empty:
        st.info("Belum ada data tersimpan untuk dianalisis.")
    else:
        positive = df[df["Lesi Terdeteksi"] != "Tidak terdeteksi"].copy()

        total_records = len(df)
        unique_cases = df["Case ID"].nunique()
        positive_records = len(positive)
        mean_conf = positive["Confidence"].mean() * 100 if not positive.empty else 0
        mean_severity = df["S_Severity"].mean() if not df["S_Severity"].empty else 0

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Record", total_records)
        with m2:
            st.metric("Case", unique_cases)
        with m3:
            st.metric("Positif", positive_records)
        with m4:
            st.metric("Mean Confidence", f"{mean_conf:.1f}%")
        with m5:
            st.metric("Mean Severity", f"{mean_severity:.1f}/10")

        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
            st.subheader("Distribusi Temuan")
            counts = (
                positive["Lesi Terdeteksi"]
                .value_counts()
                .rename("Jumlah")
                .to_frame()
            )
            if not counts.empty:
                st.bar_chart(counts)
            else:
                st.info("Belum ada temuan positif.")
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
            st.subheader("Distribusi Severity")
            severity_counts = (
                df["S_Severity"]
                .value_counts()
                .sort_index()
                .rename("Jumlah")
                .to_frame()
            )
            st.bar_chart(severity_counts)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
        st.subheader("Tren Skrining Harian")
        trend = (
            pd.to_datetime(df["Tanggal"], errors="coerce")
            .dt.date.value_counts()
            .sort_index()
            .rename("Jumlah")
            .to_frame()
        )
        st.line_chart(trend)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
        st.subheader("Confidence per Temuan")
        if not positive.empty:
            confidence_summary = (
                positive.groupby("Lesi Terdeteksi")["Confidence"]
                .agg(["count", "mean", "min", "max"])
                .reset_index()
            )
            confidence_summary["mean"] = confidence_summary["mean"] * 100
            confidence_summary["min"] = confidence_summary["min"] * 100
            confidence_summary["max"] = confidence_summary["max"] * 100
            confidence_summary.columns = [
                "Temuan",
                "N",
                "Mean (%)",
                "Min (%)",
                "Max (%)",
            ]
            st.dataframe(
                confidence_summary.round(1),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Belum ada record positif.")
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# REFERENCES
# ============================================================

elif menu == "Referensi Klinis":
    page_header(
        "ENSIKLOPEDIA EDUKATIF",
        "Referensi Klinis",
        "Orientasi ringkas terhadap kelas temuan pada model proyek.",
    )

    st.markdown(
        """
        <div class="notice notice-info">
            Referensi ini bersifat edukatif. Jangan gunakan kartu ini sebagai pengganti
            pedoman klinis, pemeriksaan langsung, atau diagnosis dokter gigi.
        </div>
        """,
        unsafe_allow_html=True,
    )

    query = st.text_input(
        "Cari temuan",
        placeholder="Ketik: karies, ulkus, torus, lichen planus...",
    ).strip().lower()

    visible_items = []
    for key, info in LESION_INFO.items():
        haystack = " ".join(
            [
                key,
                info["nama_klinis"],
                info["deskripsi"],
                info["rekomendasi"],
            ]
        ).lower()
        if not query or query in haystack:
            visible_items.append((key, info))

    cols = st.columns(3)
    for idx, (_, info) in enumerate(visible_items):
        with cols[idx % 3]:
            st.markdown('<div class="clinical-card" style="height:100%;">', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="display:flex;justify-content:space-between;gap:8px;align-items:start;">
                    <div style="font-size:1.05rem;font-weight:800;color:#1E293B;">
                        {info['nama_klinis']}
                    </div>
                    {urgency_badge(info['urgensi'])}
                </div>
                <div style="margin-top:10px;color:#64748B;font-size:.88rem;line-height:1.55;">
                    {info['deskripsi']}
                </div>
                <div class="sticky-note" style="margin-top:13px;">
                    <div style="font-size:.7rem;font-weight:800;color:#6F88FC;text-transform:uppercase;">
                        Orientasi
                    </div>
                    <div style="margin-top:4px;color:#475569;font-size:.84rem;">
                        {info['rekomendasi']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    if not visible_items:
        st.warning("Temuan yang dicari tidak ditemukan pada katalog.")


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
    <div style="margin-top:30px;padding:16px 0;border-top:1px solid #E8EDF2;color:#94A3B8;font-size:.74rem;text-align:center;">
        {CLINIC_NAME} · Klinik AI RSGM · {APP_VERSION} · Clinical Decision Support Prototype
    </div>
    """,
    unsafe_allow_html=True,
)
