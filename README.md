# Klinik AI RSGM — Fusion v6.0

Aplikasi Streamlit yang menggabungkan engine YOLO/Ultralytics dari versi Python dengan UI Frost dan alur UX dari versi HTML.

## Struktur
- `klinik_ai_rsgm_merged.py` — aplikasi utama
- `requirements_klinik_ai_rsgm.txt` — dependensi
- `best.pt` — model YOLO custom, diletakkan di folder yang sama dengan aplikasi

## Menjalankan di Windows
```bash
py -m pip install -r requirements_klinik_ai_rsgm.txt
streamlit run klinik_ai_rsgm_merged.py
```

## Catatan model
Tanpa `best.pt`, aplikasi tetap dapat dibuka untuk melihat UI, EMR, analitik, dan referensi, tetapi analisis YOLO dinonaktifkan.

## Kredensial prototipe
Kredensial default masih mengikuti prototipe sumber. Untuk deployment nyata, gunakan environment variable:
- `KLINIK_AI_OPERATOR_ID`
- `KLINIK_AI_OPERATOR_PASSWORD`
- `YOLO_MODEL_PATH`
- `KLINIK_AI_DB`

## Catatan klinis
Aplikasi ini merupakan prototipe skrining/clinical decision support. Hasil model bukan diagnosis final dan wajib diverifikasi melalui pemeriksaan klinis oleh dokter gigi.

## Penyimpanan
EMR menggunakan SQLite lokal (`klinik_ai_rsgm.sqlite3`), sehingga lebih stabil untuk filter, pencarian, dan penyimpanan berulang dibanding CSV append-only.
