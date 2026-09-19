# MAMMOUTH 7.0 "Rhythm Clinical"

**M**y **A**ssistant In **M**outh **H**ealth — asisten skrining kesehatan mulut berbasis AI (Streamlit).

## Menjalankan

```bash
pip install -r requirements.txt
streamlit run mammouth_app.py
```

Aplikasi akan membuat `mammouth.db` (SQLite) secara otomatis di folder yang sama saat pertama kali dijalankan.

## Model AI (opsional)

Untuk mengaktifkan deteksi otomatis, letakkan bobot YOLO pada folder yang sama dengan `mammouth_app.py`:

| Arsitektur | Nama berkas |
|---|---|
| YOLOv8  | `best.pt` |
| YOLOv11 | `yolov11_best.pt` |
| YOLOv12 | `yolov12_best.pt` |

Tanpa berkas model, aplikasi tetap berjalan penuh dalam **Mode Manual** — klinisi memilih temuan lesi secara langsung, dan seluruh fitur lain (anamnesis, pasien, EMR, analitik, ekspor) tetap aktif.

## Akun pertama

Akun pertama yang mendaftar melalui tab **"Buat Akun"** otomatis menjadi **Administrator**, dengan akses tambahan ke panel admin (ringkasan seluruh klinisi) di menu *Pengaturan Sistem* dan *Analitik Data*.

## Yang baru di versi 7.0 "Rhythm Clinical"

- **Penyimpanan citra** — foto pemeriksaan (setelah crop + enhancement) kini disimpan sebagai BLOB terkompresi di database, dengan thumbnail langsung di setiap kartu Riwayat EMR
- **Diagnosis banding** — setiap lesi kini membawa daftar diagnosis banding standar oral patologi, digabung otomatis dan disimpan per rekam pemeriksaan
- **Crop pada live preview** — slider margin atas/bawah/kiri/kanan sebelum brightness/contrast/sharpness, mendukung alur akuisisi citra yang lebih presisi
- **Desain ulang "Rhythm Clinical"** — palet teal/indigo/amber terinspirasi dasbor admin klinis, font Poppins, kartu KPI bergaya chip+pill berwarna, kartu chart dengan statistik mini, kartu spotlight klinisi, dan donat distribusi urgensi (CSS murni, tanpa dependensi tambahan)
- **Bilah atas persisten** — identitas aplikasi, pencarian cepat pasien/lesi (langsung lompat ke Riwayat EMR), info akun
- **Navigasi cepat kasus darurat** — tombol merah di sidebar yang muncul otomatis saat ada kasus berstatus "Perlu Tindak Lanjut", langsung membuka Riwayat EMR dengan filter status tersebut
- **Migrasi basis data ringan** — kolom baru (`image_blob`, `diagnosis_banding`) otomatis ditambahkan ke database versi lama tanpa menghapus data

## Yang baru di versi 6.0 "Aurora Clinical"

- Manajemen pasien (bukan cuma isolasi per-klinisi) dengan riwayat per pasien
- Hash password ber-*salt* (PBKDF2), bukan SHA-256 polos
- Live preview citra dengan penyesuaian brightness/contrast/sharpness
- Alur hibrida AI + konfirmasi manual sebelum data tersimpan ke EMR
- Anotasi bounding box otomatis pada citra saat deteksi AI aktif
- Riwayat EMR: pencarian, filter tanggal/status, ubah status, catatan, hapus
- Ekspor CSV, Excel, dan laporan PDF per pemeriksaan (jika `fpdf2` terpasang)
- Dashboard analitik: tren harian, frekuensi lesi, distribusi status, mode admin
- Ensiklopedia lesi dengan pencarian & filter kategori
- Pengaturan: profil, ubah password, preferensi threshold default, backup database, hapus data
- Desain ulang penuh ("Aurora Clinical": navy + koral) dengan halaman login dua-panel
