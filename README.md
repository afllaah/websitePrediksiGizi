# GiziBalita — Website Klasifikasi Status Gizi Balita

Website Flask + HTML/CSS/JavaScript yang menghubungkan input pengguna ke model KNN sungguhan yang dilatih dari CSV. Website **tidak** mengarang prediksi: tombol prediksi hanya aktif setelah dataset valid dilatih.

## Struktur folder

```text
website_gizi_balita/
├── app.py
├── requirements.txt
├── Procfile
├── README.md
├── data_balita.csv        # opsional, letakkan sendiri di folder ini
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── app.js
└── uploads/               # dibuat otomatis
```

## Format CSV wajib

Header kolom harus tepat seperti berikut:

```csv
Umur (bulan),Jenis Kelamin,Tinggi Badan (cm),Status Gizi
```

Kolom `Jenis Kelamin` menerima `laki-laki`/`perempuan` (juga variasi umum `laki laki`, `male`, `female`, `l`, `p`). Kolom `Status Gizi` adalah label target yang akan dipelajari dari data. Setiap label perlu memiliki contoh yang cukup; dataset yang terlalu kecil atau hanya punya satu kelas akan ditolak. Pastikan data sumber tepercaya dan izin penggunaannya sesuai.

## Jalankan lokal di Windows

1. Instal Python 3.10 atau lebih baru.
2. Ekstrak ZIP.
3. Letakkan `data_balita.csv` di folder yang sama dengan `app.py`, atau jalankan aplikasi lalu unggah CSV melalui menu **Dataset**.
4. Buka Terminal/CMD di folder project:

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

5. Buka `http://127.0.0.1:5000`.
6. Jika tidak meletakkan CSV di folder project, masuk menu **Dataset**, pilih `data_balita.csv`, lalu klik **Unggah & latih model**. Setelah pelatihan berhasil, fitur prediksi dan evaluasi akan aktif.

### Fitur
- Dashboard statistik dan distribusi label.
- Prediksi menggunakan KNN yang dilatih pada dataset aktif.
- Upload CSV dan pratinjau hingga 500 baris.
- Pemilihan K yang aman, standardisasi fitur, akurasi training/testing, confusion matrix dan classification report.
- Riwayat prediksi selama proses server berjalan serta unduh riwayat CSV.

## Deployment online (Render sebagai contoh)

1. Buat repository GitHub baru, lalu unggah file project. Jangan mengunggah data pribadi atau dataset sensitif ke repository publik.
2. Buat Web Service di Render dan hubungkan repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Atur environment variable `SECRET_KEY` dengan nilai acak yang kuat.
6. Deploy. Setelah situs aktif, buka menu Dataset dan unggah CSV untuk melatih model pada proses server tersebut.

**Catatan deployment:** filesystem lokal dan memori proses pada hosting dapat bersifat sementara. Upload dataset, model, dan riwayat dapat hilang saat instance restart/redeploy; beberapa instance juga tidak berbagi state yang sama. Untuk penggunaan serius, simpan dataset/model di object storage dan gunakan database. Tambahkan autentikasi, batasan akses, logging aman, dan perlindungan upload sebelum membuka aplikasi kepada publik.

## Hal teknis penting
- Backend memvalidasi kolom CSV dan input.
- KNN, scaler, median imputasi, dan mapping kategori gender berasal dari dataset aktif.
- Evaluasi memakai hold-out train/test split dengan `random_state=42`. Jika kelas memiliki sedikit sampel, stratifikasi/cross-validation bisa tidak dapat digunakan; nilai evaluasi pada dataset kecil tidak stabil.
- Riwayat saat ini disimpan di memori server dan tidak permanen.
- Model hanya belajar dari label pada dataset. Jika label/status gizi sumber salah atau tidak mewakili standar antropometri yang sesuai umur dan jenis kelamin, model dapat menghasilkan prediksi yang keliru.
- **Ini proyek pembelajaran, bukan alat diagnosis atau skrining klinis.** Jangan membuat keputusan kesehatan berdasarkan hasil website saja; konsultasikan pengukuran dan pertumbuhan balita kepada tenaga kesehatan.
