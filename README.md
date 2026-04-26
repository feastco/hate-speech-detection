<div align="center">
  <h1>🛡️ Indonesian Hate Speech Detection</h1>
  <p><i>Sistem Cerdas berbasis NLP untuk Mendeteksi Ujaran Kebencian & Bahasa Kasar</i></p>
  
  [![Vue.js](https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vue.js&logoColor=4FC08D)](#)
  [![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](#)
  [![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
  [![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](#)
  [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E?style=for-the-badge)](#)
</div>

<br/>

Selamat datang di repositori **Indonesian Hate Speech Detection**! Proyek ini mengkombinasikan kekuatan **Machine Learning tradisional (SVM, Naive Bayes)** dengan **Deep Learning (IndoBERT)** untuk menyeleksi dan mengklasifikasikan teks berbahasa Indonesia ke dalam kategori: `Normal`, `Hate Speech` (Ujaran Kebencian), atau `Abusive` (Kasar).

Semuanya dibalut dalam antarmuka web yang modern, interaktif, dan *blazing-fast*! 🚀

---

## ✨ Fitur Unggulan

- 🧠 **Multi-Model Support:** Dibekali dengan berbagai algoritma (Support Vector Machine, Complement Naive Bayes, Multinomial Naive Bayes, hingga IndoBERT).
- ⚡ **Latensi Super Rendah:** Backend ditenagai oleh FastAPI, mampu memberikan prediksi *real-time* dalam hitungan milidetik.
- 🎨 **UI/UX Modern:** Frontend Vue.js 3 + Tailwind CSS yang responsif, minimalis, dan mendukung fitur *Dark Mode*.
- 🧹 **Robust Preprocessing:** Mampu menangani *slang* (bahasa gaul Indonesia), *stemming* otomatis, hingga penghapusan *stopwords* sebelum teks masuk ke model.
- 🛡️ **Graceful Fallback:** Model IndoBERT opsional! Jika model IndoBERT (~400MB) tidak ada, sistem akan tetap berjalan mulus dan melayani *request* dengan menggunakan model klasik seperti **SVM** yang sangat cepat dan ringan.

---

## 🗂️ Struktur Direktori

```text
📦 hate-speech-detection
 ┣ 📂 backend/       👉 Source code FastAPI (API & Inference Logic)
 ┣ 📂 frontend/      👉 Source code Vue.js 3 & Tailwind CSS (Antarmuka Web)
 ┣ 📂 data/          👉 Dataset mentah & hasil pemrosesan
 ┣ 📂 notebooks/     👉 Jupyter Notebooks untuk riset, EDA, dan pelatihan model
 ┣ 📂 pipeline/      👉 Skrip evaluasi metodologi & validasi data
 ┣ 📂 pdf/           👉 Laporan, draft paper, dan dokumen pendukung
 ┗ 📂 models/        👉 [TIDAK DISERTAKAN DI GITHUB] Tempat Anda meletakkan file model .pkl
```

---

## 🚀 Tutorial Instalasi & Menjalankan Aplikasi

Ikuti langkah-langkah simpel di bawah ini untuk memutar *engine* deteksi *hate speech* di mesin lokal Anda.

### 📋 Prasyarat
Pastikan Anda sudah menginstal:
- **Python 3.8** atau lebih baru
- **Node.js (versi 16+)** beserta `npm`

---

### 🛠️ Tahap 1: Persiapan Model (Wajib!)
Karena besarnya ukuran file model (*terutama IndoBERT*), folder `models/` secara sengaja tidak kami unggah ke GitHub. Oleh karena itu, Anda harus me-*load* atau melatih ulang modelnya:

1. Buka folder `notebooks/` dan eksekusi *script* pelatihan (`03_modeling...`) untuk men-*generate* ulang file model ke direktori lokal Anda.
2. Pastikan file output (`svm_model.pkl`, `cnb_model.pkl`, `tfidf_vectorizer.pkl`, `top_terms.pkl`) tersimpan manis di dalam folder `models/` pada *root* proyek.

> **💡 Pro Tip:** Malas melatih model IndoBERT yang memakan waktu? Tidak masalah! Cukup latih model SVM dan TF-IDF saja. Server akan mendeteksi absennya IndoBERT dan berjalan otomatis menggunakan SVM sebagai model *default*!

---

### ⚙️ Tahap 2: Menjalankan Backend (FastAPI)

Mari kita nyalakan mesin utamanya! Buka terminal kesayangan Anda:

1. **Buat Virtual Environment (Sangat Disarankan)**
   ```bash
   # Pengguna Windows
   python -m venv venv
   venv\Scripts\activate

   # Pengguna Linux / Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Semua Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Nyalakan Server API**
   ```bash
   cd backend
   uvicorn main:app --reload
   ```
   🎉 *Voila! Backend Anda kini menyala! Anda dapat mengeksplorasi dan melakukan testing API secara langsung lewat dokumentasi interaktif bawaan Swagger UI di:* `http://localhost:8000/docs`.

---

### 🎨 Tahap 3: Menjalankan Frontend (Vue.js 3)

Buka **tab terminal baru** (dan pastikan terminal backend di atas tetap dibiarkan berjalan):

1. **Masuk ke direktori Frontend**
   ```bash
   cd frontend
   ```

2. **Install NPM Dependencies**
   ```bash
   npm install
   ```

3. **Jalankan Development Server**
   ```bash
   npm run dev
   ```
   🌐 *Selesai! Buka browser Anda dan kunjungi URL yang muncul di terminal (biasanya `http://localhost:5173`). Jangan lupa coba toggle Dark Mode-nya!*

---

## 🤝 Kontribusi
Punya ide keren untuk mengembangkan akurasi model? Atau ingin mempercantik UI lebih jauh? *Pull requests are highly appreciated!*
Untuk perubahan besar, silakan buat *Issue* terlebih dahulu agar bisa didiskusikan bersama.

---

<div align="center">
  Dibuat dengan ❤️ untuk Internet Indonesia yang lebih positif dan beradab.
</div>
