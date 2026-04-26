# Indonesian Hate Speech Detection System

Sistem cerdas berbasis Machine Learning dan Natural Language Processing (NLP) untuk mendeteksi *hate speech* (ujaran kebencian) dan bahasa kasar (*abusive language*) dalam teks berbahasa Indonesia.

## 📌 Fitur Utama
- **Klasifikasi Teks:** Menggunakan berbagai model yang telah dievaluasi seperti Support Vector Machine (SVM), IndoBERT, Complement Naive Bayes (CNB), dan Multinomial Naive Bayes (MNB).
- **Backend API:** Menggunakan **FastAPI** untuk melayani prediksi secara *real-time* dengan latensi rendah.
- **Frontend Interaktif:** Antarmuka pengguna modern yang dibangun menggunakan **Vue.js 3** dan **Tailwind CSS** (dilengkapi dengan *Dark Mode*).
- **Preprocessing:** Modul preprocessing teks khusus untuk bahasa Indonesia (termasuk normalisasi bahasa gaul/slang, *stemming*, dan penghapusan *stopwords*).

## 🗂 Struktur Proyek
- `backend/` : *Source code* untuk FastAPI server dan endpoint inferensi model.
- `frontend/` : *Source code* untuk antarmuka pengguna berbasis Vue.js 3.
- `models/` : Direktori penyimpanan untuk model Machine Learning yang telah dilatih.
- `data/` : Dataset yang digunakan untuk eksperimen dan pelatihan model.
- `notebooks/` : Jupyter notebooks untuk eksperimen, analisis data, evaluasi performa, dan proses *fine-tuning* model.
- `pipeline/` : Script dan laporan terkait alur kerja pemrosesan data dan pelatihan model.

## 🚀 Panduan Menjalankan Proyek (Local Development)

### Prasyarat
- **Python 3.8+**
- **Node.js & npm**

### 1. Setup Backend (FastAPI)
1. Buka terminal dan arahkan ke direktori root proyek.
2. Buat dan aktifkan *virtual environment* (Sangat disarankan):
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependensi Python (sesuaikan perintah `requirements.txt`):
   ```bash
   pip install -r requirements.txt 
   # Atau install manual dependensi seperti fastapi, uvicorn, scikit-learn, transformers, dll.
   ```

### 2. Persiapan Model (Penting!)
⚠️ **Catatan:** Direktori `models/` beserta file model di dalamnya (seperti `.pkl` atau model IndoBERT) **tidak disertakan** di dalam repositori ini karena batasan ukuran file di GitHub.

Sebelum menjalankan backend, Anda perlu menyiapkan modelnya dengan cara:
1. Menjalankan *script* pelatihan (misalnya melalui file di dalam folder `notebooks/` atau `pipeline/`) untuk men-*generate* ulang model secara lokal.
2. Memastikan model yang telah selesai dilatih disimpan ke dalam direktori `models/` yang ada di *root* proyek ini.

### 3. Menjalankan Backend
1. Setelah model tersedia, jalankan server FastAPI:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```
   *Backend API biasanya akan berjalan di `http://localhost:8000`*

### 4. Setup Frontend (Vue.js 3)
1. Buka terminal baru dan arahkan ke folder `frontend`:
   ```bash
   cd frontend
   ```
2. Install dependensi Node.js:
   ```bash
   npm install
   ```
3. Jalankan development server:
   ```bash
   npm run dev
   ```
4. Buka URL yang diberikan di terminal (biasanya `http://localhost:5173`) di browser Anda.
