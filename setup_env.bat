@echo off
REM ============================================================================
REM SETUP ENVIRONMENT - Jalankan di Anaconda Prompt
REM ============================================================================
REM Cara pakai:
REM   1. Buka Anaconda Prompt
REM   2. cd D:\www\hate-speech
REM   3. setup_env.bat
REM ============================================================================

echo ============================================
echo   HATE SPEECH DETECTION - SETUP ENVIRONMENT
echo ============================================
echo.

REM --- Step 1: Buat conda environment ---
echo [1/7] Membuat conda environment (Python 3.10)...
call conda create -n hate-speech python=3.10 -y
if errorlevel 1 (
    echo Environment mungkin sudah ada, lanjut...
)

REM --- Step 2: Aktifkan environment ---
echo.
echo [2/7] Mengaktifkan environment...
call conda activate hate-speech

REM --- Step 3: Install ML packages via conda ---
echo.
echo [3/7] Install scikit-learn, numpy, pandas, dll via conda-forge...
call conda install -c conda-forge scikit-learn=1.3.0 numpy pandas scipy matplotlib seaborn joblib statsmodels=0.14.0 nltk=3.8.1 -y

REM --- Step 4: Install NLP Indonesia via pip ---
echo.
echo [4/7] Install PySastrawi + NLTK...
pip install PySastrawi==1.2.0

REM --- Step 5: Install backend dependencies ---
echo.
echo [5/7] Install FastAPI + Uvicorn + Pydantic + Transformers...
pip install fastapi==0.103.1 "uvicorn[standard]==0.23.2" pydantic==2.3.0 python-multipart==0.0.6 transformers==4.41.2 torch --index-url https://download.pytorch.org/whl/cpu

REM --- Step 6: Install utilities + testing ---
echo.
echo [6/7] Install utilitas...
pip install python-dotenv==1.0.0 httpx==0.25.0 pytest ipykernel jupyter notebook

REM --- Step 7: Download NLTK data + register kernel ---
echo.
echo [7/7] Download NLTK stopwords + setup Jupyter kernel...
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
python -m ipykernel install --user --name hate-speech --display-name "Python (hate-speech)"

REM --- Verifikasi ---
echo.
echo ============================================
echo   VERIFIKASI INSTALASI
echo ============================================
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
python -c "import Sastrawi; print('Sastrawi: OK')"
python -c "import nltk; print('NLTK:', nltk.__version__)"
python -c "import fastapi; print('FastAPI:', fastapi.__version__)"
python -c "import statsmodels; print('statsmodels:', statsmodels.__version__)"
python -c "print(); print('=== SEMUA LIBRARY TERINSTALL ==='); print()"

echo.
echo ============================================
echo   SETUP SELESAI!
echo ============================================
echo.
echo Langkah selanjutnya:
echo   1. Download dataset ke data\raw\dataset_tweet.csv
echo   2. Jalankan: python notebooks\01_eda.py
echo   3. Jalankan: python notebooks\02_preprocessing.py
echo   4. Jalankan: python notebooks\03_modeling.py
echo.
pause
