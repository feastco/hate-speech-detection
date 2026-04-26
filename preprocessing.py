# preprocessing.py
# PENTING: File ini digunakan BAIK di notebook training maupun backend API
# Pastikan path kamus_alay.json benar relatif terhadap direktori kerja

import re
import json
import os
import csv
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ── Tentukan base directory ──────────────────────────────────────────────────
# Menggunakan path absolut agar bisa diimport dari mana saja
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KAMUS_PATH = os.path.join(BASE_DIR, 'data', 'resources', 'new_kamusalay.csv')

# ── Inisialisasi (dijalankan sekali saat import) ─────────────────────────────
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Load KamusAlay
KAMUS_ALAY = {}
with open(KAMUS_PATH, encoding='latin-1') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) == 2:
            KAMUS_ALAY[row[0].strip()] = row[1].strip()

# Stopword Indonesia
STOPWORDS_ID = set(stopwords.words('indonesian'))
CUSTOM_STOPWORDS = {
    'yuk', 'nih', 'deh', 'dong', 'sih', 'lah', 'kan', 'ya', 'eh', 'oh',
    'ah', 'ih', 'ugh', 'wah', 'nah', 'hm'
}
STOPWORDS_ID.update(CUSTOM_STOPWORDS)

# Stemmer Sastrawi (Nazief-Adriani algorithm)
factory = StemmerFactory()
STEMMER = factory.create_stemmer()

print("preprocessing.py: semua resource berhasil dimuat.")

# ── 6 Fungsi Preprocessing ───────────────────────────────────────────────────

def step1_case_folding(text: str) -> str:
    """Langkah 1: Ubah semua huruf ke lowercase."""
    return text.lower()

def step2_cleaning(text: str) -> str:
    """Langkah 2: Hapus URL, mention, hashtag, token placeholder, emoji bytes, angka, tanda baca."""
    # 1. Hapus URL asli (http/https/www)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    # 2. Hapus mention asli (@username)
    text = re.sub(r'@\w+', '', text)
    # 3. Hapus hashtag
    text = re.sub(r'#\w+', '', text)
    # 4. Hapus token placeholder dataset (Ibrohim & Budi mengganti
    #    @username → USER, URL → URL, retweet → RT di dataset asli)
    text = re.sub(r'\buser\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\brt\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\burl\b', '', text, flags=re.IGNORECASE)
    # 5. Hapus hex byte residues dari encoding latin-1 (emoji → \xf0\x9f dll)
    text = re.sub(r'\\x[0-9a-fA-F]{2}', '', text)          # literal \xNN
    text = re.sub(r'[\x00-\x1f\x80-\xff]', '', text)       # actual bytes 0x80-0xFF
    # 6. Hapus angka
    text = re.sub(r'\d+', '', text)
    # 7. Hapus tanda baca dan karakter non-alfanumerik
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'_', ' ', text)
    # 8. Hapus token sangat pendek (1 karakter — sisa dari byte cleaning)
    text = re.sub(r'\b[a-zA-Z]\b', '', text)
    # 9. Normalisasi spasi
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def step3_tokenization(text: str) -> list:
    """Langkah 3: Pecah teks menjadi list token."""
    return text.split()

def step4_slang_normalization(tokens: list) -> list:
    """Langkah 4: Normalisasi kata slang/alay menggunakan KamusAlay."""
    normalized = []
    for token in tokens:
        replacement = KAMUS_ALAY.get(token, token)
        if replacement:  # skip jika replacement adalah string kosong
            normalized.append(replacement)
    return normalized

def step5_stopword_removal(tokens: list) -> list:
    """Langkah 5: Hapus stopword bahasa Indonesia."""
    return [t for t in tokens if t not in STOPWORDS_ID and len(t) > 1]

def step6_stemming(tokens: list) -> list:
    """Langkah 6: Stemming dengan Sastrawi (Nazief-Adriani)."""
    return [STEMMER.stem(t) for t in tokens]

def preprocess_pipeline(text: str):
    """
    Pipeline lengkap 6 langkah preprocessing.

    Returns:
        List token hasil preprocessing, atau None jika:
        - Input bukan string atau kosong
        - Hasil akhir < 3 token (terlalu pendek untuk klasifikasi)
    """
    if not isinstance(text, str) or not text.strip():
        return None

    text = step1_case_folding(text)
    text = step2_cleaning(text)
    tokens = step3_tokenization(text)
    tokens = step4_slang_normalization(tokens)
    tokens = step5_stopword_removal(tokens)
    tokens = step6_stemming(tokens)

    if len(tokens) < 3:
        return None

    return tokens

def preprocess_to_string(text: str):
    """
    Wrapper: kembalikan hasil preprocessing sebagai string.
    Digunakan sebagai input ke TF-IDF Vectorizer.

    Returns:
        String token dipisah spasi, atau None jika gagal.
    """
    tokens = preprocess_pipeline(text)
    if tokens is None:
        return None
    return ' '.join(tokens)


def preprocess_pipeline_fast(text: str):
    """
    Pipeline 5 langkah TANPA stemming (Step 1-5 saja).

    Digunakan untuk cross-dataset evaluation pada IndoToxic2024
    agar lebih efisien (~50x lebih cepat dari pipeline penuh).

    Catatan metodologis: Ablation study menunjukkan delta akurasi
    antara pipeline penuh vs no-stemming hanya -0.95pp (CNB),
    sehingga omisi stemming tidak mengubah kesimpulan lintas-dataset.

    Returns:
        String token dipisah spasi, atau None jika hasil < 3 token.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    text = step1_case_folding(text)
    text = step2_cleaning(text)
    tokens = step3_tokenization(text)
    tokens = step4_slang_normalization(tokens)
    tokens = step5_stopword_removal(tokens)

    if len(tokens) < 3:
        return None

    return ' '.join(tokens)
