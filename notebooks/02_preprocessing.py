# ============================================================================
# notebooks/02_preprocessing.py
# TAHAP 2: Preprocessing Dataset
# ============================================================================
# LIBRARIES: pandas, preprocessing (custom module)
# FILES: data/raw/dataset_tweet.csv → data/processed/df_clean.csv
# ============================================================================

import sys
sys.path.insert(0, 'D:/www/hate-speech')

import pandas as pd
import time

from preprocessing import preprocess_to_string

# ── Cell 1: Load Data + Konversi Label ───────────────────────────────────────
print("=" * 60)
print("TAHAP 2: PREPROCESSING DATASET")
print("=" * 60)

df = pd.read_csv('D:/www/hate-speech/data/raw/dataset_tweet.csv', encoding='latin-1')

def convert_to_single_label(row):
    """Prioritas: Hate Speech > Abusive > Normal"""
    if row['HS'] == 1:
        return 'Hate Speech'
    elif row['Abusive'] == 1:
        return 'Abusive'
    else:
        return 'Normal'

df['label'] = df.apply(convert_to_single_label, axis=1)

print(f"\nTotal tweet awal: {len(df)}")
print(df['label'].value_counts())

# ── Cell 2: Apply Preprocessing ──────────────────────────────────────────────
# Catatan: Sastrawi stemming lambat — 13.000 tweet bisa 5-15 menit di Windows
# Harap bersabar, ini normal!

print("\nMemulai preprocessing...")
print("Estimasi waktu: 5-15 menit (tergantung kecepatan CPU)")
print("Harap tunggu...\n")

t_start = time.time()

# Progress tracking
total = len(df)
processed_count = 0

def preprocess_with_progress(text):
    global processed_count
    processed_count += 1
    if processed_count % 1000 == 0 or processed_count == total:
        elapsed = time.time() - t_start
        pct = processed_count / total * 100
        remaining = (elapsed / processed_count) * (total - processed_count)
        print(f"  [{processed_count:>6}/{total}] {pct:5.1f}% — "
              f"elapsed: {elapsed/60:.1f}min — est. remaining: {remaining/60:.1f}min")
    return preprocess_to_string(text)

df['processed'] = df['Tweet'].apply(preprocess_with_progress)

t_elapsed = time.time() - t_start
print(f"\nPreprocessing selesai dalam {t_elapsed/60:.1f} menit")

# ── Cell 3: Laporan dan Simpan ───────────────────────────────────────────────
print(f"\nTotal sebelum eliminasi : {len(df)}")
print(f"Tweet None (<3 token)   : {df['processed'].isna().sum()}")

df_clean = df.dropna(subset=['processed']).reset_index(drop=True)
print(f"Total setelah eliminasi : {len(df_clean)}")

print(f"\nDistribusi label bersih:")
print(df_clean['label'].value_counts())
print(f"\nPersentase:")
print(df_clean['label'].value_counts(normalize=True).round(4) * 100)

# Simpan
df_clean.to_csv('D:/www/hate-speech/data/processed/df_clean.csv',
                index=False, encoding='utf-8')
print(f"\nDataset tersimpan ke: D:/www/hate-speech/data/processed/df_clean.csv")

# ── Cell 4: Cek Contoh Hasil ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CONTOH HASIL PREPROCESSING")
print("=" * 60)

sample = df_clean.sample(5, random_state=42)
for _, row in sample.iterrows():
    print(f"\nAsli   : {row['Tweet'][:80]}")
    print(f"Proses : {row['processed'][:80]}")
    print(f"Label  : {row['label']}")

print("\n" + "=" * 60)
print("✅ TAHAP 2 (PREPROCESSING) SELESAI")
print(f"Dataset bersih: {len(df_clean)} tweet")
print("=" * 60)
