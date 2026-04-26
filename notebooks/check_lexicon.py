# ============================================================================
# notebooks/check_lexicon.py
# Tujuan: Menghitung kemunculan kata makian dalam dataset per kelas
# dan mencari kandidat kata pengganti untuk kata yang tidak ditemukan.
# ============================================================================

import os, sys
import pandas as pd
import numpy as np

# Load Data
df = pd.read_csv('D:/www/hate-speech/data/processed/df_clean.csv')
df['processed'] = df['processed'].fillna('')

lexicon = [
    "anjing", "babi", "monyet", "bangsat", "goblok", "bego", 
    "tolol", "idiot", "kampret", "keparat", "jancuk", "kontol", 
    "memek", "lonte", "anjir", "pekok", "bajingan", "asu", 
    "sial", "brengsek", "sundal", "bedebah", "sialan", "njir", "jancok", "perek"
]

print("="*75)
print(f"{'Kata':<15} | {'Abusive':>8} | {'Hate Speech':>11} | {'Normal':>6} | {'Total':>5} | {'Dominan':<10}")
print("-" * 75)

for word in lexicon:
    mask = df['processed'].str.contains(r'\b' + word + r'\b', regex=True, na=False)
    sub = df[mask]
    n_ab = (sub['label'] == 'Abusive').sum()
    n_hs = (sub['label'] == 'Hate Speech').sum()
    n_nm = (sub['label'] == 'Normal').sum()
    total = n_ab + n_hs + n_nm
    
    dominant = "-"
    if total > 0:
        max_val = max(n_ab, n_hs, n_nm)
        if max_val == n_hs: dominant = "Hate Speech"
        elif max_val == n_ab: dominant = "Abusive"
        else: dominant = "Normal"
        
    print(f"{word:<15} | {n_ab:>8} | {n_hs:>11} | {n_nm:>6} | {total:>5} | {dominant:<10}")

print("\n" + "="*75)
print("Mencari kata kasar potensial lainnya berdasarkan frekuensi tinggi di dataset...")

# Pencarian frekuensi kata secara umum di kelas Abusive
from sklearn.feature_extraction.text import CountVectorizer

abusive_text = df[df['label'] == 'Abusive']['processed']
vec = CountVectorizer(ngram_range=(1,1), min_df=5)
counts = vec.fit_transform(abusive_text)
words = vec.get_feature_names_out()
freqs = counts.sum(axis=0).A1

word_freq = list(zip(words, freqs))
word_freq.sort(key=lambda x: x[1], reverse=True)

print("Top 30 kata paling sering di kelas Abusive:")
top_30 = word_freq[:30]
for w, f in top_30:
    print(f"  {w}: {f}")

