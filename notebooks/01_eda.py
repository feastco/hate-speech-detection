# ============================================================================
# notebooks/01_eda.py
# TAHAP 1: Exploratory Data Analysis (EDA)
# ============================================================================
# LIBRARIES: pandas, matplotlib, seaborn
# FILES: data/raw/dataset_tweet.csv → visualisasi ke notebooks/
# ============================================================================

import sys
sys.path.insert(0, 'D:/www/hate-speech')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Cell 1: Load Dataset ─────────────────────────────────────────────────────
print("=" * 60)
print("TAHAP 1: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

df = pd.read_csv('D:/www/hate-speech/data/raw/dataset_tweet.csv', encoding='latin-1')
print(f"\nShape       : {df.shape}")
print(f"Kolom       : {df.columns.tolist()}")
print(f"\nContoh data:\n{df.head()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nInfo tipe data:")
print(df.dtypes)

# ── Cell 2: Konversi Multi-label ke Single-label ─────────────────────────────
def convert_to_single_label(row):
    """Prioritas: Hate Speech > Abusive > Normal"""
    if row['HS'] == 1:
        return 'Hate Speech'
    elif row['Abusive'] == 1:
        return 'Abusive'
    else:
        return 'Normal'

df['label'] = df.apply(convert_to_single_label, axis=1)

print("\n" + "=" * 60)
print("DISTRIBUSI LABEL")
print("=" * 60)
print("\nJumlah per kelas:")
print(df['label'].value_counts())
print("\nPersentase per kelas:")
print(df['label'].value_counts(normalize=True).round(4) * 100)

# ── Cell 3: Visualisasi Distribusi Kelas ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#E24B4A', '#EF9F27', '#639922']
counts = df['label'].value_counts()

# Bar chart
axes[0].bar(counts.index, counts.values, color=colors, edgecolor='white')
axes[0].set_title('Distribusi Kelas Dataset', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Kelas Label')
axes[0].set_ylabel('Jumlah Tweet')
for i, (idx, val) in enumerate(counts.items()):
    axes[0].text(i, val + 30, f'{val:,}\n({val/len(df)*100:.1f}%)',
                 ha='center', fontsize=10, fontweight='bold')
axes[0].set_ylim(0, 6000)
axes[0].grid(axis='y', alpha=0.3)

# Pie chart
axes[1].pie(counts.values, labels=counts.index, colors=colors,
            autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title('Proporsi Kelas', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('D:/www/hate-speech/notebooks/distribusi_kelas.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("\nVisualisasi distribusi kelas tersimpan!")

# ── Cell 4: Statistik Panjang Tweet ──────────────────────────────────────────
df['tweet_length'] = df['Tweet'].str.len()
df['word_count'] = df['Tweet'].str.split().str.len()

print("\n" + "=" * 60)
print("STATISTIK PANJANG TWEET PER KELAS")
print("=" * 60)

for label in ['Hate Speech', 'Abusive', 'Normal']:
    subset = df[df['label'] == label]
    print(f"\n[{label}]")
    print(f"  Jumlah tweet  : {len(subset)}")
    print(f"  Panjang (char) - Mean: {subset['tweet_length'].mean():.1f}, "
          f"Median: {subset['tweet_length'].median():.1f}, "
          f"Max: {subset['tweet_length'].max()}")
    print(f"  Jumlah kata    - Mean: {subset['word_count'].mean():.1f}, "
          f"Median: {subset['word_count'].median():.1f}, "
          f"Max: {subset['word_count'].max()}")

# ── Cell 5: Visualisasi Panjang Tweet ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribusi panjang karakter per kelas
for label, color in zip(['Hate Speech', 'Abusive', 'Normal'], colors):
    subset = df[df['label'] == label]
    axes[0].hist(subset['tweet_length'], bins=50, alpha=0.6, label=label,
                 color=color, edgecolor='white')
axes[0].set_title('Distribusi Panjang Tweet (Karakter)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Jumlah Karakter')
axes[0].set_ylabel('Frekuensi')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Boxplot jumlah kata per kelas
df.boxplot(column='word_count', by='label', ax=axes[1])
axes[1].set_title('Distribusi Jumlah Kata per Kelas', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Kelas Label')
axes[1].set_ylabel('Jumlah Kata')
plt.suptitle('')  # Hapus judul default dari boxplot

plt.tight_layout()
plt.savefig('D:/www/hate-speech/notebooks/statistik_panjang_tweet.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("\nVisualisasi statistik panjang tweet tersimpan!")

print("\n" + "=" * 60)
print("✅ TAHAP 1 (EDA) SELESAI")
print(f"Total tweet: {len(df)}")
print("=" * 60)
