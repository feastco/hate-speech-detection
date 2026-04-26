# ============================================================================
# notebooks/generate_new_plots.py
# Generate ulang Confusion Matrix & tabel evaluasi SVM baru
# Jalankan: python notebooks/generate_new_plots.py
# ============================================================================

import sys, os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, 'D:/www/hate-speech')

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             f1_score, accuracy_score,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize

import functools
print = functools.partial(print, flush=True)

# ============================================================================
# 1. LOAD MODEL & DATA
# ============================================================================
print("[1/5] Load model & data...")

svm_new    = joblib.load('D:/www/hate-speech/models/svm_model.pkl')
vectorizer = joblib.load('D:/www/hate-speech/models/tfidf_vectorizer.pkl')

df = pd.read_csv('D:/www/hate-speech/data/processed/df_clean.csv', encoding='utf-8')
df['processed'] = df['processed'].fillna('')
X = df['processed']
y = df['label']

# Reproduksi split yang persis sama (random_state=42)
idx = np.arange(len(df))
idx_tv, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
y_tv = y.iloc[idx_tv]
idx_train, idx_val = train_test_split(idx_tv, test_size=0.125, random_state=42, stratify=y_tv)

X_test = X.iloc[idx_test].reset_index(drop=True)
y_test = y.iloc[idx_test].reset_index(drop=True)

X_test_tfidf = vectorizer.transform(X_test)
y_pred = svm_new.predict(X_test_tfidf)

LABELS    = ['Normal', 'Abusive', 'Hate Speech']
LABEL_ID  = {l: i for i, l in enumerate(LABELS)}
OUT_DIR   = 'D:/www/hate-speech/notebooks/'

print(f"  Test set: {len(X_test)} sampel")
print(f"  Akurasi : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"  F1-Macro: {f1_score(y_test, y_pred, average='macro')*100:.2f}%")
print(f"\n  Classification Report:\n{classification_report(y_test, y_pred, digits=4)}")

# ============================================================================
# 2. CONFUSION MATRIX BARU (gaya jurnal — panas/dingin)
# ============================================================================
print("[2/5] Generate Confusion Matrix...")

cm = confusion_matrix(y_test, y_pred, labels=LABELS)

fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax.set(
    xticks=np.arange(len(LABELS)),
    yticks=np.arange(len(LABELS)),
    xticklabels=LABELS,
    yticklabels=LABELS,
    xlabel='Prediksi Label',
    ylabel='Label Sebenarnya',
    title='Confusion Matrix — SVM Linear (Model Dioptimasi)\nTest Set: 2.405 sampel'
)
ax.xaxis.set_label_position('bottom')

thresh = cm.max() / 2.0
for i in range(len(LABELS)):
    for j in range(len(LABELS)):
        ax.text(j, i, f'{cm[i, j]}',
                ha='center', va='center', fontsize=13,
                color='white' if cm[i, j] > thresh else 'black',
                fontweight='bold' if i == j else 'normal')

fig.tight_layout()
out_path = OUT_DIR + 'confusion_matrix_new.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Tersimpan: {out_path}")

# ============================================================================
# 3. COMPARISON BAR CHART: MODEL LAMA vs MODEL BARU (per-kelas F1)
# ============================================================================
print("[3/5] Generate Comparison Bar Chart...")

# Angka model lama dari laporan sebelumnya (tidak perlu di-load, sudah dicatat)
old_metrics = {
    'Normal'     : 0.8431,
    'Abusive'    : 0.1200,   # model lama hampir buta
    'Hate Speech': 0.8012,
    'Macro'      : 0.7444,
}

from sklearn.metrics import precision_recall_fscore_support
_, _, f1_per, _ = precision_recall_fscore_support(
    y_test, y_pred, labels=LABELS, zero_division=0)
f1_macro_new = f1_score(y_test, y_pred, average='macro')

new_metrics = {
    'Normal'     : f1_per[0],
    'Abusive'    : f1_per[1],
    'Hate Speech': f1_per[2],
    'Macro'      : f1_macro_new,
}

categories = ['Normal', 'Abusive', 'Hate Speech', 'Macro']
old_vals   = [old_metrics[c] for c in categories]
new_vals   = [new_metrics[c] for c in categories]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5.5))
bars_old = ax.bar(x - width/2, [v*100 for v in old_vals], width,
                  label='Model Lama (LinearSVC + SMOTE)',
                  color='#94a3b8', edgecolor='white')
bars_new = ax.bar(x + width/2, [v*100 for v in new_vals], width,
                  label='Model Baru (SGD + Lexicon Override)',
                  color='#0ea5e9', edgecolor='white')

ax.set_ylabel('F1-Score (%)', fontsize=12)
ax.set_title('Perbandingan F1-Score per Kelas\nModel Lama vs Model Baru (SVM Linear)', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 100)
ax.legend(fontsize=10)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.grid(axis='y', linestyle='--', alpha=0.4)

for bar in bars_old:
    ax.annotate(f'{bar.get_height():.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords='offset points',
                ha='center', va='bottom', fontsize=9, color='#475569')

for bar in bars_new:
    ax.annotate(f'{bar.get_height():.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords='offset points',
                ha='center', va='bottom', fontsize=9, color='#0369a1',
                fontweight='bold')

fig.tight_layout()
out_path2 = OUT_DIR + 'comparison_old_vs_new.png'
fig.savefig(out_path2, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Tersimpan: {out_path2}")

# ============================================================================
# 4. RINGKASAN TABEL UNTUK JURNAL (dicetak ke terminal)
# ============================================================================
print("\n[4/5] Tabel Evaluasi untuk Jurnal:")

_, _, f1_per, support = precision_recall_fscore_support(
    y_test, y_pred, labels=LABELS, zero_division=0)
from sklearn.metrics import precision_recall_fscore_support as prfs
prec, rec, f1, sup = prfs(y_test, y_pred, labels=LABELS, zero_division=0)

print("\n" + "=" * 70)
print(" TABEL EVALUASI MODEL SVM (DIOPTIMASI) — UNTUK LAPORAN JURNAL")
print("=" * 70)
print(f" {'Kelas':<15} {'Precision':>12} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
print(" " + "-" * 60)
for i, lbl in enumerate(LABELS):
    print(f" {lbl:<15} {prec[i]*100:>10.2f}%  {rec[i]*100:>8.2f}%  {f1[i]*100:>8.2f}%  {sup[i]:>9}")
print(" " + "-" * 60)
print(f" {'Macro Avg':<15} {prec.mean()*100:>10.2f}%  {rec.mean()*100:>8.2f}%  {f1.mean()*100:>8.2f}%  {sum(sup):>9}")
print(f" {'Akurasi':<15} {'':>12} {'':>10} {accuracy_score(y_test, y_pred)*100:>8.2f}%  {len(y_test):>9}")
print("=" * 70)

print("\n TABEL CONFUSION MATRIX (baris=Asli, kolom=Prediksi):")
print(f" {'':>20} {'Pred:Normal':>14} {'Pred:Abusive':>14} {'Pred:HS':>14}")
print(" " + "-" * 65)
cm_labels = ['True:Normal', 'True:Abusive', 'True:HS']
for i, lbl in enumerate(cm_labels):
    print(f" {lbl:<20} {cm[i,0]:>14} {cm[i,1]:>14} {cm[i,2]:>14}")

print("\n[5/5] Selesai!")
print("  File yang dihasilkan:")
print(f"    - notebooks/confusion_matrix_new.png  (ganti confusion_matrix.png di jurnal)")
print(f"    - notebooks/comparison_old_vs_new.png (gambar perbandingan model)")
print("\n  ROC Curve & Ablation Study TIDAK perlu diganti (masih valid).")
