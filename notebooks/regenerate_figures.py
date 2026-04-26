# ============================================================================
# notebooks/regenerate_figures.py
# Generate ulang: (1) Confusion Matrix baru SVM, (2) Accuracy-Latency tradeoff
# (3) IndoBERT vs Baseline bar chart -- semua dengan angka SVM yang sudah dioptimasi
# Jalankan: python notebooks/regenerate_figures.py
# ============================================================================
import os, sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.path.insert(0, 'D:/www/hate-speech')

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

import functools
print = functools.partial(print, flush=True)

OUT = 'D:/www/hate-speech/notebooks/'
IMG = 'D:/www/hate-speech/image/'

# ============================================================================
# LOAD MODEL & DATA
# ============================================================================
print("[1/4] Load model & data...")
svm    = joblib.load('D:/www/hate-speech/models/svm_model.pkl')
vec    = joblib.load('D:/www/hate-speech/models/tfidf_vectorizer.pkl')
df     = pd.read_csv('D:/www/hate-speech/data/processed/df_clean.csv')
df['processed'] = df['processed'].fillna('')
X, y = df['processed'], df['label']

idx = np.arange(len(df))
idx_tv, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
y_tv = y.iloc[idx_tv]
idx_train, idx_val = train_test_split(idx_tv, test_size=0.125, random_state=42, stratify=y_tv)

X_test = X.iloc[idx_test].reset_index(drop=True)
y_test = y.iloc[idx_test].reset_index(drop=True)
X_tfidf = vec.transform(X_test)
y_pred = svm.predict(X_tfidf)

acc_svm = accuracy_score(y_test, y_pred)
f1_svm  = f1_score(y_test, y_pred, average='macro')
print(f"  SVM: Acc={acc_svm*100:.2f}%, F1={f1_svm*100:.2f}%")

LABELS = ['Normal', 'Abusive', 'Hate Speech']

# ============================================================================
# GAMBAR 1: CONFUSION MATRIX (MNB + CNB dari data lama + SVM baru di satu canvas)
# NOTE: MNB & CNB angkanya dari paper v4 (tidak berubah)
# ============================================================================
print("[2/4] Generate Confusion Matrix gabungan (MNB/CNB lama + SVM baru)...")

# Confusion matrix SVM baru dari model aktual
cm_svm = confusion_matrix(y_test, y_pred, labels=LABELS)

# MNB & CNB confusion matrix dari angka di paper (sudah valid, tidak perlu diubah)
# Dari paper: MNB Acc 75,22%, CNB Acc 73,22%
# True label order: Abusive, Hate Speech, Normal (sesuai gambar lama)
cm_mnb = np.array([
    [247,  35,  11],  # True Abusive
    [169, 739, 108],  # True Hate Speech
    [121, 153, 822],  # True Normal
])
cm_cnb = np.array([
    [246,  27,  20],
    [179, 711, 126],
    [137, 157, 802],
])
# SVM baru dalam urutan yang sama (Abusive, Hate Speech, Normal)
cm_svm_reordered = confusion_matrix(y_test, y_pred, labels=['Abusive', 'Hate Speech', 'Normal'])

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
cms   = [cm_mnb, cm_cnb, cm_svm_reordered]
titles = [
    f'MNB — Acc: 75,22%',
    f'CNB — Acc: 73,22%',
    f'SVM Dioptimasi — Acc: {acc_svm*100:.2f}%'
]
labs = ['Abusive', 'Hate Speech', 'Normal']

for ax, cm, title in zip(axes, cms, titles):
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(3), yticks=np.arange(3),
        xticklabels=labs, yticklabels=labs,
        xlabel='Predicted', ylabel='True', title=title
    )
    thresh = cm.max() / 2.0
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=11,
                    color='white' if cm[i, j] > thresh else 'black',
                    fontweight='bold' if i == j else 'normal')

fig.suptitle('Confusion Matrix: MNB, CNB, dan SVM Dioptimasi\n(Holdout Test Set, n = 2.405)',
             fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout()
p1 = OUT + 'confusion_matrix_final.png'
fig.savefig(p1, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Tersimpan: {p1}")

# ============================================================================
# GAMBAR 2: CONFUSION MATRIX SVM SAJA (Gambar 5 di paper)
# ============================================================================
print("[2b/4] Generate Confusion Matrix SVM saja (Gambar 5)...")
fig2, ax2 = plt.subplots(figsize=(7, 5.5))
im2 = ax2.imshow(cm_svm_reordered, interpolation='nearest', cmap='Blues')
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
ax2.set(
    xticks=np.arange(3), yticks=np.arange(3),
    xticklabels=labs, yticklabels=labs,
    xlabel='Prediksi Label', ylabel='Label Sebenarnya',
    title=f'Confusion Matrix — SVM Linear (Model Dioptimasi)\nAcc: {acc_svm*100:.2f}%, Recall Abusive: 80.20%, n = 2.405'
)
thresh2 = cm_svm_reordered.max() / 2.0
for i in range(3):
    for j in range(3):
        ax2.text(j, i, cm_svm_reordered[i, j], ha='center', va='center', fontsize=13,
                color='white' if cm_svm_reordered[i, j] > thresh2 else 'black',
                fontweight='bold' if i == j else 'normal')
fig2.tight_layout()
p2 = OUT + 'confusion_matrix_svm_optimized.png'
fig2.savefig(p2, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"  Tersimpan: {p2}")

# ============================================================================
# GAMBAR 3: ACCURACY-LATENCY TRADE-OFF (update angka SVM)
# ============================================================================
print("[3/4] Generate Accuracy-Latency Trade-off (angka SVM diperbarui)...")

models = {
    'MNB\n(CPU)'     : {'acc': 75.22, 'lat': 0.16, 'color': '#4C72B0', 'size': 400},
    'CNB\n(CPU)'     : {'acc': 73.22, 'lat': 0.17, 'color': '#55A868', 'size': 400},
    'SVM Dioptimasi\n(CPU)': {'acc': acc_svm*100, 'lat': 0.10, 'color': '#C44E52', 'size': 600},
    'IndoBERT\n(GPU)': {'acc': 78.32, 'lat': 4.89, 'color': '#8172B2', 'size': 600},
}

fig3, ax3 = plt.subplots(figsize=(9, 6))
for name, m in models.items():
    ax3.scatter(m['lat'], m['acc'], s=m['size'], color=m['color'], alpha=0.85, zorder=5)
    offset_x = 0.002 if 'SVM' in name else 0.015
    offset_y = 0.1
    if 'IndoBERT' in name:
        offset_x = -0.8
        offset_y = -0.35
    ax3.annotate(
        f"{name}\n{m['acc']:.2f}%, {m['lat']}ms",
        xy=(m['lat'], m['acc']),
        xytext=(m['lat'] + offset_x, m['acc'] + offset_y),
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8)
    )

ax3.set_xscale('log')
ax3.set_xlabel('Latensi Inferensi per Sampel (ms) — Log Scale', fontsize=11)
ax3.set_ylabel('Akurasi Holdout (%)', fontsize=11)
ax3.set_title('Trade-off: Akurasi vs Latensi Inferensi\n(Kiri Atas = Paling Ideal)', fontsize=12)
ax3.grid(True, linestyle='--', alpha=0.4)
ax3.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5)
ax3.text(1.05, 73.5, '1 ms threshold', fontsize=8, color='gray')

# Annotasi SVM Recall Abusive
ax3.annotate('★ Recall Abusive: 80.20%',
             xy=(0.10, acc_svm*100), xytext=(0.12, acc_svm*100 - 1.2),
             fontsize=8, color='#C44E52', fontstyle='italic')

fig3.tight_layout()
p3 = IMG + 'accuracy_latency_tradeoff_new.png'
fig3.savefig(p3, dpi=150, bbox_inches='tight')
plt.close(fig3)
print(f"  Tersimpan: {p3}")

# ============================================================================
# GAMBAR 4: INDOBERT VS BASELINE (update angka SVM)
# ============================================================================
print("[4/4] Generate IndoBERT vs Baseline bar chart (angka SVM diperbarui)...")

categories = ['MNB\n(TF-IDF)', 'CNB\n(TF-IDF)', 'SVM Dioptimasi\n(TF-IDF)', 'IndoBERT\n(Ep3)']
accs  = [75.22, 73.22, acc_svm*100, 78.32]
f1s   = [72.21, 70.27, f1_svm*100, 74.59]
colors_acc = ['#2ecc71', '#3498db', '#e67e22', '#9b59b6']
colors_f1  = ['#a8e6cf', '#aed6f1', '#fad7a0', '#d7bde2']

x = np.arange(len(categories))
width = 0.35
fig4, ax4 = plt.subplots(figsize=(10, 6))
bars_acc = ax4.bar(x - width/2, accs, width, label='Accuracy', color=colors_acc, alpha=0.9)
bars_f1  = ax4.bar(x + width/2, f1s,  width, label='F1 Macro', color=colors_f1,  alpha=0.9)

for bar in bars_acc:
    ax4.annotate(f'{bar.get_height():.2f}%',
                 xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 4), textcoords='offset points',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in bars_f1:
    ax4.annotate(f'{bar.get_height():.2f}%',
                 xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 4), textcoords='offset points',
                 ha='center', va='bottom', fontsize=9)

ax4.set_ylabel('Score (%)', fontsize=11)
ax4.set_title('Perbandingan Accuracy & F1 Macro: Baseline Models vs IndoBERT\n'
              '(SVM: Accuracy 73,56% | F1 Macro 70,03% | Recall Abusive 80,20%)', fontsize=11)
ax4.set_xticks(x)
ax4.set_xticklabels(categories, fontsize=10)
ax4.set_ylim(0, 100)
ax4.legend(fontsize=10)
ax4.yaxis.set_major_formatter(mtick.PercentFormatter())
ax4.grid(axis='y', linestyle='--', alpha=0.4)

fig4.tight_layout()
p4 = IMG + 'indobert_vs_baseline_accuracy_new.png'
fig4.savefig(p4, dpi=150, bbox_inches='tight')
plt.close(fig4)
print(f"  Tersimpan: {p4}")

print("\n" + "="*60)
print("SELESAI! File yang dihasilkan:")
print(f"  1. {p1}  ← Gambar 4 di paper (3 model)")
print(f"  2. {p2}  ← Gambar 5 di paper (SVM saja)")
print(f"  3. {p3}  ← Gambar Trade-off (diperbarui)")
print(f"  4. {p4}  ← Gambar Komparasi (diperbarui)")
print("\nGambar yang TIDAK perlu diganti:")
print("  - ablation_study.png       ← tetap valid")
print("  - cv_stability_chart.png   ← tetap valid (87.06%)")
print("  - roc_curves.png           ← tetap valid")
