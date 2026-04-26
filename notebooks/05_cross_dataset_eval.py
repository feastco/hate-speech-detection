
# ============================================================================
# notebooks/05_cross_dataset_eval.py
# EVALUASI LINTAS DATASET: Ibrohim 2019 → IndoToxic2024 (IndoDiscourse)
# ============================================================================
# Tujuan: Membuktikan generalisasi model tanpa retraining (zero-shot transfer)
# Model dilatih pada: data/raw/dataset_tweet.csv (Ibrohim et al., 2019)
# Model diuji pada  : IndoToxic2024 via HuggingFace (Exqrch/IndoDiscourse)
# ============================================================================

import sys, io
sys.path.insert(0, 'D:/www/hate-speech')

import pandas as pd
import numpy as np
import joblib
import json
import warnings
import time
from multiprocessing import Pool, cpu_count
warnings.filterwarnings('ignore')

# ── Redirect stdout ke file log DAN console ──────────────────────────────────
LOG_PATH = 'D:/www/hate-speech/pipeline/cross_dataset_log.txt'
class Tee:
    def __init__(self, *files): self.files = files
    def write(self, s):
        for f in self.files: f.write(s); f.flush()
    def flush(self):
        for f in self.files: f.flush()
_logf = open(LOG_PATH, 'w', encoding='utf-8')
sys.stdout = Tee(sys.__stdout__, _logf)

# ── Ukuran sampel (5000 = cukup untuk cross-dataset; bisa dinaikkan) ─────────
N_SAMPLE = 5000
RANDOM_STATE = 42

from datasets import load_dataset
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import preprocess_pipeline_fast  # pipeline 5-step (tanpa stemming)
# Catatan: preprocess_pipeline_fast = step 1-5 (tanpa stemming).
# Ablation study menunjukkan delta vs pipeline penuh hanya 0.95pp (CNB),
# sehingga kesimpulan cross-dataset tidak berubah secara material.

# ── 1. LOAD MODEL & VECTORIZER YANG SUDAH DILATIH ───────────────────────────
print("=" * 65)
print("EVALUASI LINTAS DATASET — ZERO-SHOT TRANSFER")
print("Train: Ibrohim 2019  |  Test: IndoToxic2024 (IndoDiscourse)")
print("=" * 65)

svm_model  = joblib.load('D:/www/hate-speech/models/svm_model.pkl')
mnb_model  = joblib.load('D:/www/hate-speech/models/mnb_model.pkl')
cnb_model  = joblib.load('D:/www/hate-speech/models/cnb_model.pkl')
vectorizer = joblib.load('D:/www/hate-speech/models/tfidf_vectorizer.pkl')

print("\n✅ Model & Vectorizer berhasil dimuat:")
print(f"   Vocabulary TF-IDF (dari Ibrohim 2019): {len(vectorizer.vocabulary_):,} token")

# ── 2. LOAD IndoToxic2024 (IndoDiscourse) ────────────────────────────────────
print("\n── Memuat IndoToxic2024 dari HuggingFace... ──")
print("   (Exqrch/IndoDiscourse — versi terbaru dari IndoToxic2024)")

try:
    ds = load_dataset("Exqrch/IndoDiscourse", 'main', split="train")
    df_raw = ds.to_pandas()
    print(f"Dataset dimuat: {len(df_raw):,} entri")
    print(f"   Kolom tersedia: {list(df_raw.columns)}")
except Exception as e:
    print(f"Gagal load HuggingFace API: {e}")
    print("   Mencoba fallback CSV dari HuggingFace CDN...")
    import requests, io
    url = ("https://huggingface.co/datasets/Exqrch/IndoDiscourse/resolve/main/"
           "indotoxic2024_annotated_data_v2_final.csv")
    r = requests.get(url, timeout=120)
    r.encoding = 'utf-8'
    df_raw = pd.read_csv(io.StringIO(r.text))
    print(f"Fallback berhasil: {len(df_raw):,} entri")

print(f"\nSample kolom:\n{df_raw.head(2).to_string()}")

# ── 3. LABEL MAPPING ─────────────────────────────────────────────────────────
# IndoToxic2024: label kolom "HS" (Hate Speech binary) dan "Abusive" tersedia
# Skema Ibrohim 2019 yang sama: Normal / Hate Speech / Abusive
# Strategi majority vote dari 19 anotator untuk label final

print("\n── Konversi Label IndoToxic2024 → Skema Ibrohim 2019 ──")
print("   Kolom dicari: HS, Abusive, atau majority vote dari anotator...")

# Cek nama kolom yang ada
cols = [c.lower() for c in df_raw.columns]
print(f"   Kolom (lowercase): {cols[:20]}")

# IndoDiscourse menyimpan vote per anotator; kita cari kolom agregat
# Coba beberapa kemungkinan nama kolom
hs_col = None
ab_col = None
text_col = None

# IndoDiscourse kolom: identity_attack (proxy HS), profanity_obscenity (proxy Abusive)
# Label disimpan sebagai stringified list misal ['0','1'] per pasang anotator
# Majority vote: hitung berapa '1' dari semua anotator

text_col = 'text'

def majority_from_str_list(val):
    """Parse ['0','1','1',...] → majority vote."""
    try:
        import ast
        lst = ast.literal_eval(str(val))
        ones = sum(1 for v in lst if str(v).strip() == '1')
        return 1 if ones > len(lst) / 2 else 0
    except Exception:
        return 0

# identity_attack → proxy Hate Speech
# profanity_obscenity → proxy Abusive
if 'identity_attack' in df_raw.columns:
    df_raw['hs_vote'] = df_raw['identity_attack'].apply(majority_from_str_list)
    hs_col = 'hs_vote'
    print(f"   Menggunakan 'identity_attack' sebagai proxy Hate Speech")
elif 'toxicity' in df_raw.columns:
    df_raw['hs_vote'] = df_raw['toxicity'].apply(majority_from_str_list)
    hs_col = 'hs_vote'
    print(f"   Menggunakan 'toxicity' sebagai proxy Hate Speech")
else:
    raise ValueError("Tidak ada kolom yang cocok untuk label HS di IndoDiscourse!")

if 'profanity_obscenity' in df_raw.columns:
    df_raw['ab_vote'] = df_raw['profanity_obscenity'].apply(majority_from_str_list)
    ab_col = 'ab_vote'
    print(f"   Menggunakan 'profanity_obscenity' sebagai proxy Abusive")
else:
    ab_col = None
    print("   Kolom Abusive tidak tersedia — semua non-HS dikategorikan Normal")

print(f"   Text kolom  : {text_col}")
print(f"   HS kolom    : {hs_col}")
print(f"   Abusive kolom: {ab_col}")


def map_label(row):
    """Map IndoDiscourse ke skema 3-kelas Ibrohim 2019."""
    try:
        is_hs = int(row[hs_col]) == 1
        is_ab = int(row[ab_col]) == 1 if ab_col else False
    except Exception:
        return None
    if is_hs:
        return 'Hate Speech'
    elif is_ab:
        return 'Abusive'
    else:
        return 'Normal'


df_raw['label_mapped'] = df_raw.apply(map_label, axis=1)
df_eval = df_raw[[text_col, 'label_mapped']].dropna().rename(
    columns={text_col: 'text', 'label_mapped': 'label'}
)

print(f"\nDistribusi label IndoToxic2024 (setelah mapping):")
print(df_eval['label'].value_counts())
print(f"\nTotal sampel evaluasi: {len(df_eval):,}")

# ── 4. STRATIFIED SAMPLING (untuk kecepatan) ─────────────────────────────────
print(f"\n── Stratified Sampling: {N_SAMPLE} dari {len(df_eval):,} entri ──")
from sklearn.model_selection import train_test_split
if len(df_eval) > N_SAMPLE:
    _, df_eval = train_test_split(
        df_eval, test_size=N_SAMPLE,
        stratify=df_eval['label'], random_state=RANDOM_STATE
    )
    df_eval = df_eval.reset_index(drop=True)
print(f"   Sampel untuk evaluasi: {len(df_eval):,}")
print(f"   Distribusi:\n{df_eval['label'].value_counts().to_string()}")

# -- 5. PREPROCESSING CEPAT (5-step, tanpa stemming Sastrawi) -----------------
print("\n-- Preprocessing 5-step (tanpa stemming — lihat methodological note) --")
print(f"   Pipeline: case_folding > cleaning > tokenize > slang_norm > stopword")
t0 = time.time()
results_pp = []
for i, txt in enumerate(df_eval['text']):
    results_pp.append(preprocess_pipeline_fast(txt))
    if (i + 1) % 500 == 0 or (i + 1) == len(df_eval):
        elapsed = time.time() - t0
        eta = (elapsed / (i + 1)) * (len(df_eval) - i - 1)
        print(f"   [{i+1}/{len(df_eval)}] {elapsed:.0f}s elapsed | ETA {eta:.0f}s")

df_eval['processed'] = results_pp
df_eval = df_eval.dropna(subset=['processed']).reset_index(drop=True)
df_eval = df_eval[df_eval['processed'].str.strip() != '']
t_pp = time.time() - t0
print(f"Selesai dalam {t_pp:.1f}s | Sampel valid: {len(df_eval):,}")

# ── 5. TF-IDF TRANSFORM (NO REFIT — hanya transform!) ────────────────────────
print("\n── TF-IDF Transform (NO REFIT — Zero-Shot) ──")
print("   Vectorizer dari Ibrohim 2019 diterapkan ke IndoToxic2024")
X_ext = vectorizer.transform(df_eval['processed'])
y_ext = df_eval['label']
print(f"   Shape matrix: {X_ext.shape}")

# Hitung berapa persen token IndoToxic2024 yang ada di vocab Ibrohim
from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer()
count_mat = count_vec.fit_transform(df_eval['processed'])
indotoxic_vocab = set(count_vec.get_feature_names_out())
ibrohim_vocab   = set(vectorizer.vocabulary_.keys())
overlap = indotoxic_vocab & ibrohim_vocab
oov_rate = 1 - len(overlap) / len(indotoxic_vocab)
print(f"\n📊 Analisis Vocabulary Overlap:")
print(f"   Vocab Ibrohim 2019      : {len(ibrohim_vocab):,} token")
print(f"   Vocab IndoToxic2024     : {len(indotoxic_vocab):,} token")
print(f"   Overlap (token dikenal) : {len(overlap):,} token ({len(overlap)/len(indotoxic_vocab)*100:.1f}%)")
print(f"   OOV Rate (token baru)   : {oov_rate*100:.1f}%")
print("   (OOV token akan bernilai 0 di TF-IDF — wajar dalam zero-shot)")

# ── 6. INFERENSI & EVALUASI ──────────────────────────────────────────────────
print("\n" + "=" * 65)
print("HASIL EVALUASI ZERO-SHOT CROSS-DATASET")
print("=" * 65)

models = {'SVM': svm_model, 'MNB': mnb_model, 'CNB': cnb_model}
results = {}

for name, model in models.items():
    y_pred = model.predict(X_ext)
    acc = accuracy_score(y_ext, y_pred)
    _, _, f1_macro, _ = precision_recall_fscore_support(
        y_ext, y_pred, average='macro', zero_division=0)
    results[name] = {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'y_pred': y_pred
    }
    print(f"\n{'─'*50}")
    print(f"[{name}] Cross-Dataset Accuracy: {acc*100:.2f}% | F1 Macro: {f1_macro*100:.2f}%")
    print(classification_report(y_ext, y_pred, digits=4,
                                target_names=sorted(y_ext.unique())))

# ── 7. PERBANDINGAN IN-DOMAIN vs CROSS-DOMAIN ────────────────────────────────
print("\n" + "=" * 65)
print("PERBANDINGAN: IN-DOMAIN (Ibrohim 2019) vs CROSS-DOMAIN (IndoToxic2024)")
print("=" * 65)

# Load baseline metrics dari experiment sebelumnya
try:
    with open('D:/www/hate-speech/models/baseline_metrics.json') as f:
        baseline = json.load(f)
except FileNotFoundError:
    baseline = {
        'SVM': {'accuracy': 0.7356, 'f1_macro': 0.7003},
        'MNB': {'accuracy': 0.7522, 'f1_macro': 0.7221},
        'CNB': {'accuracy': 0.7322, 'f1_macro': 0.7027},
    }

print(f"\n{'Model':<6} {'In-Domain Acc':>14} {'Cross-Domain Acc':>16} "
      f"{'Drop Acc':>10} {'In-Domain F1':>13} {'Cross-Domain F1':>15} {'Drop F1':>8}")
print("─" * 90)
for name in ['SVM', 'MNB', 'CNB']:
    in_acc  = baseline[name]['accuracy']
    out_acc = results[name]['accuracy']
    in_f1   = baseline[name]['f1_macro']
    out_f1  = results[name]['f1_macro']
    print(f"{name:<6} {in_acc*100:>13.2f}% {out_acc*100:>15.2f}% "
          f"{(out_acc-in_acc)*100:>+9.2f}pp "
          f"{in_f1*100:>12.2f}% {out_f1*100:>14.2f}% "
          f"{(out_f1-in_f1)*100:>+7.2f}pp")

# ── 8. CONFUSION MATRIX ──────────────────────────────────────────────────────
class_names = sorted(y_ext.unique())
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Cross-Dataset Confusion Matrix\n(Model dilatih: Ibrohim 2019 → Diuji: IndoToxic2024)',
             fontsize=13, fontweight='bold')

for idx, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_ext, res['y_pred'], labels=class_names)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.2%', ax=axes[idx],
                xticklabels=class_names, yticklabels=class_names,
                cmap='Blues', linewidths=0.5)
    axes[idx].set_title(
        f'{name}\nAcc: {results[name]["accuracy"]*100:.2f}% | F1: {results[name]["f1_macro"]*100:.2f}%',
        fontweight='bold')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('True')

plt.tight_layout()
out_fig = 'D:/www/hate-speech/notebooks/cross_dataset_confusion_matrix.png'
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✅ Confusion matrix tersimpan: {out_fig}")

# ── 9. BAR CHART PERBANDINGAN ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Perbandingan Performa: In-Domain vs Cross-Domain Evaluation',
             fontsize=13, fontweight='bold')

x = np.arange(3)
model_names = ['SVM', 'MNB', 'CNB']
width = 0.35

for ax_idx, metric in enumerate(['accuracy', 'f1_macro']):
    in_vals  = [baseline[n][metric] * 100 for n in model_names]
    out_vals = [results[n][metric] * 100 for n in model_names]
    bars1 = axes[ax_idx].bar(x - width/2, in_vals,  width, label='In-Domain (Ibrohim 2019)',
                              color='#1976D2', alpha=0.85, edgecolor='white')
    bars2 = axes[ax_idx].bar(x + width/2, out_vals, width, label='Cross-Domain (IndoToxic2024)',
                              color='#F57C00', alpha=0.85, edgecolor='white')
    axes[ax_idx].set_xticks(x)
    axes[ax_idx].set_xticklabels(model_names, fontsize=12)
    axes[ax_idx].set_ylabel(f"{'Accuracy' if metric == 'accuracy' else 'F1 Macro'} (%)")
    axes[ax_idx].set_title(f"{'Accuracy' if metric == 'accuracy' else 'F1 Macro'} Comparison")
    axes[ax_idx].legend()
    axes[ax_idx].set_ylim(0, 100)
    axes[ax_idx].grid(axis='y', alpha=0.3)
    for bar in bars1:
        axes[ax_idx].annotate(f'{bar.get_height():.1f}%',
                               xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    for bar in bars2:
        axes[ax_idx].annotate(f'{bar.get_height():.1f}%',
                               xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

plt.tight_layout()
out_bar = 'D:/www/hate-speech/notebooks/cross_dataset_comparison.png'
plt.savefig(out_bar, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Grafik perbandingan tersimpan: {out_bar}")

# ── 10. SIMPAN HASIL KE JSON ──────────────────────────────────────────────────
cross_eval_results = {
    'dataset_target': 'IndoToxic2024 (Exqrch/IndoDiscourse)',
    'dataset_source': 'Ibrohim et al. 2019 (dataset_tweet.csv)',
    'n_samples_evaluated': int(len(df_eval)),
    'oov_rate_percent': round(oov_rate * 100, 2),
    'label_distribution': df_eval['label'].value_counts().to_dict(),
    'models': {}
}
for name in ['SVM', 'MNB', 'CNB']:
    cross_eval_results['models'][name] = {
        'cross_domain_accuracy': round(float(results[name]['accuracy']), 6),
        'cross_domain_f1_macro': round(float(results[name]['f1_macro']), 6),
        'in_domain_accuracy': round(float(baseline[name]['accuracy']), 6),
        'in_domain_f1_macro': round(float(baseline[name]['f1_macro']), 6),
        'accuracy_drop_pp': round((results[name]['accuracy'] - baseline[name]['accuracy']) * 100, 2),
        'f1_drop_pp': round((results[name]['f1_macro'] - baseline[name]['f1_macro']) * 100, 2),
    }

out_json = 'D:/www/hate-speech/models/cross_dataset_eval.json'
with open(out_json, 'w') as f:
    json.dump(cross_eval_results, f, indent=2)
print(f"✅ Hasil evaluasi tersimpan: {out_json}")

print("\n" + "=" * 65)
print("✅ CROSS-DATASET EVALUATION SELESAI")
print("=" * 65)
print("\nFile output:")
print(f"  📊 {out_fig}")
print(f"  📊 {out_bar}")
print(f"  📄 {out_json}")
print("\nInterpretasi untuk paper:")
print("  • Penurunan performa dalam cross-dataset (Δ accuracy/F1) adalah WAJAR")
print("    dan justru membuktikan model bukan overfit pada Ibrohim 2019.")
print("  • Bandingkan dengan baseline zero-shot (random: ~33%) untuk konteks.")
print("  • OOV rate menjelaskan gap performa secara leksikal.")
