# ============================================================================
# notebooks/03_modeling.py
# TAHAP 3-6: TF-IDF + Training + Evaluasi + Simpan Model
# ============================================================================
# LIBRARIES: scikit-learn, numpy, pandas, matplotlib, seaborn, 
#            statsmodels, joblib
# FILES: data/processed/df_clean.csv → models/*.pkl + visualisasi
# ============================================================================
# CATATAN: Semua angka accuracy adalah HASIL EKSPERIMEN, bukan target.
#          Jalankan script ini dan lihat hasilnya secara empiris.
# ============================================================================

import sys
sys.path.insert(0, 'D:/www/hate-speech')

import pandas as pd
import numpy as np
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import json
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_curve, auc,
                             precision_recall_fscore_support)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.contingency_tables import mcnemar
from imblearn.over_sampling import SMOTE
from preprocessing import (preprocess_pipeline, step1_case_folding,
                           step2_cleaning, step3_tokenization,
                           step4_slang_normalization, step5_stopword_removal,
                           step6_stemming)

# ============================================================================
# TAHAP 3: LOAD DATA + TF-IDF + SPLIT
# ============================================================================
print("=" * 60)
print("TAHAP 3: TF-IDF VECTORIZATION + TRAIN-VAL-TEST SPLIT")
print("=" * 60)

# ── Load Data ────────────────────────────────────────────────────────────────
df_clean = pd.read_csv('D:/www/hate-speech/data/processed/df_clean.csv',
                        encoding='utf-8')
print(f"\nDataset dimuat: {len(df_clean)} tweet")
print(df_clean['label'].value_counts())

X = df_clean['processed']
y = df_clean['label']

# ── Shared Train-Val-Test Split (CRITICAL — SEBELUM TF-IDF!) ────────────────
# Split ini disimpan agar dapat dipakai ulang oleh IndoBERT (P2.3: identical test set).
idx_all = np.arange(len(df_clean))
idx_train_val, idx_test = train_test_split(
    idx_all,
    test_size=0.2,
    random_state=42,
    stratify=y
)

y_train_val = y.iloc[idx_train_val]
idx_train, idx_val = train_test_split(
    idx_train_val,
    test_size=0.1,
    random_state=42,
    stratify=y_train_val
)

X_train = X.iloc[idx_train]
X_val = X.iloc[idx_val]
X_test = X.iloc[idx_test]

y_train = y.iloc[idx_train]
y_val = y.iloc[idx_val]
y_test = y.iloc[idx_test]

split_artifact_path = 'D:/www/hate-speech/models/shared_split_indices.npz'
np.savez(
    split_artifact_path,
    idx_train=idx_train,
    idx_val=idx_val,
    idx_test=idx_test
)
print(f"\nShared split tersimpan ke: {split_artifact_path}")

print(f"\nTraining set : {len(X_train)} tweet")
print(f"Validation set: {len(X_val)} tweet")
print(f"Test set     : {len(X_test)} tweet")
print(f"\nDistribusi training:\n{y_train.value_counts()}")
print(f"\nDistribusi validation:\n{y_val.value_counts()}")
print(f"\nDistribusi test:\n{y_test.value_counts()}")

# ── TF-IDF Vectorizer ───────────────────────────────────────────────────────
# PENTING: fit HANYA pada X_train — DATA LEAKAGE PREVENTION!
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    norm='l2',
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)  # fit + transform training
X_test_tfidf = vectorizer.transform(X_test)          # transform saja test!

print(f"\nJumlah fitur TF-IDF: {len(vectorizer.vocabulary_)}")
print(f"Shape X_train_tfidf (sebelum SMOTE) : {X_train_tfidf.shape}")
print(f"Shape X_test_tfidf  : {X_test_tfidf.shape}")

# ── SMOTE (Synthetic Minority Over-sampling Technique) ──────────────────────
print("\nMelakukan resampling SMOTE pada data training...")
smote = SMOTE(random_state=42)
X_train_tfidf, y_train = smote.fit_resample(X_train_tfidf, y_train)

print(f"Distribusi y_train setelah SMOTE:\n{y_train.value_counts()}")
print(f"Shape X_train_tfidf (setelah SMOTE): {X_train_tfidf.shape}")

# Simpan vectorizer
joblib.dump(vectorizer, 'D:/www/hate-speech/models/tfidf_vectorizer.pkl')
print("\nVectorizer tersimpan!")

# ============================================================================
# TAHAP 4: TRAINING 3 MODEL + CROSS VALIDATION
# ============================================================================
print("\n" + "=" * 60)
print("TAHAP 4: TRAINING & CROSS VALIDATION (10-Fold Stratified)")
print("=" * 60)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

models = {
    'MNB': MultinomialNB(alpha=1.0),
    'CNB': ComplementNB(alpha=1.0),
    'SVM': LinearSVC(class_weight='balanced', C=1.0, dual='auto', max_iter=10000, random_state=42)
}

cv_results = {}

for name, model in models.items():
    print(f"\n[{name}] Memulai cross validation...")
    t_start = time.time()

    scores = cross_val_score(
        model,
        X_train_tfidf,
        y_train,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1
    )

    t_elapsed = time.time() - t_start
    cv_results[name] = scores

    print(f"  Accuracy per fold : {[f'{s*100:.2f}%' for s in scores]}")
    print(f"  Mean ± Std        : {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")
    print(f"  Waktu CV          : {t_elapsed:.1f} detik")

# CV Summary
print("\n" + "-" * 60)
print("RINGKASAN CROSS VALIDATION")
print("-" * 60)
print(f"{'Model':<6} {'Mean Acc':<12} {'Std':<10} {'Min':<10} {'Max':<10}")
print("-" * 48)
for name, scores in cv_results.items():
    print(f"{name:<6} {scores.mean()*100:.2f}%      "
          f"±{scores.std()*100:.2f}%    "
          f"{scores.min()*100:.2f}%    "
          f"{scores.max()*100:.2f}%")

# ── Training Final + Simpan Model ────────────────────────────────────────────
print("\nMelatih model final pada seluruh training data...")
trained_models = {}

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    trained_models[name] = model
    save_path = f'D:/www/hate-speech/models/{name.lower()}_model.pkl'
    joblib.dump(model, save_path)
    print(f"  [{name}] Disimpan ke {save_path}")

# ============================================================================
# TAHAP 5: EVALUASI HOLDOUT TEST SET
# ============================================================================
print("\n" + "=" * 60)
print("TAHAP 5: EVALUASI HOLDOUT TEST SET")
print("=" * 60)

# ── Holdout Test Evaluation ──────────────────────────────────────────────────
predictions = {}
holdout_accuracies = {}

for name, model in trained_models.items():
    y_pred = model.predict(X_test_tfidf)
    predictions[name] = y_pred
    acc = accuracy_score(y_test, y_pred)
    holdout_accuracies[name] = acc
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc*100:.4f}%")
    print(classification_report(y_test, y_pred, digits=4))

# Summary comparison
print("\n" + "-" * 60)
print("PERBANDINGAN HOLDOUT ACCURACY")
print("-" * 60)
for name in ['SVM', 'CNB', 'MNB']:
    acc = holdout_accuracies[name]
    cv_mean = cv_results[name].mean()
    diff = (acc - cv_mean) * 100
    print(f"  {name}: Holdout={acc*100:.2f}% | CV Mean={cv_mean*100:.2f}% | "
          f"Diff={diff:+.2f}pp")

svm_pred_artifact_path = 'D:/www/hate-speech/models/svm_test_predictions.npz'
np.savez(
    svm_pred_artifact_path,
    y_true=y_test.values,
    y_pred_svm=predictions['SVM']
)
print(f"\nPrediksi SVM pada test set tersimpan ke: {svm_pred_artifact_path}")

# ── Export Metrik ke JSON (untuk digunakan oleh 04_indobert_colab.py) ─────────
baseline_metrics = {}
for name, y_pred in predictions.items():
    acc = holdout_accuracies[name]
    _, _, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )
    baseline_metrics[name] = {
        'accuracy': round(float(acc), 6),
        'f1_macro': round(float(f1), 6)
    }

# ── Pengukuran Latensi Inference (Model Saja, Tanpa Overhead Web) ────────────
# CATATAN PENTING UNTUK PAPER:
# Angka "87ms" di paper v9 Seksi 4.5 adalah latensi sistem web FastAPI+SVM+cache
# (n=500 HTTP request dengan model di-cache in-memory) — ini BELUM DIUKUR secara
# empiris dalam skrip ini. Blok di bawah mengukur latensi inference model murni
# (tanpa overhead HTTP, praproses, atau TF-IDF vectorization).
print("\n" + "-" * 60)
print("LATENSI INFERENCE — Model SVM Murni (Tanpa HTTP/TF-IDF)")
print("-" * 60)
print("⚠️  DISCLAIMER: Angka ini bukan angka 87ms di paper.")
print("    87ms di paper adalah latensi sistem web FastAPI end-to-end.")
print("    Angka di bawah: inference model SVM setelah TF-IDF sudah dilakukan.")

_N_LATENCY = 500
_X_bench = X_test_tfidf[:1]  # 1 sampel per request (simulasi single-tweet API)

_svm_latency_runs = []
for _ in range(_N_LATENCY):
    _t0 = time.perf_counter()
    trained_models['SVM'].predict(_X_bench)
    _t1 = time.perf_counter()
    _svm_latency_runs.append((_t1 - _t0) * 1000)  # ms

_lat_mean = np.mean(_svm_latency_runs)
_lat_median = np.median(_svm_latency_runs)
_lat_p95 = np.percentile(_svm_latency_runs, 95)
_lat_std = np.std(_svm_latency_runs)

print(f"\n  N runs          : {_N_LATENCY}")
print(f"  Mean latensi    : {_lat_mean:.3f} ms")
print(f"  Median latensi  : {_lat_median:.3f} ms")
print(f"  Std             : {_lat_std:.3f} ms")
print(f"  P95 latensi     : {_lat_p95:.3f} ms")
print(f"\n  Interpretasi untuk paper:")
print(f"  Inference model SVM murni ~{_lat_mean:.2f}ms per sampel.")
print(f"  Overhead praproses + TF-IDF + HTTP menambah latensi hingga level 87ms")
print(f"  dalam deployment FastAPI penuh.")

# Simpan latensi ke metrics JSON
for name in ['SVM', 'MNB', 'CNB']:
    _bench = X_test_tfidf[:1]
    _runs = []
    for _ in range(_N_LATENCY):
        _t0 = time.perf_counter()
        trained_models[name].predict(_bench)
        _runs.append((time.perf_counter() - _t0) * 1000)
    baseline_metrics[name]['latency_inference_ms_mean']   = round(np.mean(_runs), 4)
    baseline_metrics[name]['latency_inference_ms_median'] = round(np.median(_runs), 4)
    baseline_metrics[name]['latency_inference_ms_p95']    = round(np.percentile(_runs, 95), 4)
    baseline_metrics[name]['latency_n_runs']              = _N_LATENCY
    baseline_metrics[name]['latency_note'] = (
        'Inference model murni (post-TF-IDF), 1 sampel per run, '
        'tanpa overhead HTTP/praproses. '
        'Angka 87ms di paper v9 Seksi 4.5 adalah latensi sistem FastAPI end-to-end.'
    )
    print(f"  [{name}] Mean={np.mean(_runs):.3f}ms | P95={np.percentile(_runs, 95):.3f}ms")

metrics_path = 'D:/www/hate-speech/models/baseline_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(baseline_metrics, f, indent=2)
print(f"\n✅ Metrik baseline + latensi disimpan ke: {metrics_path}")
print(f"   Model metrics: {list(baseline_metrics.keys())}")


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
class_names = sorted(y_test.unique())

for idx, (name, y_pred) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx],
                xticklabels=class_names, yticklabels=class_names,
                cmap='Blues', linewidths=0.5)
    acc = holdout_accuracies[name]
    axes[idx].set_title(f'{name} — Acc: {acc*100:.2f}%', fontweight='bold')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('True')

plt.tight_layout()
plt.savefig('D:/www/hate-speech/notebooks/confusion_matrix.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("\nConfusion matrix tersimpan ke notebooks/confusion_matrix.png")

# ── McNemar Test dengan Koreksi Bonferroni ───────────────────────────────────
print("\n" + "-" * 60)
print("McNEMAR TEST (Bonferroni α = 0.05/3 = 0.0167)")
print("-" * 60)

alpha_bonferroni = 0.05 / 3  # 0.0167

pairs = [
    ('SVM', 'MNB'),
    ('SVM', 'CNB'),
    ('CNB', 'MNB')
]

mcnemar_results = []

for name_a, name_b in pairs:
    pred_a = predictions[name_a]
    pred_b = predictions[name_b]
    
    correct_a = (pred_a == y_test.values)
    correct_b = (pred_b == y_test.values)
    
    # Contingency table
    n00 = np.sum(correct_a & correct_b)       # both correct
    n01 = np.sum(correct_a & ~correct_b)      # A correct, B wrong
    n10 = np.sum(~correct_a & correct_b)      # A wrong, B correct
    n11 = np.sum(~correct_a & ~correct_b)     # both wrong
    
    table = np.array([[n00, n01], [n10, n11]])
    
    # McNemar test with continuity correction
    result = mcnemar(table, exact=False, correction=True)
    
    significant = "✅ SIGNIFIKAN" if result.pvalue < alpha_bonferroni else "❌ TIDAK signifikan"
    
    mcnemar_results.append({
        'Pair': f'{name_a} vs {name_b}',
        'b (only A correct)': n01,
        'c (only B correct)': n10,
        'Chi-squared': round(result.statistic, 4),
        'p-value': result.pvalue,
        'Significant': result.pvalue < alpha_bonferroni
    })
    
    print(f"\n  {name_a} vs {name_b}:")
    print(f"    b={n01} (hanya {name_a} benar), c={n10} (hanya {name_b} benar)")
    print(f"    McNemar χ² = {result.statistic:.4f}")
    print(f"    p-value    = {result.pvalue:.6f}")
    print(f"    Status     : {significant} (α = {alpha_bonferroni:.4f})")

# McNemar summary table
mcnemar_df = pd.DataFrame(mcnemar_results)
print("\n" + "-" * 60)
print("RINGKASAN McNEMAR TEST")
print("-" * 60)
print(mcnemar_df.to_string(index=False))

# ── Ablation Study (6 config × 3 models = 18 experiments) ───────────────────
print("\n" + "-" * 60)
print("ABLATION STUDY (6 konfigurasi × 3 model = 18 eksperimen)")
print("-" * 60)

# 6 konfigurasi preprocessing
configs = {
    'Full (6 step)': [step1_case_folding, step2_cleaning, 
                      step3_tokenization, step4_slang_normalization,
                      step5_stopword_removal, step6_stemming],
    'No Stemming (5 step)': [step1_case_folding, step2_cleaning, 
                              step3_tokenization, step4_slang_normalization,
                              step5_stopword_removal],
    'No Stopword (5 step)': [step1_case_folding, step2_cleaning, 
                              step3_tokenization, step4_slang_normalization,
                              step6_stemming],
    'No Slang Norm (5 step)': [step1_case_folding, step2_cleaning, 
                                step3_tokenization, step5_stopword_removal,
                                step6_stemming],
    'Clean+Tokenize (3 step)': [step1_case_folding, step2_cleaning, 
                                  step3_tokenization],
    'Lower+Tokenize (2 step)': [step1_case_folding, step3_tokenization],
}


def apply_config(text, steps):
    """Apply specific pipeline configuration."""
    if not isinstance(text, str) or not text.strip():
        return None
    result = text
    for step in steps:
        if step == step3_tokenization:
            result = step(result)  # returns list
        elif isinstance(result, list):
            if step in [step4_slang_normalization, step5_stopword_removal, step6_stemming]:
                result = step(result)
            else:
                result = step(' '.join(result))
        else:
            result = step(result)
    
    if isinstance(result, list):
        if len(result) < 3:
            return None
        return ' '.join(result)
    else:
        tokens = result.split()
        if len(tokens) < 3:
            return None
        return result


# Load raw data for ablation
df_raw = pd.read_csv('D:/www/hate-speech/data/raw/dataset_tweet.csv', encoding='latin-1')


def convert_label(row):
    if row['HS'] == 1:
        return 'Hate Speech'
    elif row['Abusive'] == 1:
        return 'Abusive'
    else:
        return 'Normal'


df_raw['label'] = df_raw.apply(convert_label, axis=1)

ablation_results = []

for config_name, steps in configs.items():
    print(f"\n  Konfigurasi: {config_name}")
    t_start = time.time()
    
    # Apply preprocessing config
    df_temp = df_raw.copy()
    df_temp['processed'] = df_temp['Tweet'].apply(lambda x: apply_config(x, steps))
    df_temp = df_temp.dropna(subset=['processed']).reset_index(drop=True)
    
    n_samples = len(df_temp)
    if n_samples < 100:
        print(f"    ⚠ Terlalu sedikit data ({n_samples}) — skip")
        continue
    
    X_abl = df_temp['processed']
    y_abl = df_temp['label']
    
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_abl, y_abl, test_size=0.2, random_state=42, stratify=y_abl
    )
    
    vec = TfidfVectorizer(ngram_range=(1,1), min_df=3, max_df=0.95,
                          norm='l2', sublinear_tf=True)
    X_tr_tfidf = vec.fit_transform(X_tr)
    X_te_tfidf = vec.transform(X_te)
    
    ablation_models = {
        'MNB': MultinomialNB(alpha=1.0),
        'CNB': ComplementNB(alpha=1.0),
        'SVM': LinearSVC(C=1.0, dual='auto', max_iter=10000, random_state=42)
    }
    
    for model_name, model in ablation_models.items():
        model.fit(X_tr_tfidf, y_tr)
        y_pred = model.predict(X_te_tfidf)
        acc = accuracy_score(y_te, y_pred)
        ablation_results.append({
            'Config': config_name,
            'Model': model_name,
            'Accuracy (%)': round(acc * 100, 2),
            'N_samples': n_samples,
            'N_features': X_tr_tfidf.shape[1]
        })
        print(f"    {model_name}: {acc*100:.2f}% "
              f"({n_samples} samples, {X_tr_tfidf.shape[1]} features)")
    
    t_elapsed = time.time() - t_start
    print(f"    Waktu: {t_elapsed:.1f} detik")

# Summary table
ablation_df = pd.DataFrame(ablation_results)
print("\n" + "-" * 60)
print("RINGKASAN ABLATION STUDY")
print("-" * 60)

if len(ablation_df) > 0:
    pivot = ablation_df.pivot_table(index='Config', columns='Model', 
                                     values='Accuracy (%)', aggfunc='first')
    pivot = pivot[['SVM', 'CNB', 'MNB']]  # reorder columns
    print(pivot.to_string())
    
    # Save ablation results
    ablation_df.to_csv('D:/www/hate-speech/notebooks/ablation_results.csv',
                       index=False, encoding='utf-8')
    print("\nAblation results tersimpan ke notebooks/ablation_results.csv")

    # Visualize ablation
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind='bar', ax=ax, rot=25, edgecolor='white')
    ax.set_title('Ablation Study — Accuracy per Konfigurasi Preprocessing', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Konfigurasi Preprocessing')
    ax.legend(title='Model')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('D:/www/hate-speech/notebooks/ablation_study.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Visualisasi ablation tersimpan ke notebooks/ablation_study.png")

# ── ROC Curve per Kelas ──────────────────────────────────────────────────────
print("\n" + "-" * 60)
print("ROC CURVES")
print("-" * 60)

class_names_roc = sorted(y_test.unique())
y_test_bin = label_binarize(y_test, classes=class_names_roc)
n_classes = len(class_names_roc)

# Get probability/decision scores for each model
model_scores = {}

# MNB — has predict_proba
model_scores['MNB'] = trained_models['MNB'].predict_proba(X_test_tfidf)

# CNB — has predict_proba
model_scores['CNB'] = trained_models['CNB'].predict_proba(X_test_tfidf)

# SVM — use CalibratedClassifierCV for proper probabilities
print("  Calibrating SVM for probability estimation (5-fold)...")
svm_calibrated = CalibratedClassifierCV(
    LinearSVC(C=1.0, dual='auto', max_iter=10000, random_state=42), 
    cv=5, method='sigmoid'
)
svm_calibrated.fit(X_train_tfidf, y_train)
model_scores['SVM'] = svm_calibrated.predict_proba(X_test_tfidf)

# Determine class order for each model
model_class_mapping = {
    'MNB': list(trained_models['MNB'].classes_),
    'CNB': list(trained_models['CNB'].classes_),
    'SVM': list(svm_calibrated.classes_),
}

# Plot ROC curves: 3 panels (one per class), all 3 models in each
fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5))
model_colors = {'MNB': '#2196F3', 'CNB': '#FF9800', 'SVM': '#4CAF50'}

roc_auc_data = {}

for i, cls in enumerate(class_names_roc):
    for mname, scores in model_scores.items():
        cls_list = model_class_mapping[mname]
        cls_idx = cls_list.index(cls)
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], scores[:, cls_idx])
        roc_auc_val = auc(fpr, tpr)
        
        roc_auc_data[f'{mname}_{cls}'] = roc_auc_val
        
        axes[i].plot(fpr, tpr, color=model_colors[mname], lw=2,
                     label=f'{mname} (AUC = {roc_auc_val:.4f})')
    
    axes[i].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    axes[i].set_xlim([0.0, 1.0])
    axes[i].set_ylim([0.0, 1.05])
    axes[i].set_xlabel('False Positive Rate')
    axes[i].set_ylabel('True Positive Rate')
    axes[i].set_title(f'ROC — {cls}', fontweight='bold')
    axes[i].legend(loc='lower right', fontsize=9)
    axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('D:/www/hate-speech/notebooks/roc_curves.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("ROC curves tersimpan ke notebooks/roc_curves.png")

# Print AUC summary
print("\nAUC Summary:")
print(f"{'Model':<6} ", end='')
for cls in class_names_roc:
    print(f"{cls:<14}", end='')
print()
print("-" * (6 + 14 * n_classes))
for mname in ['SVM', 'CNB', 'MNB']:
    print(f"{mname:<6} ", end='')
    for cls in class_names_roc:
        val = roc_auc_data[f'{mname}_{cls}']
        print(f"{val:.4f}        ", end='')
    print()

# ── Wilson CI Error Analysis ─────────────────────────────────────────────────
print("\n" + "-" * 60)
print("WILSON CONFIDENCE INTERVAL — ERROR ANALYSIS")
print("-" * 60)


def wilson_ci(n_success, n_total, z=1.96):
    """Calculate Wilson score confidence interval."""
    if n_total == 0:
        return (0.0, 0.0)
    p_hat = n_success / n_total
    denominator = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denominator
    margin = (z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_total)) 
              / n_total)) / denominator
    return (max(0, center - margin), min(1, center + margin))


error_analysis_data = []

for name in ['SVM', 'CNB', 'MNB']:
    y_pred = predictions[name]
    print(f"\n  [{name}] Error Analysis:")
    
    for true_label in class_names:
        for pred_label in class_names:
            if true_label == pred_label:
                continue
            mask = (y_test.values == true_label) & (y_pred == pred_label)
            n_errors = int(mask.sum())
            n_true = int((y_test.values == true_label).sum())
            if n_errors > 0:
                error_rate = n_errors / n_true
                ci_low, ci_high = wilson_ci(n_errors, n_true)
                error_analysis_data.append({
                    'Model': name,
                    'True': true_label,
                    'Predicted': pred_label,
                    'Errors': n_errors,
                    'Total': n_true,
                    'Error Rate (%)': round(error_rate * 100, 2),
                    'CI Low (%)': round(ci_low * 100, 2),
                    'CI High (%)': round(ci_high * 100, 2),
                })
                print(f"    {true_label} → {pred_label}: "
                      f"{n_errors}/{n_true} ({error_rate*100:.2f}%) "
                      f"Wilson 95% CI: [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")

error_df = pd.DataFrame(error_analysis_data)
if len(error_df) > 0:
    error_df.to_csv('D:/www/hate-speech/notebooks/error_analysis.csv',
                    index=False, encoding='utf-8')
    print("\nError analysis tersimpan ke notebooks/error_analysis.csv")

# ============================================================================
# TAHAP 6: SIMPAN MODEL & ARTIFACTS
# ============================================================================
print("\n" + "=" * 60)
print("TAHAP 6: SIMPAN MODEL & ARTIFACTS")
print("=" * 60)

# Top Terms SVM
feature_names = vectorizer.get_feature_names_out()
svm_model = trained_models['SVM']
classes = svm_model.classes_
top_terms = {}

print("\nTop 5 terms per kelas (SVM koefisien tertinggi):")
for i, cls in enumerate(classes):
    coef = svm_model.coef_[i]
    top_indices = np.argsort(coef)[-5:][::-1]
    top_terms[cls] = [feature_names[j] for j in top_indices]
    print(f"  [{cls}]: {top_terms[cls]}")

joblib.dump(top_terms, 'D:/www/hate-speech/models/top_terms.pkl')
print("\ntop_terms.pkl tersimpan!")

# ── Auto-Generate Full Report ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("GENERATE FULL TRAINING REPORT")
print("=" * 60)

report_content = f"""# Laporan Otomatis Hasil Pelatihan Model
*Waktu eksekusi: {time.strftime('%Y-%m-%d %H:%M:%S')}*

## 1. Informasi Dataset
- **Total Dataset Bersih**: {len(df_clean)} tweet
- **Fitur TF-IDF**: {len(vectorizer.vocabulary_)} kosa kata
- **Training Set**: {X_train_tfidf.shape[0]} tweet
- **Test Set**: {X_test_tfidf.shape[0]} tweet

## 2. Ringkasan Cross Validation (10-Fold)
| Model | Mean Acc | Std | Min | Max |
|---|---|---|---|---|
"""
for name, scores in cv_results.items():
    report_content += f"| **{name}** | {scores.mean()*100:.2f}% | ±{scores.std()*100:.2f}% | {scores.min()*100:.2f}% | {scores.max()*100:.2f}% |\n"

report_content += """
## 3. Evaluasi Holdout Test Set
"""
for name in ['SVM', 'CNB', 'MNB']:
    acc = holdout_accuracies[name]
    cv_mean = cv_results[name].mean()
    diff = (acc - cv_mean) * 100
    report_content += f"\n### {name}\n- **Accuracy**: {acc*100:.4f}%\n- **Selisih Holdout vs CV**: {diff:+.2f}pp\n\n"
    report_content += "```text\n" + classification_report(y_test, predictions[name], digits=4) + "```\n"

report_content += """
## 4. McNemar Test (Signifikansi Statistik)
| Pair | P-Value | Signifikan (α = 0.0167) |
|---|---|---|
"""
for row in mcnemar_results:
    report_content += f"| {row['Pair']} | {row['p-value']:.6f} | {'✅ Ya' if row['Significant'] else '❌ Tidak'} |\n"

if len(ablation_df) > 0:
    report_content += """
## 5. Ringkasan Ablation Study
```text
"""
    report_content += pivot.to_string() + "\n```\n"

report_content += """
## 6. Analisis Top 5 Fitur Berpengaruh (SVM)
"""
for cls in classes:
    report_content += f"- **{cls}**: {', '.join(top_terms[cls])}\n"

report_path = 'D:/www/hate-speech/pipeline/auto_training_report.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_content)

print(f"Laporan lengkap otomatis tersimpan di {report_path}")

# ── Final Verification ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ PIPELINE ML SELESAI — VERIFIKASI FINAL")
print("=" * 60)

import os
models_dir = 'D:/www/hate-speech/models'
expected_files = ['svm_model.pkl', 'mnb_model.pkl', 'cnb_model.pkl',
                  'tfidf_vectorizer.pkl', 'top_terms.pkl']

print("\nModel files:")
for f in expected_files:
    path = os.path.join(models_dir, f)
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    status = "✅" if exists else "❌"
    print(f"  {status} {f} ({size/1024:.1f} KB)")

print(f"\n{'='*40}")
print(f"  Dataset bersih    : {len(df_clean)} tweet")
print(f"  Fitur TF-IDF      : {len(vectorizer.vocabulary_)}")
print(f"  Training set      : {X_train_tfidf.shape[0]}")
print(f"  Test set          : {X_test_tfidf.shape[0]}")
print(f"{'='*40}")

print(f"\nHasil Eksperimen — Holdout Accuracy:")
for name in ['SVM', 'CNB', 'MNB']:
    acc = holdout_accuracies[name]
    print(f"  {name}: {acc*100:.4f}%")

print(f"\nHasil Eksperimen — CV Mean Accuracy:")
for name in ['SVM', 'CNB', 'MNB']:
    cv_mean = cv_results[name].mean()
    cv_std = cv_results[name].std()
    print(f"  {name}: {cv_mean*100:.4f}% ± {cv_std*100:.4f}%")

print("\nVisualisasi tersimpan:")
print("  📊 notebooks/confusion_matrix.png")
print("  📊 notebooks/roc_curves.png")
print("  📊 notebooks/ablation_study.png")
print("  📄 notebooks/ablation_results.csv")
print("  📄 notebooks/error_analysis.csv")

print("\n" + "=" * 60)
print("NEXT: Jalankan backend")
print("  cd backend")
print("  uvicorn main:app --reload --port 8000")
print("=" * 60)
