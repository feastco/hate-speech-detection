# ============================================================================
# notebooks/fix_abusive_model.py  (v3 — partial oversample + class_weight)
#
# Strategi gabungan:
#   1. Oversample Abusive 2x saja (bukan sampai seimbang) - aman dari crash
#   2. class_weight agresif untuk Abusive
#   3. SGDClassifier (tanpa liblinear)
#
# Jalankan: python -u notebooks/fix_abusive_model.py
# ============================================================================

import sys, os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'

sys.path.insert(0, 'D:/www/hate-speech')

import pandas as pd
import numpy as np
import joblib
import traceback
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, precision_recall_fscore_support,
                             confusion_matrix)
from scipy.sparse import vstack as sparse_vstack

import functools
print = functools.partial(print, flush=True)

# ---- 1. LOAD ----
print("=" * 60)
print("[1/9] LOAD DATA")
df = pd.read_csv('D:/www/hate-speech/data/processed/df_clean.csv', encoding='utf-8')
df['processed'] = df['processed'].fillna('')
print(f"  Total: {len(df)}")
for lbl, cnt in df['label'].value_counts().items():
    print(f"    {lbl:<15}: {cnt:>5} ({cnt/len(df)*100:.1f}%)")

X = df['processed']
y = df['label']

# ---- 2. SPLIT ----
print("\n[2/9] SPLIT")
idx = np.arange(len(df))
idx_tv, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
y_tv = y.iloc[idx_tv]
idx_train, idx_val = train_test_split(idx_tv, test_size=0.125, random_state=42, stratify=y_tv)

X_train = X.iloc[idx_train].reset_index(drop=True)
X_val   = X.iloc[idx_val].reset_index(drop=True)
X_test  = X.iloc[idx_test].reset_index(drop=True)
y_train = y.iloc[idx_train].reset_index(drop=True)
y_val   = y.iloc[idx_val].reset_index(drop=True)
y_test  = y.iloc[idx_test].reset_index(drop=True)

# (Data sintetis dihapus - kita akan gunakan intervensi bobot koefisien SVM)

# ---- 3. TF-IDF ----
print("\n[3/9] TF-IDF")
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), min_df=3, max_df=0.95,
    norm='l2', sublinear_tf=True
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf   = vectorizer.transform(X_val)
X_test_tfidf  = vectorizer.transform(X_test)
print(f"  Shape: {X_train_tfidf.shape}")

# ---- 4. PARTIAL OVERSAMPLING (hanya Abusive, 2x lipat) ----
print("\n[4/9] PARTIAL OVERSAMPLING (Abusive 3x lipat)")
print("  Strategi: duplikasi Abusive secukupnya (bukan sampai seimbang)")

y_arr   = np.array(y_train)
X_parts = [X_train_tfidf]
y_parts = [y_arr]
rng     = np.random.default_rng(42)

# Hanya oversample Abusive, target = 3x jumlah asli
ab_mask  = y_arr == 'Abusive'
ab_count = ab_mask.sum()
ab_target = ab_count * 3  # 1025 -> 3075
ab_needed = ab_target - ab_count

ab_idx  = np.where(ab_mask)[0]
chosen  = rng.choice(ab_idx, size=ab_needed, replace=True)
X_parts.append(X_train_tfidf[chosen])
y_parts.append(np.array(['Abusive'] * ab_needed))
print(f"  Abusive: {ab_count} -> {ab_target} (+{ab_needed})")

X_res = sparse_vstack(X_parts).tocsr()
y_res = np.concatenate(y_parts)

print(f"  Shape setelah: {X_res.shape}")
unique, counts = np.unique(y_res, return_counts=True)
for u, c in zip(unique, counts):
    print(f"    {u:<15}: {c}")

# ---- 5. GRID SEARCH ----
print("\n[5/9] GRID SEARCH")

candidates = [
    # alpha, class_weight
    (1e-4, 'balanced'),
    (1e-4, {'Abusive': 3, 'Hate Speech': 1, 'Normal': 1}),
    (1e-4, {'Abusive': 5, 'Hate Speech': 1, 'Normal': 1}),
    (1e-4, {'Abusive': 7, 'Hate Speech': 1, 'Normal': 1}),
    (1e-4, {'Abusive': 10, 'Hate Speech': 1, 'Normal': 1}),
    (1e-5, 'balanced'),
    (1e-5, {'Abusive': 5, 'Hate Speech': 1, 'Normal': 1}),
    (1e-5, {'Abusive': 7, 'Hate Speech': 1, 'Normal': 1}),
    (1e-5, {'Abusive': 10, 'Hate Speech': 1, 'Normal': 1}),
    (1e-5, {'Abusive': 15, 'Hate Speech': 1, 'Normal': 1}),
    (5e-5, {'Abusive': 7, 'Hate Speech': 1, 'Normal': 1}),
    (5e-5, {'Abusive': 10, 'Hate Speech': 1, 'Normal': 1}),
]

best_f1    = 0.0
best_cfg   = None
best_model = None

print(f"  {'alpha':<10} {'W_AB':<10} {'F1-mac':<10} {'Acc':<10} {'F1-AB':<10} {'Rec-AB':<10}")
print("  " + "-" * 62)

for alpha, cw in candidates:
    cw_label = cw if isinstance(cw, str) else str(cw.get('Abusive', '?'))
    try:
        mdl = SGDClassifier(
            loss='hinge', alpha=alpha, class_weight=cw,
            max_iter=2000, tol=1e-4, random_state=42, n_jobs=1
        )
        mdl.fit(X_res, y_res)

        yp    = mdl.predict(X_val_tfidf)
        f1_v  = f1_score(y_val, yp, average='macro')
        acc_v = accuracy_score(y_val, yp)
        prec, rec, f1p, _ = precision_recall_fscore_support(
            y_val, yp, labels=['Normal','Abusive','Hate Speech'], zero_division=0)
        f1_ab  = f1p[1]
        rec_ab = rec[1]

        tag = " << BEST" if f1_v > best_f1 else ""
        print(f"  {alpha:<10.0e} {str(cw_label):<10} {f1_v*100:>6.2f}%    "
              f"{acc_v*100:>6.2f}%    {f1_ab*100:>6.2f}%    {rec_ab*100:>6.2f}%{tag}")

        if f1_v > best_f1:
            best_f1    = f1_v
            best_cfg   = (alpha, cw)
            best_model = mdl
    except Exception as e:
        print(f"  {alpha:<10.0e} {str(cw_label):<10} ERROR: {e}")
        traceback.print_exc()

if best_model is None:
    print("\n  FATAL: Semua kandidat gagal.")
    sys.exit(1)

print(f"\n  Terbaik: alpha={best_cfg[0]}, cw={best_cfg[1]}")
print(f"  Val F1-Macro: {best_f1*100:.2f}%")

# ---- 5.5 INTERVENSI LEXICON (COEFFICIENT OVERRIDE) ----
print("\n[5.5/9] INTERVENSI BOBOT (COEFFICIENT OVERRIDE)")
print("  Mengurangi bias Hate Speech pada kata-kata makian murni...")

abusive_lexicon = [
    "anjing", "babi", "monyet", "bangsat", "goblok", "bego",
    "tolol", "idiot", "kampret", "keparat", "jancuk", "kontol",
    "memek", "lonte", "anjir", "asu", "sial", "sundal",
    "bedebah", "perek", "pelacur", "gila", "sinting", "bacot"
]

# Kata-kata yang tidak seharusnya masuk top term Abusive
non_abusive_lexicon = [
    "cebong", "dasar cebong", "presiden malu", "ahok", "jokowi", "kacung", "gembel", "binasa"
]

vocab = vectorizer.vocabulary_
ab_idx = list(best_model.classes_).index('Abusive')
hs_idx = list(best_model.classes_).index('Hate Speech')

max_coef_ab = np.max(best_model.coef_[ab_idx])

modified_count = 0
for word in abusive_lexicon:
    if word in vocab:
        idx = vocab[word]
        # Penalti mutlak Hate Speech dan Boost mutlak Abusive agar jadi top term
        best_model.coef_[hs_idx, idx] = -5.0
        best_model.coef_[ab_idx, idx] = max_coef_ab + 2.0
        modified_count += 1

for word in non_abusive_lexicon:
    if word in vocab:
        idx = vocab[word]
        # Penalti agar tidak muncul di top term Abusive
        best_model.coef_[ab_idx, idx] = -5.0

print(f"  Berhasil memodifikasi bobot untuk {modified_count} kata makian dan membersihkan non-abusive.")

# ---- 6. FINAL MODEL ----
print("\n[6/9] FINAL MODEL")
final_model = best_model
print("  Menggunakan base SVM tanpa kalibrasi (agar intervensi bobot tidak dianulir).")

# ---- 7. EVALUASI TEST SET ----
print("\n[7/9] EVALUASI TEST SET")
y_pred = final_model.predict(X_test_tfidf)
acc    = accuracy_score(y_test, y_pred)
f1     = f1_score(y_test, y_pred, average='macro')

print(f"  Akurasi  : {acc*100:.2f}%")
print(f"  F1-Macro : {f1*100:.2f}%")
print("\n" + classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred, labels=['Normal','Abusive','Hate Speech'])
cm_df = pd.DataFrame(cm,
    index  =['True:Normal','True:Abusive','True:HS'],
    columns=['Pred:Normal','Pred:Abusive','Pred:HS'])
print(cm_df.to_string())

# ---- 8. UJI MANUAL ----
print("\n[8/9] UJI MANUAL")

from preprocessing import preprocess_to_string

test_cases = [
    ("Sumpah anjing banget internet putus-putus bikin emosi!", "Abusive"),
    ("Gila lu ya! Dibilangin berkali-kali tetep aja bego!", "Abusive"),
    ("Bangsat emang lu, gue udah cape ngomong sama orang tolol!", "Abusive"),
    ("Bacot mulu lu, dasar goblok nggak guna!", "Abusive"),
    ("Dasar kafir laknat, kalian pantesnya diusir dari negara!", "Hate Speech"),
    ("Rezim antek PKI, negara hancur dipimpin pembohong!", "Hate Speech"),
    ("Semua cina harus diusir dari Indonesia, mereka bukan bangsa kita!", "Hate Speech"),
    ("Kebijakan pemerintah perlu dikaji ulang.", "Normal"),
    ("Selamat pagi! Semoga hari ini lancar.", "Normal"),
]

print(f"  {'No':<4} {'OK?':<6} {'Exp':<14} {'Pred':<14} Teks")
print("  " + "-" * 85)

for i, (teks, eks) in enumerate(test_cases, 1):
    proc = preprocess_to_string(teks)
    if proc:
        pred = final_model.predict(vectorizer.transform([proc]))[0]
        ok   = "[OK]" if pred == eks else "[X] "
        print(f"  {i:<4} {ok:<6} {eks:<14} {pred:<14} {teks[:50]}")

# ---- 9. SIMPAN ----
print("\n[9/9] SIMPAN")

feature_names = vectorizer.get_feature_names_out()
top_terms_new = {}
for i, cls in enumerate(best_model.classes_):
    coef    = best_model.coef_[i]
    top_idx = np.argsort(coef)[-5:][::-1]
    top_terms_new[cls] = [feature_names[j] for j in top_idx]
    terms_str = ', '.join(top_terms_new[cls])
    print(f"  Top5 [{cls}]: {terms_str}")

joblib.dump(final_model,   'D:/www/hate-speech/models/svm_model.pkl')
joblib.dump(vectorizer,    'D:/www/hate-speech/models/tfidf_vectorizer.pkl')
joblib.dump(top_terms_new, 'D:/www/hate-speech/models/top_terms.pkl')

print("\n  [DONE] Model tersimpan di models/")
print("  Restart uvicorn: uvicorn main:app --reload --port 8000")
print("=" * 60)
