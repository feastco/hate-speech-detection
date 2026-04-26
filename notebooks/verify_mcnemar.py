"""
verify_mcnemar.py
Verifikasi nilai McNemar (b, c, chi2, p-value) untuk K-10.
Jalankan dari direktori: D:/www/hate-speech
  python notebooks/verify_mcnemar.py
"""

import sys
sys.path.insert(0, 'D:/www/hate-speech')

import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from statsmodels.stats.contingency_tables import mcnemar

print("=" * 60)
print("VERIFIKASI McNEMAR — K-10")
print("=" * 60)

# ── 1. Load data ──────────────────────────────────────────────
df_clean = pd.read_csv('D:/www/hate-speech/data/processed/df_clean.csv',
                       encoding='utf-8')
splits = np.load('D:/www/hate-speech/models/shared_split_indices.npz')
idx_test = splits['idx_test']

X = df_clean['processed']
y = df_clean['label']
X_test = X.iloc[idx_test]
y_test = y.iloc[idx_test]

print(f"\nTest set: {len(y_test)} sampel")
print(f"Distribusi:\n{y_test.value_counts()}")

# ── 2. Load vectorizer & models ───────────────────────────────
vec  = joblib.load('D:/www/hate-speech/models/tfidf_vectorizer.pkl')
svm  = joblib.load('D:/www/hate-speech/models/svm_model.pkl')
mnb  = joblib.load('D:/www/hate-speech/models/mnb_model.pkl')
cnb  = joblib.load('D:/www/hate-speech/models/cnb_model.pkl')

X_test_tfidf = vec.transform(X_test)

# ── 3. Prediksi ───────────────────────────────────────────────
y_pred_svm = svm.predict(X_test_tfidf)
y_pred_mnb = mnb.predict(X_test_tfidf)
y_pred_cnb = cnb.predict(X_test_tfidf)

# ── 4. Accuracy & Classification Report ──────────────────────
print("\n" + "-" * 60)
print("ACCURACY MASING-MASING MODEL")
print("-" * 60)
for name, pred in [('SVM', y_pred_svm), ('MNB', y_pred_mnb), ('CNB', y_pred_cnb)]:
    acc = accuracy_score(y_test, pred)
    print(f"  {name}: {acc*100:.4f}%")

print("\n--- Classification Report SVM ---")
print(classification_report(y_test, y_pred_svm, digits=4))

# ── 5. McNemar ────────────────────────────────────────────────
print("-" * 60)
print("McNEMAR TEST (Bonferroni α = 0.05/3 = 0.0167)")
print("-" * 60)

alpha = 0.05 / 3  # 0.0167

pairs = [
    ('SVM', y_pred_svm, 'MNB', y_pred_mnb),
    ('SVM', y_pred_svm, 'CNB', y_pred_cnb),
    ('CNB', y_pred_cnb, 'MNB', y_pred_mnb),
]

results = []
for name_a, pred_a, name_b, pred_b in pairs:
    correct_a = (pred_a == y_test.values)
    correct_b = (pred_b == y_test.values)

    n00 = int(np.sum( correct_a &  correct_b))   # keduanya benar
    n01 = int(np.sum( correct_a & ~correct_b))   # A benar, B salah  → b
    n10 = int(np.sum(~correct_a &  correct_b))   # A salah, B benar  → c
    n11 = int(np.sum(~correct_a & ~correct_b))   # keduanya salah

    b, c = n01, n10
    table = np.array([[n00, n01], [n10, n11]])
    res = mcnemar(table, exact=False, correction=True)

    sig = "✅ SIGNIFIKAN" if res.pvalue < alpha else "❌ Tidak signifikan"
    print(f"\n  {name_a} vs {name_b}:")
    print(f"    Contingency table: [[{n00}, {n01}], [{n10}, {n11}]]")
    print(f"    b = {b}  (hanya {name_a} benar)")
    print(f"    c = {c}  (hanya {name_b} benar)")
    print(f"    χ²  = {res.statistic:.4f}")
    print(f"    p   = {res.pvalue:.6f}")
    print(f"    → {sig} (α = {alpha:.4f})")

    results.append({
        'Pasangan': f'{name_a} vs {name_b}',
        'b': b, 'c': c,
        'chi2': round(res.statistic, 4),
        'p-value': round(res.pvalue, 6),
        'Signifikan': res.pvalue < alpha
    })

# ── 6. Tabel ringkasan ────────────────────────────────────────
print("\n" + "=" * 60)
print("RINGKASAN — SALIN KE LAPORAN")
print("=" * 60)
df_res = pd.DataFrame(results)
print(df_res.to_string(index=False))

# Simpan ke CSV
out_path = 'D:/www/hate-speech/notebooks/mcnemar_verification.csv'
df_res.to_csv(out_path, index=False, encoding='utf-8')
print(f"\n✅ Hasil disimpan ke: {out_path}")
