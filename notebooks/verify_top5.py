"""
verify_top5.py
Ekstrak Top-5 fitur dari model SVM final.
"""

import sys
sys.path.insert(0, 'D:/www/hate-speech')

import joblib
import numpy as np

print("=" * 60)
print("VERIFIKASI TOP-5 FITUR — K-15")
print("=" * 60)

# Load vectorizer dan model
vec = joblib.load('D:/www/hate-speech/models/tfidf_vectorizer.pkl')
svm = joblib.load('D:/www/hate-speech/models/svm_model.pkl')

feature_names = vec.get_feature_names_out()
classes = svm.classes_

print("\n--- TOP-5 FITUR SVM FINAL ---")
for i, cls in enumerate(classes):
    # Ambil koefisien untuk kelas tersebut
    coef = svm.coef_[i]
    
    # Ambil 5 indeks dengan koefisien tertinggi
    top_indices = np.argsort(coef)[-5:][::-1]
    top_terms = [feature_names[j] for j in top_indices]
    
    print(f"  [{cls}]: {', '.join(top_terms)}")

print("\n" + "=" * 60)
print("Salin kata-kata di atas ke Tabel 14 di jurnal Anda.")
print("=" * 60)
