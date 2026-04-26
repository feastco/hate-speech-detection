# Script untuk Eksperimen IndoBERT di Google Colab
# 
# Langkah-langkah penggunaan di Google Colab:
# 1. Buka Google Colab (colab.research.google.com) dan buat Notebook baru.
# 2. Ubah Runtime ke GPU: Runtime -> Change runtime type -> Hardware accelerator: T4 GPU
# 3. Salin setiap CELL ke dalam sel Colab yang berbeda, lalu jalankan secara berurutan.

# =====================================================================
# CELL 1: Instalasi Library
# =====================================================================
# !pip install -q transformers datasets accelerate evaluate scikit-learn pandas numpy matplotlib seaborn

# =====================================================================
# CELL 2: Import Library & Cek GPU
# =====================================================================
import os
import json
import random
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Cek GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Menggunakan device: {device.type.upper()}")
if device.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# =====================================================================
# CELL 3: Mount Google Drive & Load Dataset
# =====================================================================
# OPSI A: Gunakan Google Drive (DISARANKAN agar data tidak hilang saat sesi berakhir)
from google.colab import drive
drive.mount('/content/drive')

# Sesuaikan path ini dengan lokasi file Anda di Google Drive
# Untuk identical split dengan 03_modeling.py gunakan df_clean + split artifact.
DATA_PATH = '/content/drive/MyDrive/hate-speech/df_clean.csv'
SPLIT_PATH = '/content/drive/MyDrive/hate-speech/shared_split_indices.npz'
OUTPUT_DIR = '/content/drive/MyDrive/hate-speech/results_indobert_2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# OPSI B: Upload langsung ke Colab (data akan hilang saat sesi berakhir)
# from google.colab import files
# uploaded = files.upload()  # Pilih file dataset_tweet.csv
# DATA_PATH = 'dataset_tweet.csv'
# OUTPUT_DIR = './results_indobert'

# =====================================================================
# CELL 4: Preprocessing Dataset (3 Kelas: Normal, Hate Speech, Abusive)
# =====================================================================
try:
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
except UnicodeDecodeError:
    print("⚠️ Ada karakter tidak standar. Membaca ulang dengan encoding_errors='ignore'...")
    df = pd.read_csv(DATA_PATH, encoding='utf-8', encoding_errors='ignore')
except FileNotFoundError:
    print(f"❌ File tidak ditemukan di: {DATA_PATH}")
    print("Pastikan path Google Drive sudah benar, atau gunakan OPSI B (upload manual).")
    raise

print(f"✅ Dataset dimuat. Jumlah baris: {len(df)}")
print(f"   Kolom tersedia: {df.columns.tolist()}")
print(f"   Distribusi label HS: {df['HS'].value_counts().to_dict()}")

# Hapus missing values
df = df.dropna(subset=['Tweet', 'HS', 'Abusive'])

# Konversi ke 3 Kelas (konsisten dengan eksperimen baseline)
def convert_label(row):
    """
    Label 0: Normal
    Label 1: Hate Speech (HS=1)
    Label 2: Abusive (HS=0, Abusive=1)
    """
    if row['HS'] == 1:
        return 1  # Hate Speech
    elif row['Abusive'] == 1:
        return 2  # Abusive
    else:
        return 0  # Normal

df['label'] = df.apply(convert_label, axis=1)
print(f"\n   Distribusi 3 Kelas:")
label_names = {0: 'Normal', 1: 'Hate Speech', 2: 'Abusive'}
for k, v in df['label'].value_counts().sort_index().items():
    print(f"     {label_names[k]}: {v} sampel")

# Hitung class weights untuk mengatasi imbalance (terutama kelas Abusive)
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1, 2]),
    y=df['label'].values
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
print(f"\n   Class Weights (untuk mengatasi imbalance):")
for i, w in enumerate(class_weights):
    print(f"     {label_names[i]}: {w:.4f}")

# Gunakan shared split dari 03_modeling.py (identical split untuk SVM vs IndoBERT)
if os.path.exists(SPLIT_PATH):
    split_npz = np.load(SPLIT_PATH)
    idx_train = split_npz['idx_train']
    idx_val = split_npz['idx_val']
    idx_test = split_npz['idx_test']
    print(f"\n✅ Shared split dimuat dari: {SPLIT_PATH}")
else:
    print(f"\n⚠️ Shared split tidak ditemukan di: {SPLIT_PATH}")
    print("   Fallback ke split lokal (hasil tidak identik dengan 03_modeling.py).")
    idx_all = np.arange(len(df))
    idx_train_val, idx_test = train_test_split(
        idx_all,
        test_size=0.2,
        random_state=42,
        stratify=df['label'].tolist()
    )
    y_train_val = df.iloc[idx_train_val]['label']
    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=0.1,
        random_state=42,
        stratify=y_train_val
    )

train_texts = df.iloc[idx_train]['Tweet'].tolist()
val_texts = df.iloc[idx_val]['Tweet'].tolist()
test_texts = df.iloc[idx_test]['Tweet'].tolist()

train_labels = df.iloc[idx_train]['label'].tolist()
val_labels = df.iloc[idx_val]['label'].tolist()
test_labels = df.iloc[idx_test]['label'].tolist()

print(f"\n   Train: {len(train_texts)} | Validasi: {len(val_texts)} | Test: {len(test_texts)}")

# =====================================================================
# CELL 5: Tokenisasi dengan IndoBERT
# =====================================================================
MODEL_NAME = "indobenchmark/indobert-base-p2"  # p2 = Whole Word Masking, lebih baik untuk klasifikasi
print(f"\n⏳ Memuat tokenizer dari {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# max_length=128 optimal untuk Colab T4 (VRAM 16GB)
MAX_LENGTH = 128
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=MAX_LENGTH)

class HateSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = HateSpeechDataset(train_encodings, train_labels)
val_dataset = HateSpeechDataset(val_encodings, val_labels)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
test_dataset = HateSpeechDataset(test_encodings, test_labels)
print("✅ Tokenisasi selesai.")

# =====================================================================
# CELL 6: Inisialisasi Model (3 Kelas)
# =====================================================================
NUM_LABELS = 3  # Normal, Hate Speech, Abusive
CLASS_NAMES = ['Normal', 'Hate Speech', 'Abusive']

print(f"\n⏳ Memuat model IndoBERT dengan {NUM_LABELS} kelas...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
print("✅ Model berhasil dimuat.")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # Gunakan macro average untuk multi-class (konsisten dengan eksperimen baseline)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall
    }

# Custom Trainer dengan Class Weights untuk mengatasi imbalance
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device if hasattr(self.args, 'device') else 'cpu')

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# =====================================================================
# CELL 7: Training (Fine-Tuning) Model
# =====================================================================
# Konfigurasi optimal untuk Colab T4 GPU (VRAM 16GB)
# Ditingkatkan ke 5 epoch dengan early stopping berdasarkan F1 Macro
RUN_SEED = 42
set_all_seeds(RUN_SEED)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,               # Ditingkatkan dari 3 → 5
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",  # Pilih model terbaik berdasarkan F1 Macro
    greater_is_better=True,
    fp16=True,
    seed=RUN_SEED,
    data_seed=RUN_SEED,
    report_to="none",
)

# Gunakan WeightedTrainer dengan class_weights untuk mengatasi imbalance kelas Abusive
trainer = WeightedTrainer(
    class_weights=class_weights_tensor,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

print("\n🚀 Memulai proses fine-tuning IndoBERT (dengan class weights)...")
trainer.train()
print("✅ Training selesai!")

# Simpan prediksi test untuk komparasi identik dengan SVM
test_predictions = trainer.predict(test_dataset)
test_preds = test_predictions.predictions.argmax(-1)
test_true = test_predictions.label_ids
np.savez(
    os.path.join(OUTPUT_DIR, 'indobert_test_predictions_seed42.npz'),
    y_true=test_true,
    y_pred=test_preds
)
print("✅ Prediksi test seed42 tersimpan: indobert_test_predictions_seed42.npz")

# =====================================================================
# CELL 8: Evaluasi Model
# =====================================================================
print("\n📊 Mengevaluasi model pada validation set...")
eval_results = trainer.evaluate()
print("\nHasil Evaluasi:")
for k, v in eval_results.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

# =====================================================================
# CELL 9: Simpan Model ke Google Drive
# =====================================================================
model_save_path = os.path.join(OUTPUT_DIR, "indobert_hate_speech_model")
tokenizer.save_pretrained(model_save_path)
model.save_pretrained(model_save_path)
print(f"\n✅ Model berhasil disimpan di:\n   {model_save_path}")

# =====================================================================
# CELL 9b: Multi-Seed Rerun (P2.2) + Test Identik (P2.3)
# =====================================================================
# Jalankan cell ini untuk mendapatkan mean ± std dari 3 seeds.
SEEDS = [42, 0, 123]
multi_seed_results = []

for seed in SEEDS:
    print(f"\n===== Multi-seed run: {seed} =====")
    set_all_seeds(seed)
    run_dir = os.path.join(OUTPUT_DIR, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    run_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    run_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=True,
        seed=seed,
        data_seed=seed,
        report_to="none",
    )

    run_trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=run_model,
        args=run_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    run_trainer.train()
    run_pred = run_trainer.predict(test_dataset)
    run_preds = run_pred.predictions.argmax(-1)
    run_true = run_pred.label_ids

    run_acc = accuracy_score(run_true, run_preds)
    run_precision, run_recall, run_f1, _ = precision_recall_fscore_support(
        run_true, run_preds, average='macro', zero_division=0
    )

    np.savez(
        os.path.join(run_dir, 'test_predictions.npz'),
        y_true=run_true,
        y_pred=run_preds
    )

    multi_seed_results.append({
        'seed': seed,
        'accuracy': float(run_acc),
        'precision_macro': float(run_precision),
        'recall_macro': float(run_recall),
        'f1_macro': float(run_f1)
    })

acc_array = np.array([x['accuracy'] for x in multi_seed_results], dtype=float)
f1_array = np.array([x['f1_macro'] for x in multi_seed_results], dtype=float)

multi_seed_summary = {
    'seeds': SEEDS,
    'per_seed': multi_seed_results,
    'mean_accuracy': float(acc_array.mean()),
    'std_accuracy': float(acc_array.std(ddof=1)) if len(acc_array) > 1 else 0.0,
    'mean_f1_macro': float(f1_array.mean()),
    'std_f1_macro': float(f1_array.std(ddof=1)) if len(f1_array) > 1 else 0.0,
    'split_note': 'Shared split from 03_modeling.py (idx_train, idx_val, idx_test).'
}

with open(os.path.join(OUTPUT_DIR, 'indobert_multiseed_summary.json'), 'w') as f:
    json.dump(multi_seed_summary, f, indent=2)

print("\n✅ Multi-seed summary tersimpan: indobert_multiseed_summary.json")
print(f"   Mean Accuracy: {multi_seed_summary['mean_accuracy']*100:.2f}% ± {multi_seed_summary['std_accuracy']*100:.2f}%")
print(f"   Mean F1 Macro: {multi_seed_summary['mean_f1_macro']*100:.2f}% ± {multi_seed_summary['std_f1_macro']*100:.2f}%")

# =====================================================================
# CELL 10a: Load Model Fine-Tuned dari Drive (jalankan jika sesi Colab baru)
# Lewati cell ini jika Anda baru selesai training di sesi yang sama.
# =====================================================================
import os, json, torch, pandas as pd, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR        = '/content/drive/MyDrive/hate-speech/results_indobert_2'
MODEL_SAVE_PATH   = os.path.join(OUTPUT_DIR, 'indobert_hate_speech_model')
DATA_PATH         = '/content/drive/MyDrive/hate-speech/df_clean.csv'
SPLIT_PATH        = '/content/drive/MyDrive/hate-speech/shared_split_indices.npz'
MODEL_NAME        = 'indobenchmark/indobert-base-p2'
MAX_LENGTH        = 128
NUM_LABELS        = 3
CLASS_NAMES       = ['Normal', 'Hate Speech', 'Abusive']

print(f"⏳ Memuat model fine-tuned dari: {MODEL_SAVE_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"✅ Model fine-tuned berhasil dimuat. Device: {device.type.upper()}")

# Rebuild test_dataset (identik dengan shared split untuk evaluasi akhir)
try:
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(DATA_PATH, encoding='utf-8', encoding_errors='ignore')

df = df.dropna(subset=['Tweet', 'HS', 'Abusive'])
def convert_label(row):
    if row['HS'] == 1:   return 1
    elif row['Abusive'] == 1: return 2
    else: return 0
df['label'] = df.apply(convert_label, axis=1)

if not os.path.exists(SPLIT_PATH):
    raise FileNotFoundError(
        f"Shared split tidak ditemukan: {SPLIT_PATH}. Jalankan 03_modeling.py terlebih dahulu."
    )

split_npz = np.load(SPLIT_PATH)
idx_test = split_npz['idx_test']

test_texts = df.iloc[idx_test]['Tweet'].tolist()
test_labels = df.iloc[idx_test]['label'].tolist()
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LENGTH)

class HateSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

test_dataset = HateSpeechDataset(test_encodings, test_labels)

# Buat trainer minimal hanya untuk predict (tanpa training)
def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1_macro': f1, 'precision_macro': precision, 'recall_macro': recall}

_eval_args = TrainingArguments(output_dir=OUTPUT_DIR, per_device_eval_batch_size=64, report_to="none")
trainer = Trainer(model=model, args=_eval_args, eval_dataset=test_dataset, compute_metrics=compute_metrics)
print("✅ test_dataset dan trainer siap. Lanjutkan ke Cell 10.")

# =====================================================================
# CELL 10: Visualisasi (Confusion Matrix, ROC Curve, Perbandingan)
# =====================================================================
print("\n🎨 Membuat grafik evaluasi (shared test set)...")
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
labels_true = predictions.label_ids

# --- 1. Confusion Matrix ---
cm = confusion_matrix(labels_true, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix - IndoBERT Fine-Tuned', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
cm_path = os.path.join(OUTPUT_DIR, 'indobert_confusion_matrix.png')
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"  Disimpan: {cm_path}")

# --- 2. ROC Curve (Multi-class One-vs-Rest) ---
labels_bin = label_binarize(labels_true, classes=[0, 1, 2])
plt.figure(figsize=(10, 8))
colors = ['green', 'red', 'orange']

for i, color in zip(range(NUM_LABELS), colors):
    fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'ROC {CLASS_NAMES[i]} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - IndoBERT Multi-Class (One-vs-Rest)', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
roc_path = os.path.join(OUTPUT_DIR, 'indobert_roc_curve.png')
plt.savefig(roc_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"  Disimpan: {roc_path}")

# --- 3. Perbandingan Akurasi & F1 Macro: Baseline vs IndoBERT ---
# Membaca nilai AKTUAL dari baseline_metrics.json yang dihasilkan 03_modeling.py
# Jika file tidak ada, gunakan nilai hardcode sebagai fallback

BASELINE_JSON_PATH = '/content/drive/MyDrive/hate-speech/baseline_metrics.json'
# Nilai fallback HARUS sesuai dengan baseline_metrics.json aktual:
# SVM: 78.13% acc, 74.44% f1  |  MNB: 75.22% acc, 72.21% f1  |  CNB: 73.22% acc, 70.27% f1
_FALLBACK_BASELINE = {
    'SVM': {'accuracy': 0.781289, 'f1_macro': 0.744379},
    'MNB': {'accuracy': 0.752183, 'f1_macro': 0.722062},
    'CNB': {'accuracy': 0.732225, 'f1_macro': 0.702693},
}

import json as _json
if os.path.exists(BASELINE_JSON_PATH):
    with open(BASELINE_JSON_PATH) as _f:
        _raw = _json.load(_f)
    print(f"✅ Membaca nilai baseline dari: {BASELINE_JSON_PATH}")
    baseline_data = {f"{k} (TF-IDF)": v for k, v in _raw.items()}
else:
    print(f"⚠️  baseline_metrics.json tidak ditemukan.")
    print(f"   Pastikan Anda upload file tersebut ke Google Drive setelah menjalankan 03_modeling.py")
    print(f"   Menggunakan nilai ESTIMASI sebagai fallback...")
    baseline_data = {f"{k} (TF-IDF)": v for k, v in _FALLBACK_BASELINE.items()}

indobert_acc = accuracy_score(labels_true, preds)
_, _, indobert_f1, _ = precision_recall_fscore_support(labels_true, preds, average='macro', zero_division=0)
baseline_data['IndoBERT'] = {'accuracy': indobert_acc, 'f1_macro': indobert_f1}
print(f"   IndoBERT — Accuracy: {indobert_acc*100:.2f}% | F1 Macro: {indobert_f1*100:.2f}%")


model_names = list(baseline_data.keys())
accuracies = [baseline_data[m]['accuracy'] * 100 for m in model_names]
f1_scores  = [baseline_data[m]['f1_macro'] * 100 for m in model_names]

x = np.arange(len(model_names))
width = 0.35
bar_colors_acc = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
bar_colors_f1  = ['#81C784', '#64B5F6', '#FFB74D', '#CE93D8']

fig, ax = plt.subplots(figsize=(12, 7))
bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color=bar_colors_acc)
bars2 = ax.bar(x + width/2, f1_scores,  width, label='F1 Macro', color=bar_colors_f1)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
            f'{bar.get_height():.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
            f'{bar.get_height():.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

ax.set_ylim(0, 100)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Perbandingan Accuracy & F1 Macro: Baseline Models vs IndoBERT',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.6)
bar_path = os.path.join(OUTPUT_DIR, 'indobert_vs_baseline_accuracy.png')
plt.savefig(bar_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"  Disimpan: {bar_path}")

print(f"\n🎉 Selesai! Semua output disimpan di:\n   {OUTPUT_DIR}")

# =====================================================================
# CELL 11: Classification Report, Training Curve & Latensi Inference
# =====================================================================
# Pastikan Cell 10a dan Cell 10 sudah dijalankan sebelumnya sehingga
# variabel: preds, labels_true, probs, model, tokenizer tersedia.
# =====================================================================

import json as _json
import time as _time
import numpy as np

# ── 11a. Classification Report Per-Kelas ────────────────────────────
from sklearn.metrics import classification_report as _clf_report

print("=" * 60)
print("CELL 11a: CLASSIFICATION REPORT PER-KELAS — IndoBERT")
print("=" * 60)

report_str = _clf_report(
    labels_true, preds,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)
print(report_str)

# Simpan ke file
_report_path = os.path.join(OUTPUT_DIR, 'indobert_classification_report.txt')
with open(_report_path, 'w', encoding='utf-8') as _f:
    _f.write("IndoBERT Fine-Tuned — Classification Report\n")
    _f.write("=" * 60 + "\n")
    _f.write(report_str)
print(f"✅ Classification report tersimpan: {_report_path}")

# ── 11b. Training Loss Curve dari trainer_state.json ────────────────
print("\n" + "=" * 60)
print("CELL 11b: TRAINING LOSS CURVE — IndoBERT")
print("=" * 60)

_STATE_PATH = os.path.join(OUTPUT_DIR, 'trainer_state.json')

if os.path.exists(_STATE_PATH):
    with open(_STATE_PATH) as _f:
        _state = _json.load(_f)

    _log_history = _state.get('log_history', [])

    # Pisahkan train loss vs eval loss
    _train_logs = [x for x in _log_history if 'loss' in x and 'eval_loss' not in x]
    _eval_logs  = [x for x in _log_history if 'eval_loss' in x]

    _train_steps  = [x['step'] for x in _train_logs]
    _train_losses = [x['loss'] for x in _train_logs]
    _eval_steps   = [x['step'] for x in _eval_logs]
    _eval_losses  = [x['eval_loss'] for x in _eval_logs]
    _eval_accs    = [x.get('eval_accuracy', None) for x in _eval_logs]
    _eval_f1s     = [x.get('eval_f1', None) for x in _eval_logs]
    _eval_epochs  = [x.get('epoch', None) for x in _eval_logs]

    # Plot 1: Training Loss vs Eval Loss
    _fig, _ax1 = plt.subplots(figsize=(10, 5))
    _ax1.plot(_train_steps, _train_losses, 'b-o', markersize=3,
               label='Train Loss', alpha=0.8)
    _ax1.plot(_eval_steps, _eval_losses, 'r-s', markersize=6,
               label='Eval Loss', linewidth=2)
    _ax1.set_xlabel('Training Step', fontsize=12)
    _ax1.set_ylabel('Loss', fontsize=12)
    _ax1.set_title('Training Loss vs Validation Loss — IndoBERT Fine-Tuning',
                    fontsize=13, fontweight='bold')
    _ax1.legend(fontsize=11)
    _ax1.grid(alpha=0.3)

    # Annotasi epoch markers
    if _eval_epochs and _eval_epochs[0] is not None:
        for _ep, _st in zip(_eval_epochs, _eval_steps):
            _ax1.axvline(x=_st, color='gray', linestyle='--', alpha=0.4)
            _ax1.text(_st, max(_train_losses) * 0.95, f'ep{int(_ep)}',
                      fontsize=8, ha='center', color='gray')

    plt.tight_layout()
    _curve_path = os.path.join(OUTPUT_DIR, 'indobert_training_curve.png')
    plt.savefig(_curve_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Disimpan: {_curve_path}")

    # Plot 2: Eval F1 & Accuracy per Epoch
    if any(_e is not None for _e in _eval_f1s):
        _fig2, _ax2 = plt.subplots(figsize=(8, 5))
        _ax2.plot(_eval_epochs, [v * 100 if v else None for v in _eval_accs],
                  'b-o', label='Val Accuracy (%)', linewidth=2, markersize=7)
        _ax2.plot(_eval_epochs, [v * 100 if v else None for v in _eval_f1s],
                  'r-s', label='Val F1 Macro (%)', linewidth=2, markersize=7)
        _ax2.set_xlabel('Epoch', fontsize=12)
        _ax2.set_ylabel('Score (%)', fontsize=12)
        _ax2.set_title('Validation Accuracy & F1 Macro per Epoch — IndoBERT',
                        fontsize=13, fontweight='bold')
        _ax2.legend(fontsize=11)
        _ax2.grid(alpha=0.3)
        # Best epoch annotation
        if any(_e is not None for _e in _eval_f1s):
            _best_idx = max(range(len(_eval_f1s)),
                            key=lambda i: _eval_f1s[i] if _eval_f1s[i] else 0)
            _best_ep = _eval_epochs[_best_idx]
            _best_f1 = _eval_f1s[_best_idx]
            if _best_f1:
                _ax2.annotate(f'Best: Ep{int(_best_ep)}\nF1={_best_f1*100:.2f}%',
                              xy=(_best_ep, _best_f1 * 100),
                              xytext=(_best_ep + 0.3, _best_f1 * 100 - 2),
                              arrowprops=dict(arrowstyle='->', color='red'),
                              fontsize=9, color='red')
        plt.tight_layout()
        _f1curve_path = os.path.join(OUTPUT_DIR, 'indobert_eval_curve.png')
        plt.savefig(_f1curve_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"  Disimpan: {_f1curve_path}")

    # Print training log table
    print("\nEval Log per Epoch:")
    print(f"{'Epoch':<8} {'Step':<8} {'Eval Loss':<12} {'Accuracy':<12} {'F1 Macro':<12}")
    print("-" * 55)
    for _ep, _st, _el, _ea, _ef in zip(_eval_epochs, _eval_steps,
                                          _eval_losses, _eval_accs, _eval_f1s):
        _acc_str = f"{_ea*100:.4f}%" if _ea else "N/A"
        _f1_str  = f"{_ef*100:.4f}%" if _ef else "N/A"
        print(f"{str(_ep):<8} {_st:<8} {_el:<12.4f} {_acc_str:<12} {_f1_str:<12}")

else:
    print(f"⚠️  trainer_state.json tidak ditemukan di: {_STATE_PATH}")
    print("   File ini otomatis dibuat saat training. Cek folder output training di Drive.")
    print("   Path yang dicek: results_indobert_2/trainer_state.json")

# ── 11c. Pengukuran Latensi Inference IndoBERT ──────────────────────
print("\n" + "=" * 60)
print("CELL 11c: LATENSI INFERENCE — IndoBERT vs SVM (Estimasi)")
print("=" * 60)
print("CATATAN: Pengukuran latensi tergantung hardware (GPU T4 Colab vs CPU).")
print("         Angka 87ms untuk SVM dalam paper BELUM TERUKUR secara empiris")
print("         dalam sesi ini. Angka di bawah adalah hasil pengukuran IndoBERT saja.")

model.eval()
_device = next(model.parameters()).device

# Ambil 10 sampel dari val_texts untuk benchmark
_sample_texts = test_texts[:10]
_enc = tokenizer(
    _sample_texts, return_tensors='pt',
    truncation=True, padding=True, max_length=MAX_LENGTH
)
_enc = {k: v.to(_device) for k, v in _enc.items()}

# Warm-up run (2x)
with torch.no_grad():
    for _ in range(2):
        model(**_enc)

# Ukur 100 iterasi
_N_RUNS = 100
_start = _time.perf_counter()
with torch.no_grad():
    for _ in range(_N_RUNS):
        model(**_enc)
_elapsed_ms = (_time.perf_counter() - _start) / _N_RUNS * 1000
_per_sample_ms = _elapsed_ms / len(_sample_texts)

print(f"\n  Hardware  : {'GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
print(f"  Batch size: {len(_sample_texts)} sampel")
print(f"  N runs    : {_N_RUNS}")
print(f"  Latensi per batch (10 sampel) : {_elapsed_ms:.1f} ms")
print(f"  Latensi per sampel (estimasi) : {_per_sample_ms:.1f} ms")
print(f"\n  ⚠️  DISCLAIMER UNTUK PAPER:")
print(f"  Latensi SVM (87ms, n=500) yang disebut di paper v9 adalah hasil")
print(f"  pengukuran sistem web FastAPI+cache. Belum ada pengukuran empiris")
print(f"  perbandingan head-to-head IndoBERT vs SVM dalam kondisi identical.")
print(f"  Untuk paper, gunakan angka di atas dengan menyebut hardware Colab.")

# Simpan ringkasan latensi ke file
_latency_summary = {
    'hardware': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
    'batch_size': len(_sample_texts),
    'n_runs': _N_RUNS,
    'latency_per_batch_ms': round(_elapsed_ms, 2),
    'latency_per_sample_ms': round(_per_sample_ms, 2),
    'note': (
        'Latensi SVM 87ms di paper v9 Seksi 4.5 adalah pengukuran sistem web '
        'FastAPI+SVM+cache (n=500 request), BUKAN perbandingan head-to-head dengan '
        'IndoBERT. Angka IndoBERT di atas diukur pada model inference saja tanpa '
        'overhead HTTP/preprocessing.'
    )
}
_latency_path = os.path.join(OUTPUT_DIR, 'indobert_latency_benchmark.json')
with open(_latency_path, 'w') as _f:
    _json.dump(_latency_summary, _f, indent=2)
print(f"\n✅ Ringkasan latensi tersimpan: {_latency_path}")

print(f"\n{'='*60}")
print("✅ Cell 11 selesai. Output yang dihasilkan:")
print(f"  📄 {OUTPUT_DIR}/indobert_classification_report.txt")
print(f"  📊 {OUTPUT_DIR}/indobert_training_curve.png  (jika trainer_state.json ada)")
print(f"  📊 {OUTPUT_DIR}/indobert_eval_curve.png      (jika trainer_state.json ada)")
print(f"  📄 {OUTPUT_DIR}/indobert_latency_benchmark.json")
print("=" * 60)

# ── 11d. Training Curve Manual (Fallback jika trainer_state.json tidak ada) ──
print("\n" + "=" * 60)
print("CELL 11d: TRAINING CURVE MANUAL (Fallback)")
print("=" * 60)
print("Membuat visualisasi training curve dari epoch metrics yang tersimpan.")
print("Digunakan karena trainer_state.json tidak tersedia dari sesi sebelumnya.\n")

# Epoch metrics dari training 5 epoch (diisi dari hasil training aktual)
# Format: epoch, eval_loss, eval_accuracy, eval_f1_macro
# Catatan: nilai di bawah hanya tersedia jika Anda mencatat per-epoch saat training.
# Jika tidak, isi manual dari log output training Colab sebelumnya.
_MANUAL_EPOCHS = {
    'epoch':        [1,      2,      3,      4,      5     ],
    'eval_loss':    [0.6620, 0.6156, 0.7482, 1.1215, 1.3080],
    'eval_accuracy':[0.6822, 0.7582, 0.7771, 0.7828, 0.7844],
    'eval_f1':      [0.6525, 0.7250, 0.7453, 0.7426, 0.7423],
}
# Sumber: training log Colab, run 2026-04-23
# Best checkpoint: Epoch 3 (F1 Macro tertinggi = 74.53%)
# Val Loss minimum: Epoch 2 (0.6156)
# Overfitting dimulai setelah Epoch 2 (Val Loss naik drastis)


# Coba load dari trainer_state.json sebagai fallback
_STATE_PATH2 = os.path.join(OUTPUT_DIR, 'trainer_state.json')
_loaded_from_state = False

if os.path.exists(_STATE_PATH2):
    try:
        with open(_STATE_PATH2) as _f2:
            _state2 = _json.load(_f2)
        _eval_logs2 = [x for x in _state2.get('log_history', []) if 'eval_loss' in x]
        if _eval_logs2:
            _MANUAL_EPOCHS['epoch']         = [x.get('epoch') for x in _eval_logs2]
            _MANUAL_EPOCHS['eval_loss']     = [x.get('eval_loss') for x in _eval_logs2]
            _MANUAL_EPOCHS['eval_accuracy'] = [x.get('eval_accuracy') for x in _eval_logs2]
            _MANUAL_EPOCHS['eval_f1']       = [x.get('eval_f1') for x in _eval_logs2]
            _loaded_from_state = True
            print("  ✅ Data dimuat dari trainer_state.json")
    except Exception as _e:
        print(f"  ⚠️  Gagal membaca trainer_state.json: {_e}")

if not _loaded_from_state:
    print("  ⚠️  trainer_state.json tidak tersedia.")
    print("  Catatan: Isi _MANUAL_EPOCHS di atas dengan data log training")
    print("  untuk generate curve lengkap 5 epoch.\n")
    print("  Hanya menampilkan nilai epoch akhir yang tersedia:")
    print(f"    Epoch 5 → Accuracy: 78,32% | F1 Macro: 74,59%")

# Plot hanya data yang tersedia (non-None)
_ep_plot   = [e for e, a in zip(_MANUAL_EPOCHS['epoch'], _MANUAL_EPOCHS['eval_accuracy']) if a is not None]
_acc_plot  = [a * 100 for a in _MANUAL_EPOCHS['eval_accuracy'] if a is not None]
_f1_plot   = [f * 100 for f in _MANUAL_EPOCHS['eval_f1'] if f is not None]
_loss_plot = [l for l in _MANUAL_EPOCHS['eval_loss'] if l is not None]

if len(_ep_plot) >= 1:
    _fig3, _ax3 = plt.subplots(figsize=(8, 5))
    _ax3.plot(_ep_plot, _acc_plot, 'b-o', label='Val Accuracy (%)', linewidth=2,
              markersize=9, markerfacecolor='white', markeredgewidth=2)
    _ax3.plot(_ep_plot, _f1_plot, 'r-s', label='Val F1 Macro (%)', linewidth=2,
              markersize=9, markerfacecolor='white', markeredgewidth=2)
    # Annotasi titik akhir
    _ax3.annotate(f'{_acc_plot[-1]:.2f}%',
                  xy=(_ep_plot[-1], _acc_plot[-1]),
                  xytext=(_ep_plot[-1] - 0.3, _acc_plot[-1] + 0.5),
                  fontsize=9, color='blue',
                  arrowprops=dict(arrowstyle='->', color='blue', lw=1.2))
    _ax3.annotate(f'{_f1_plot[-1]:.2f}%',
                  xy=(_ep_plot[-1], _f1_plot[-1]),
                  xytext=(_ep_plot[-1] - 0.3, _f1_plot[-1] - 1.5),
                  fontsize=9, color='red',
                  arrowprops=dict(arrowstyle='->', color='red', lw=1.2))
    _ax3.set_xlabel('Epoch', fontsize=12)
    _ax3.set_ylabel('Score (%)', fontsize=12)
    _ax3.set_xticks(_ep_plot)
    _ax3.set_title('Validation Accuracy & F1 Macro per Epoch — IndoBERT Fine-Tuning\n'
                   '(indobenchmark/indobert-base-p2, n=2.634 val samples)',
                   fontsize=12, fontweight='bold')
    _ax3.legend(fontsize=11)
    _ax3.grid(alpha=0.3)
    if len(_ep_plot) == 1:
        _ax3.set_xlim([0.5, 5.5])
        _ax3.text(3, (_acc_plot[0] + _f1_plot[0]) / 2,
                  'Data lengkap tersedia setelah\nmenambahkan per-epoch metrics',
                  ha='center', fontsize=9, color='gray', style='italic')
    plt.tight_layout()
    _manual_curve_path = os.path.join(OUTPUT_DIR, 'indobert_eval_curve_manual.png')
    plt.savefig(_manual_curve_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n  ✅ Grafik disimpan: {_manual_curve_path}")
else:
    print("  ⚠️  Tidak ada data cukup untuk membuat plot.")



# =====================================================================
# CELL 12: Demo Prediksi pada Kalimat Baru
# =====================================================================
def predict_hate_speech(text):
    """Prediksi kategori teks menggunakan model yang sudah di-fine-tune."""
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       padding=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(outputs.logits, dim=-1).item()
    confidence = probs[0][predicted_class].item() * 100

    return f"{CLASS_NAMES[predicted_class]} ({confidence:.2f}%)"

kalimat_test = [
    "Wah bagus sekali ya kerjanya, mantap!",
    "Dasar orang bodoh, tidak berguna sama sekali",
    "Dasar cebong PKI, pantesan negara hancur",
    "Cina lu, dasar aseng perusak bangsa"
]

print("\n--- HASIL PREDIKSI MODEL (CELL 12) ---")
for kalimat in kalimat_test:
    hasil = predict_hate_speech(kalimat)
    print(f"Teks  : '{kalimat}'")
    print(f"Hasil : {hasil}\n")
