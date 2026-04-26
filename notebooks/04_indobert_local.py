# Script untuk Eksperimen IndoBERT di Komputer Lokal (Jupyter Lab / Terminal)
# 
# Peringatan: 
# Jika komputer Anda tidak memiliki GPU NVIDIA (CUDA), proses training akan 
# menggunakan CPU dan memakan waktu cukup lama (bisa berjam-jam).
# Jika Anda punya GPU, pastikan PyTorch versi CUDA sudah terinstall.

import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

# =====================================================================
# 1. Load dan Persiapan Dataset
# =====================================================================
# Mencari path dataset yang benar (mengantisipasi jika dijalankan dari root atau folder notebooks)
possible_paths = [
    "../data/raw/dataset_tweet.csv", 
    "data/raw/dataset_tweet.csv",
    "dataset_tweet.csv"
]

DATA_PATH = None
for path in possible_paths:
    if os.path.exists(path):
        DATA_PATH = path
        break

if DATA_PATH is None:
    print("Error: File dataset dataset_tweet.csv tidak ditemukan di direktori manapun.")
    print("Membuat dummy dataset untuk testing kode agar tidak error...")
    df = pd.DataFrame({
        "Tweet": ["dasar bodoh kau", "halo apa kabar", "cebong pki", "selamat pagi dunia"], 
        "HS": [0, 0, 1, 0],
        "Abusive": [1, 0, 1, 0]
    })
else:
    try:
        df = pd.read_csv(DATA_PATH, encoding='utf-8')
    except UnicodeDecodeError:
        print("Peringatan: Ada karakter tidak standar di CSV. Membaca ulang dengan mode ignore errors...")
        df = pd.read_csv(DATA_PATH, encoding='utf-8', encoding_errors='ignore')
    print(f"Dataset dimuat dari {DATA_PATH}. Jumlah baris: {len(df)}")

# Hapus missing values
df = df.dropna(subset=['Tweet', 'HS', 'Abusive'])

# Konversi ke 3 Kelas seperti model Klasik
def convert_label(row):
    if row['HS'] == 1:
        return 1 # Hate Speech
    elif row['Abusive'] == 1:
        return 2 # Abusive
    else:
        return 0 # Normal

df['label'] = df.apply(convert_label, axis=1)

# Split data: 80% train, 20% test
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['Tweet'].tolist(), 
    df['label'].tolist(), 
    test_size=0.2, 
    random_state=42, 
    stratify=df['label'].tolist()
)

# =====================================================================
# 2. Inisialisasi Tokenizer dari IndoBERT
# =====================================================================
MODEL_NAME = "indobenchmark/indobert-base-p1"
print(f"Memuat tokenizer dari {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# Convert to HuggingFace Dataset format
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

# =====================================================================
# 3. Inisialisasi Model IndoBERT dan Metrik
# =====================================================================
print(f"Memuat model {MODEL_NAME}...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3) # Ubah ke 3 kelas

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Menggunakan device komputasi: {device.type.upper()}")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # Gunakan macro average untuk multi-class classification
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall
    }

# =====================================================================
# 4. Training (Fine-Tuning) Model
# =====================================================================
training_args = TrainingArguments(
    output_dir='./results_indobert', # direktori output
    num_train_epochs=3,              # jumlah epoch
    per_device_train_batch_size=16,  # ukuran batch saat training
    per_device_eval_batch_size=64,   # ukuran batch saat evaluasi
    warmup_steps=500,                # langkah warmup untuk learning rate
    weight_decay=0.01,               # weight decay untuk regularisasi
    logging_dir='./logs_indobert',   # direktori untuk log Tensorboard
    logging_steps=10,
    eval_strategy="epoch",           # evaluasi setiap akhir epoch
    save_strategy="epoch",           # simpan checkpoint setiap epoch
    load_best_model_at_end=True,     # load model terbaik di akhir
    use_cpu=(device.type == "cpu")   # Eksplisit memaksa CPU jika CUDA tidak tersedia
)

trainer = Trainer(
    model=model,                         # model untuk dilatih
    args=training_args,                  # argumen training
    train_dataset=train_dataset,         # dataset training
    eval_dataset=val_dataset,            # dataset validasi
    compute_metrics=compute_metrics      # fungsi kalkulasi metrik
)

print("Memulai proses fine-tuning IndoBERT...")
if device.type == "cpu":
    print("PERINGATAN: Anda menggunakan CPU. Proses training akan berjalan sangat lambat. Anda bisa membatalkan (Ctrl+C) jika dirasa terlalu lama.")

trainer.train()

# =====================================================================
# 5. Evaluasi dan Simpan Model
# =====================================================================
print("Mengevaluasi model pada validation set...")
eval_results = trainer.evaluate()
print("Hasil Evaluasi:", eval_results)

# Menyimpan model
model_save_path = "./indobert_hate_speech_model_local"
tokenizer.save_pretrained(model_save_path)
model.save_pretrained(model_save_path)
print(f"Model berhasil disimpan di {model_save_path}")

# =====================================================================
# 6. Visualisasi (Confusion Matrix & ROC Curve)
# =====================================================================
print("Mulai membuat grafik evaluasi IndoBERT...")
predictions = trainer.predict(val_dataset)
preds = predictions.predictions.argmax(-1)
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
labels = predictions.label_ids

class_names = ['Normal', 'Hate Speech', 'Abusive']

# 1. Confusion Matrix
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Confusion Matrix - IndoBERT')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('indobert_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("Disimpan: indobert_confusion_matrix.png")

# 2. ROC Curve (Multi-class One-vs-Rest)
from sklearn.preprocessing import label_binarize
labels_bin = label_binarize(labels, classes=[0, 1, 2])
plt.figure(figsize=(10, 8))
colors = ['green', 'red', 'orange']

for i, color in zip(range(3), colors):
    fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC curve of class {class_names[i]} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - IndoBERT Multi-Class')
plt.legend(loc="lower right")
plt.savefig('indobert_roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("Disimpan: indobert_roc_curve.png")

# 3. Model Comparison Bar Chart (IndoBERT vs Baseline)
# Nilai akurasi baseline ini didapat dari hasil eksperimen 03_modeling.py
baseline_accuracies = {
    'SVM (TF-IDF)': 0.7813,
    'MNB (TF-IDF)': 0.7522,
    'CNB (TF-IDF)': 0.7322
}
indobert_acc = accuracy_score(labels, preds)

all_models = list(baseline_accuracies.keys()) + ['IndoBERT']
all_accuracies = list(baseline_accuracies.values()) + [indobert_acc]

plt.figure(figsize=(10, 6))
colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
bars = plt.bar(all_models, [acc * 100 for acc in all_accuracies], color=colors)

# Menambahkan label nilai di atas bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold')

plt.ylim(0, 105)
plt.ylabel('Accuracy (%)')
plt.title('Perbandingan Akurasi: Baseline Models vs IndoBERT')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('indobert_vs_baseline_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()
print("Disimpan: indobert_vs_baseline_accuracy.png")

print("Selesai! Semua grafik evaluasi dan perbandingan sudah disimpan ke dalam folder ini.")
