import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Setup
os.makedirs(r'd:\www\hate-speech\image', exist_ok=True)
sns.set_theme(style="whitegrid")

print("1. Generating 10-Fold CV Chart...")
# 1. 10-Fold CV Bar Chart with Error Bars (representing stability)
models = ['MNB', 'CNB', 'SVM']
means = [81.62, 81.05, 87.06]
stds = [1.00, 0.89, 0.71]

plt.figure(figsize=(8, 6))
bars = plt.bar(models, means, yerr=stds, capsize=10, color=['#4C72B0', '#55A868', '#C44E52'], alpha=0.8)
plt.ylim(75, 90)
plt.ylabel('10-Fold CV Mean Accuracy (%)', fontsize=12)
plt.title('Stabilitas Model Klasikal (10-Fold Cross Validation)\nSemakin tinggi dan error bar semakin kecil = semakin stabil', fontsize=14, pad=15)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval - 2, f'{yval:.2f}%', ha='center', va='bottom', color='white', fontweight='bold', fontsize=11)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(r'd:\www\hate-speech\image\cv_stability_chart.png', dpi=300, bbox_inches='tight')
plt.close()


print("2. Generating Accuracy vs Latency Bubble Chart...")
# 2. Accuracy vs Latency Bubble Chart
labels = ['SVM (CPU)', 'MNB (CPU)', 'CNB (CPU)', 'IndoBERT (GPU)']
latency = [0.10, 0.16, 0.17, 4.89]
accuracy = [78.13, 75.22, 73.22, 78.32]
colors = ['#C44E52', '#4C72B0', '#55A868', '#8172B2']

plt.figure(figsize=(10, 6))
for i in range(len(labels)):
    plt.scatter(latency[i], accuracy[i], s=800, c=colors[i], label=labels[i], alpha=0.7, edgecolors='k', linewidth=1.5)

# Annotations
for i, txt in enumerate(labels):
    offset = (15, -15) if i == 3 else (15, 10)
    plt.annotate(f"{txt}\n{accuracy[i]}%, {latency[i]}ms", (latency[i], accuracy[i]), 
                 xytext=offset, textcoords='offset points', fontsize=11, fontweight='500',
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))

plt.xscale('log')
plt.xlabel('Latensi Inferensi per Sampel (ms) - Log Scale', fontsize=12)
plt.ylabel('Akurasi Holdout (%)', fontsize=12)
plt.title('Trade-off: Akurasi vs Latensi Inferensi\n(Kiri Atas = Paling Ideal)', fontsize=14, pad=15)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(r'd:\www\hate-speech\image\accuracy_latency_tradeoff.png', dpi=300, bbox_inches='tight')
plt.close()


print("3. Generating Learning Curve...")
# 3. Learning Curve (Loss vs Epoch)
epochs = [1, 2, 3, 4, 5]
train_loss = [0.6521, 0.4631, 0.2839, 0.1375, 0.0594]
val_loss = [0.6620, 0.6156, 0.7482, 1.1215, 1.3080]

plt.figure(figsize=(9, 6))
plt.plot(epochs, train_loss, marker='o', linestyle='-', color='#4C72B0', linewidth=2, markersize=8, label='Training Loss')
plt.plot(epochs, val_loss, marker='s', linestyle='--', color='#C44E52', linewidth=2, markersize=8, label='Validation Loss')

# Highlights
plt.axvline(x=3, color='gray', linestyle=':', alpha=0.7)
plt.text(3.1, 1.0, 'Best Checkpoint\n(Max F1: 74.53%)', fontsize=11, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
plt.plot(2, 0.6156, marker='*', markersize=18, color='gold', markeredgecolor='k', label='Min Val Loss')

plt.xticks(epochs)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Cross-Entropy Loss', fontsize=12)
plt.title('Dinamika Training IndoBERT\n(Val Loss naik setelah Epoch 2 mengindikasikan Overfitting)', fontsize=14, pad=15)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig(r'd:\www\hate-speech\image\indobert_learning_curve.png', dpi=300, bbox_inches='tight')
plt.close()


print("4. Generating SVM Confusion Matrix...")
# 4. SVM Confusion Matrix (Reconstructed from Error Analysis CSV)
# Abusive: 293 total (198 True A, 60 Pred HS, 35 Pred N)
# Hate Speech: 1016 total (92 Pred A, 777 True HS, 147 Pred N)
# Normal: 1096 total (50 Pred A, 142 Pred HS, 904 True N)
svm_cm = np.array([
    [198, 60, 35],
    [92, 777, 147],
    [50, 142, 904]
])
classes = ['Abusive', 'Hate Speech', 'Normal']

plt.figure(figsize=(7, 6))
sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, annot_kws={"size": 12})
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion Matrix SVM (Baseline Terbaik)', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig(r'd:\www\hate-speech\image\svm_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()


print("5. Generating Token Distribution...")
# 5. Token Distribution (from dataset)
try:
    df = pd.read_csv(r'd:\www\hate-speech\data\raw\dataset_tweet.csv')
    if 'Tweet' in df.columns:
        # Simple whitespace tokenization for estimation
        token_lengths = df['Tweet'].astype(str).apply(lambda x: len(x.split()))
        
        plt.figure(figsize=(9, 6))
        sns.histplot(token_lengths, bins=50, kde=True, color='#4C72B0')
        plt.axvline(x=128, color='red', linestyle='--', linewidth=2, label='max_length = 128')
        
        # Calculate 99th percentile
        p99 = np.percentile(token_lengths, 99)
        plt.text(130, plt.ylim()[1]*0.8, f'Batas 128 Token\nmencakup >99% data\n(P99 = {p99:.0f} kata)', color='red', fontsize=11, bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
        
        plt.xlabel('Jumlah Kata (Token) per Tweet', fontsize=12)
        plt.ylabel('Frekuensi', fontsize=12)
        plt.title('Distribusi Panjang Tweet pada Dataset\n(Justifikasi Pemilihan max_length=128)', fontsize=14, pad=15)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(r'd:\www\hate-speech\image\token_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Token distribution generated.")
except Exception as e:
    print(f"Skipping token distribution due to error: {e}")

print("Done! All visualizations generated in d:\\www\\hate-speech\\image\\")
