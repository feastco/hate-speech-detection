import matplotlib.pyplot as plt
import numpy as np
import os

# Data for the plot
models = ['MNB', 'CNB', 'SVM Baseline\n(LinearSVC)', 'SVM Final\n(SGD+CSL)']
cv_means = [81.62, 81.05, 80.21, 79.39]
cv_stds = [1.00, 0.89, 0.79, 0.88]

# Colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

fig, ax = plt.subplots(figsize=(9, 6))

# Create bar chart with error bars
bars = ax.bar(models, cv_means, yerr=cv_stds, capsize=8, 
              color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

# Add values on top of bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
            f'{height:.2f}%\n±{cv_stds[i]:.2f}',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

# Customize plot
ax.set_ylim(70, 85)
ax.set_ylabel('10-Fold CV Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Stabilitas Model Klasikal - 10-Fold Cross Validation\n(Error bar menunjukkan ±1 Standar Deviasi)', 
             fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Make axes bold
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

plt.tight_layout()

# Save the figure
save_path = 'D:/www/hate-speech/image/cv_stability_chart_new.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {save_path}")
