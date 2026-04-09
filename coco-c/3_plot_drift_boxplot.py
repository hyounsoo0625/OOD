import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

with open("./analysis/cococ_features.pkl", "rb") as f:
    data = pickle.load(f)

corruptions = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 'shot_noise', 'snow', 'zoom_blur']
severity = '3' # 중간 강도 기준

plot_data = []

for corr in corruptions:
    for ann_id, v in data.items():
        if v['clean'] is not None and severity in v['corrupted'][corr] and v['corrupted'][corr][severity] is not None:
            clean_emb = v['clean'].reshape(1, -1)
            corr_emb = v['corrupted'][corr][severity].reshape(1, -1)
            sim = cosine_similarity(clean_emb, corr_emb)[0][0]
            plot_data.append({'Corruption': corr.replace('_', '\n').title(), 'Similarity': sim})

df = pd.DataFrame(plot_data)

fig, ax = plt.subplots(figsize=(18, 16))
sns.boxplot(x='Corruption', y='Similarity', data=df, palette="Set2", width=0.6, ax=ax, showfliers=False)
sns.stripplot(x='Corruption', y='Similarity', data=df, color=".25", alpha=0.3, size=3, ax=ax)

ax.set_ylabel('Similarity to Clean Anchor')
ax.set_xlabel('')
ax.set_title(f'Semantic Drift Distribution (Severity {severity})')
ax.set_ylim(0.2, 1.0)
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("./analysis/fig3_drift_boxplot.png", dpi=300)
print("Saved fig3")