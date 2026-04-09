import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# 학술 논문용 matplotlib 세팅
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

with open("./analysis/cococ_features.pkl", "rb") as f:
    data = pickle.load(f)

corruptions = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 'shot_noise', 'snow', 'zoom_blur']
severities = ['1', '2', '3', '4', '5']
colors = [
    '#e6194b', # Red
    '#3cb44b', # Green
    '#4363d8', # Blue
    '#f58231', # Orange
    '#911eb4', # Purple
    '#46f0f0', # Cyan
    '#f032e6', # Magenta
    '#bcf60c', # Lime
    '#fabebe', # Pink
    '#008080', # Teal
    '#e6beff', # Lavender
    '#9a6324', # Brown
    '#800000', # Maroon
    '#aaffc3', # Mint
    '#808000'  # Olive
]

fig, ax = plt.subplots(figsize=(18, 16))

for idx, corr in enumerate(corruptions):
    means, stds = [], []
    for sev in severities:
        sims = []
        for ann_id, v in data.items():
            if v['clean'] is not None and sev in v['corrupted'][corr] and v['corrupted'][corr][sev] is not None:
                clean_emb = v['clean'].reshape(1, -1)
                corr_emb = v['corrupted'][corr][sev].reshape(1, -1)
                sim = cosine_similarity(clean_emb, corr_emb)[0][0]
                sims.append(sim)
        
        means.append(np.mean(sims) if sims else 0)
        stds.append(np.std(sims) if sims else 0)
        
    # 에러바가 포함된 Line Plot
    ax.errorbar(severities, means, yerr=stds, label=corr.replace('_', ' ').title(), 
                color=colors[idx], marker='o', capsize=5, linewidth=2, markersize=8)

ax.set_xlabel('Corruption Severity')
ax.set_ylabel('Cosine Similarity to Clean Prototype')
ax.set_title('Feature Robustness against Corruptions')
ax.set_ylim(0.0, 1.05)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='lower left')

plt.tight_layout()
plt.savefig("./analysis/fig1_severity_drop.png", dpi=300)
print("Saved fig1")