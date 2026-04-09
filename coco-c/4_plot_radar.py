import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

with open("./analysis/cococ_features.pkl", "rb") as f:
    data = pickle.load(f)

corruptions = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 'shot_noise', 'snow', 'zoom_blur']
severity = '4'

scores = []
for corr in corruptions:
    sims = []
    for ann_id, v in data.items():
        if v['clean'] is not None and severity in v['corrupted'][corr] and v['corrupted'][corr][severity] is not None:
            sim = cosine_similarity(v['clean'].reshape(1, -1), v['corrupted'][corr][severity].reshape(1, -1))[0][0]
            sims.append(sim)
    scores.append(np.mean(sims) if sims else 0)

# 레이더 차트 세팅
angles = np.linspace(0, 2 * np.pi, len(corruptions), endpoint=False).tolist()
scores += scores[:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(17, 17), subplot_kw=dict(polar=True))
ax.plot(angles, scores, color='#1f77b4', linewidth=2, linestyle='solid', label='Original YOLOE')
ax.fill(angles, scores, color='#1f77b4', alpha=0.25)

# 축 설정
labels = [c.replace('_', ' ').title() for c in corruptions]
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)

ax.set_title('Robustness Area across Corruptions (Severity 4)', size=15, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.savefig("./analysis/fig4_radar_chart.png", dpi=300)
print("Saved fig4")