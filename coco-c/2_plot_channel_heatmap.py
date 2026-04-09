import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

with open("./analysis/cococ_features.pkl", "rb") as f:
    data = pickle.load(f)

corruptions = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 'shot_noise', 'snow', 'zoom_blur']
severity = '5' # 가장 극한의 상황(5)에서 채널 민감도 비교

channel_stds = []

for corr in corruptions:
    diffs = []
    for ann_id, v in data.items():
        if v['clean'] is not None and severity in v['corrupted'][corr] and v['corrupted'][corr][severity] is not None:
            # Clean과 Corrupted의 채널별 절대적 차이 계산
            diff = np.abs(v['clean'] - v['corrupted'][corr][severity])
            diffs.append(diff)
    
    if diffs:
        # 해당 손상에 대한 채널별 평균 변동폭
        channel_stds.append(np.mean(diffs, axis=0))

heatmap_data = np.array(channel_stds)

# 상위 50개 채널만 시각화 (가독성을 위함)
heatmap_data = heatmap_data[:, :50] 

fig, ax = plt.subplots(figsize=(12, 14))
sns.heatmap(heatmap_data, cmap="YlOrRd", cbar_kws={'label': 'Mean Absolute Deviation'}, ax=ax)

ax.set_yticklabels([c.replace('_', ' ').title() for c in corruptions], rotation=0, fontsize=12)
ax.set_xlabel('Embedding Dimension / Channels (Top 50)')
ax.set_title('Channel Sensitivity Map (Severity 5)')

plt.tight_layout()
plt.savefig("./analysis/fig2_channel_heatmap.png", dpi=300)
print("Saved fig2")