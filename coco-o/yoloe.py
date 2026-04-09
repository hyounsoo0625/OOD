import argparse
import torch
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE 
from pycocotools.coco import COCO 
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
import random
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOE Domain Distance Visualization (All Data)")
    parser.add_argument("--model", type=str, default="yoloe-11l-seg.pt")
    parser.add_argument("--domains", type=str, nargs='+', default=["cartoon", "handmake", "painting", "sketch", "tattoo", "weather"])
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./analysis")
    parser.add_argument("--save_prefix", type=str, default="domain_compactness_all")
    return parser.parse_args()

def main(args):
    try:
        model = YOLOE(args.model)
    except Exception as e:
        print(f"[Error] Failed to load model '{args.model}': {e}")
        return

    visual_features = {}
    
    def savpe_hook_fn(module, input, output):
        visual_features['savpe_emb'] = output

    hook_attached = False
    for name, module in model.model.named_modules():
        if 'savpe' in module.__class__.__name__.lower():
            module.register_forward_hook(savpe_hook_fn)
            hook_attached = True
            break

    if not hook_attached:
        print("[Error] SAVPE module not found in the model.")
        return

    all_visual_embeddings = []
    labels_domain = []

    random.seed(args.seed)

    # 1. 입력받은 도메인들을 순회하며 모든 임베딩 추출
    for dom in tqdm(args.domains):
        ann_file = os.path.join(args.data_dir, "ood_coco", dom, "annotations", "instances_val2017.json")
        img_dir = os.path.join(args.data_dir, "ood_coco", dom, "val2017")
        
        print(f"[Info] Processing domain: {dom}")
        
        if not os.path.exists(ann_file):
            print(f"[Warning] Annotation file not found: {ann_file}")
            continue
            
        try:
            coco = COCO(ann_file)
        except Exception as e:
            print(f"[Error] Failed to load COCO annotations for {dom}: {e}")
            continue

        ann_ids = coco.getAnnIds()
        if not ann_ids: continue
        
        print(f"[Info] Found {len(ann_ids)} annotations in {dom}. Extracting features...")
        
        # 모든 Annotation ID에 대해 순회 (제한 없음)
        for ann_id in ann_ids:
            try:
                ann = coco.loadAnns(ann_id)[0]
                img_info = coco.loadImgs(ann['image_id'])[0]
                img_path = os.path.join(img_dir, img_info['file_name'])
            except Exception:
                continue
            
            if not os.path.exists(img_path): continue
            
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0: continue

            visual_prompts = dict(
                bboxes=np.array([[float(x), float(y), float(x + w), float(y + h)]], dtype=np.float32),
                cls=np.array([0], dtype=np.int32)
            )

            visual_features.clear()
            try:
                _ = model.predict(source=img_path, visual_prompts=visual_prompts, predictor=YOLOEVPSegPredictor, save=False, verbose=False)
            except Exception:
                continue
            
            if 'savpe_emb' in visual_features:
                savpe_out = visual_features['savpe_emb']
                if isinstance(savpe_out, tuple): savpe_out = savpe_out[0]
                
                v_vec = savpe_out.squeeze()
                if v_vec.numel() == 0 or v_vec.dim() != 1:
                    v_vec = v_vec.flatten()
                
                all_visual_embeddings.append(v_vec.detach().cpu().numpy())
                labels_domain.append(dom)

    if len(all_visual_embeddings) == 0:
        print("[Error] No visual embeddings were extracted.")
        return

    print(f"[Info] Total features extracted: {len(all_visual_embeddings)}")
    print("[Info] Running t-SNE and generating plots (This may take a while for large datasets)...")
    
    X_vis = np.array(all_visual_embeddings)
    perp_vis = max(1, min(50, len(X_vis) - 1)) # 데이터가 많아지므로 Perplexity 상한을 약간 높였습니다.
    tsne_vis = TSNE(n_components=2, metric='euclidean', perplexity=perp_vis, random_state=args.seed)
    vis_2d = tsne_vis.fit_transform(X_vis)

    df_vis = pd.DataFrame({'x': vis_2d[:, 0], 'y': vis_2d[:, 1], 'Category': labels_domain})
    
    # 2. 중심점(Centroid) 및 평균 거리 계산
    cluster_centers = df_vis.groupby('Category')[['x', 'y']].mean().reset_index()
    mean_distances = []
    
    for dom in cluster_centers['Category']:
        cx, cy = cluster_centers[cluster_centers['Category'] == dom][['x', 'y']].values[0]
        dom_points = df_vis[df_vis['Category'] == dom]
        distances = np.sqrt((dom_points['x'] - cx)**2 + (dom_points['y'] - cy)**2)
        mean_distances.append({'Category': dom, 'Mean Distance': distances.mean()})
        
    df_dist = pd.DataFrame(mean_distances)

    # 3. 그래프 시각화
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(20, 9), gridspec_kw={'width_ratios': [2, 1]})
    
    valid_domains = sorted(list(set(labels_domain)))
    palette = sns.color_palette("husl", len(valid_domains))
    color_map = dict(zip(valid_domains, palette))

    # [왼쪽] t-SNE Scatter Plot
    ax_tsne = axes[0]
    # 데이터 포인트가 많아지므로 점 크기(s)를 줄이고 투명도(alpha)를 조절하여 밀집도를 잘 보이게 함
    sns.scatterplot(x="x", y="y", hue="Category", palette=color_map, data=df_vis, ax=ax_tsne, s=30, alpha=0.5, edgecolor='none')
    
    for cat in valid_domains:
        cx, cy = cluster_centers[cluster_centers['Category'] == cat][['x', 'y']].values[0]
        cat_points = df_vis[df_vis['Category'] == cat]
        
        ax_tsne.scatter(cx, cy, marker='*', s=1200, facecolor=color_map[cat], edgecolor='black', linewidth=1.5, zorder=10)
        
        # 선이 너무 많아져 그래프를 가리는 것을 방지하기 위해 중심점 선의 투명도를 더 낮춤
        for _, row in cat_points.iterrows():
            ax_tsne.plot([cx, row['x']], [cy, row['y']], color=color_map[cat], linestyle='-', linewidth=0.3, alpha=0.05, zorder=2)

    ax_tsne.set_title("t-SNE: Features Distrubution by Domain (All Data)", fontsize=16, fontweight='bold')
    ax_tsne.legend(title="Domains", fontsize=12, title_fontsize=14)

    # [오른쪽] 도메인별 거리 비교 Bar Chart
    ax_bar = axes[1]
    sns.barplot(x="Category", y="Mean Distance", data=df_dist, palette=color_map, ax=ax_bar)
    
    ax_bar.set_title("Mean Distance to Domain Centroid", fontsize=16, fontweight='bold')
    ax_bar.set_ylabel("Average Euclidean Distance in Space", fontsize=12)
    ax_bar.set_xlabel("Domain Category", fontsize=12)
    
    for i, val in enumerate(df_dist['Mean Distance']):
        ax_bar.text(i, val + (val * 0.02), f"{val:.1f}", ha='center', va='bottom', fontsize=13, fontweight='bold')

    plt.tight_layout()
    
    # 저장
    save_filename = f"{args.save_prefix}.png"
    final_save_path = os.path.join(args.save_dir, save_filename)
    os.makedirs(args.save_dir, exist_ok=True)
    plt.savefig(final_save_path, dpi=300, bbox_inches='tight')
    print(f"[Success] Plot saved to {final_save_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)