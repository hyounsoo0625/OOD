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
    parser = argparse.ArgumentParser(description="YOLOE Embedding Compactness by BBox Size")
    parser.add_argument("--model", type=str, default="yoloe-11l-seg.pt")
    parser.add_argument("--ann_file", type=str, default="../data/coco/annotations/instances_val2017.json")
    parser.add_argument("--img_dir", type=str, default="../data/coco/val2017")
    parser.add_argument("--samples_per_class", type=int, default=100, help="Number of images to sample PER CLASS (0 for ALL)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./analysis")
    parser.add_argument("--save_prefix", type=str, default="compactness_by_size")
    return parser.parse_args()

def get_size_category(area):
    """COCO 공식 기준에 따른 객체 크기 분류"""
    if area < 32 * 32:
        return 'Small'
    elif area < 96 * 96:
        return 'Medium'
    else:
        return 'Large'

def main(args):
    model = YOLOE(args.model)
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

    try:
        coco = COCO(args.ann_file)
    except Exception as e:
        print(f"[Error] Failed to load COCO annotations: {e}")
        return

    all_cat_ids = coco.getCatIds()
    if not all_cat_ids:
        print("[Error] No categories found in dataset.")
        return

    random.seed(args.seed)
    print(f"[Info] Evaluating Compactness across {len(all_cat_ids)} Categories by BBox Size...")
    
    # 1. 데이터 샘플링
    sampled_data = []
    
    for cat_id in all_cat_ids:
        cat_name = coco.loadCats(cat_id)[0]['name']
        ann_ids = coco.getAnnIds(catIds=cat_id)
        if not ann_ids: continue
        
        target_ann_ids = random.sample(ann_ids, min(len(ann_ids), args.samples_per_class)) if args.samples_per_class > 0 else ann_ids
            
        for ann_id in target_ann_ids:
            try:
                ann = coco.loadAnns(ann_id)[0]
                img_info = coco.loadImgs(ann['image_id'])[0]
                img_path = os.path.join(args.img_dir, img_info['file_name'])
                x, y, w, h = ann['bbox']
                
                if w > 0 and h > 0 and os.path.exists(img_path):
                    area = w * h
                    sampled_data.append({
                        'cat_name': cat_name, 
                        'img_path': img_path,
                        'bbox': [float(x), float(y), float(x + w), float(y + h)],
                        'size': get_size_category(area)
                    })
            except Exception:
                continue

    # 추론 함수
    def extract_embedding(img_bgr, bbox):
        visual_prompts = dict(
            bboxes=np.array([bbox], dtype=np.float32),
            cls=np.array([0], dtype=np.int32)
        )
        visual_features.clear()
        try:
            _ = model.predict(source=img_bgr, visual_prompts=visual_prompts, predictor=YOLOEVPSegPredictor, save=False, verbose=False)
            if 'savpe_emb' in visual_features:
                savpe_out = visual_features['savpe_emb']
                if isinstance(savpe_out, tuple): savpe_out = savpe_out[0]
                v_vec = savpe_out.squeeze()
                if v_vec.numel() == 0 or v_vec.dim() != 1:
                    v_vec = v_vec.flatten()
                return v_vec.detach().cpu().numpy()
        except Exception:
            pass
        return None

    # 2. 특징(Embedding) 추출
    print(f"[Info] Extracting features for {len(sampled_data)} images...")
    extracted_features = []
    
    for idx, data in tqdm(enumerate(sampled_data), total=len(sampled_data), desc="Processing"):
        img_bgr = cv2.imread(data['img_path'])
        if img_bgr is None: continue
        
        emb = extract_embedding(img_bgr, data['bbox'])
        if emb is not None:
            extracted_features.append({
                'Category': data['cat_name'],
                'Size': data['size'],
                'emb': emb
            })

    if not extracted_features:
        print("[Error] No features extracted.")
        return

    # 3. 카테고리별 중심점(Centroid) 계산 및 거리 측정
    print("[Info] Calculating Centroids and Euclidean Distances...")
    
    # 카테고리별로 묶어서 중심점 구하기
    embeddings_by_cat = {}
    for item in extracted_features:
        cat = item['Category']
        if cat not in embeddings_by_cat:
            embeddings_by_cat[cat] = []
        embeddings_by_cat[cat].append(item['emb'])
        
    centroids = {cat: np.mean(embs, axis=0) for cat, embs in embeddings_by_cat.items()}

    # 각 샘플과 자신이 속한 카테고리 중심점 사이의 유클리디안 거리 계산
    for item in extracted_features:
        cat = item['Category']
        cx = centroids[cat]
        # 유클리디안 거리 연산
        item['Distance'] = np.linalg.norm(item['emb'] - cx)

    df_results = pd.DataFrame(extracted_features)

    # 4. t-SNE 차원 축소 (시각화 용도)
    print("[Info] Running t-SNE for visualization...")
    X = np.stack(df_results['emb'].values)
    perp = max(1, min(30, len(X) - 1))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=args.seed)
    X_2d = tsne.fit_transform(X)
    
    df_results['tsne_1'] = X_2d[:, 0]
    df_results['tsne_2'] = X_2d[:, 1]

    # 5. 시각화 (t-SNE 산점도 & Boxplot)
    print("[Info] Generating plots...")
    sns.set_theme(style="whitegrid")
    
    size_order = ['Small', 'Medium', 'Large']
    df_results['Size'] = pd.Categorical(df_results['Size'], categories=size_order, ordered=True)
    size_palette = {"Small": "#e74c3c", "Medium": "#f39c12", "Large": "#2ecc71"}

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [1, 1]})
    
    # [왼쪽] Boxplot (크기별 중심점 이탈 거리 분포)
    ax_box = axes[0]
    sns.boxplot(data=df_results, x="Size", y="Distance", palette=size_palette, ax=ax_box, showfliers=False)
    # 점들을 겹쳐 그려서 실제 데이터 분포를 보여줌
    sns.stripplot(data=df_results, x="Size", y="Distance", color=".25", alpha=0.3, size=3, ax=ax_box)
    
    ax_box.set_title("Embedding Compactness by BBox Size", fontsize=16, fontweight='bold')
    ax_box.set_ylabel("Euclidean Distance to Category Centroid", fontsize=14)
    ax_box.set_xlabel("Object Size", fontsize=14)

    # 평균값 텍스트 추가
    mean_dists = df_results.groupby('Size', observed=True)['Distance'].mean()
    for i, size_label in enumerate(size_order):
        if size_label in mean_dists:
            mean_val = mean_dists[size_label]
            ax_box.text(i, ax_box.get_ylim()[1]*0.95, f"Mean: {mean_val:.2f}", 
                        ha='center', va='top', fontsize=12, fontweight='bold', 
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor=size_palette[size_label]))

    # [오른쪽] t-SNE Scatter Plot (어떤 크기가 공간에서 더 뭉쳐있고 퍼져있는지 시각화)
    ax_tsne = axes[1]
    sns.scatterplot(data=df_results, x="tsne_1", y="tsne_2", hue="Size", 
                    palette=size_palette, s=50, alpha=0.6, edgecolor='white', ax=ax_tsne)
    
    ax_tsne.set_title("t-SNE Space: Feature Distribution by Size", fontsize=16, fontweight='bold')
    ax_tsne.legend(title="Object Size", fontsize=12)

    plt.tight_layout()
    
    os.makedirs(args.save_dir, exist_ok=True)
    save_filename = f"{args.save_prefix}.png"
    final_save_path = os.path.join(args.save_dir, save_filename)
    plt.savefig(final_save_path, dpi=300, bbox_inches='tight')
    
    # 데이터프레임 저장시 임베딩 원본 배열은 제거하고 저장
    df_save = df_results.drop(columns=['emb'])
    csv_path = os.path.join(args.save_dir, f"{args.save_prefix}.csv")
    df_save.to_csv(csv_path, index=False)
    
    print(f"[Success] Plot saved to {final_save_path}")
    print(f"[Success] Data saved to {csv_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)