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
    parser = argparse.ArgumentParser(description="YOLOE Embedding Compactness by Continuous BBox Size")
    parser.add_argument("--model", type=str, default="yoloe-11l-seg.pt")
    parser.add_argument("--ann_file", type=str, default="../data/coco/annotations/instances_val2017.json")
    parser.add_argument("--img_dir", type=str, default="../data/coco/val2017")
    parser.add_argument("--samples_per_class", type=int, default=100, help="Number of images to sample PER CLASS (0 for ALL)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./analysis")
    parser.add_argument("--save_prefix", type=str, default="compactness_by_continuous_size")
    return parser.parse_args()

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
    print(f"[Info] Evaluating Compactness across {len(all_cat_ids)} Categories by Continuous BBox Size...")
    
    # 1. 데이터 샘플링 (연속적인 Area 값 저장)
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
                        'area': area # 카테고리 대신 실제 면적 값 저장
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
                'Area': data['area'],
                'emb': emb
            })

    if not extracted_features:
        print("[Error] No features extracted.")
        return

    # 3. 카테고리별 중심점(Centroid) 계산 및 거리 측정
    print("[Info] Calculating Centroids and Euclidean Distances...")
    
    embeddings_by_cat = {}
    for item in extracted_features:
        cat = item['Category']
        if cat not in embeddings_by_cat:
            embeddings_by_cat[cat] = []
        embeddings_by_cat[cat].append(item['emb'])
        
    centroids = {cat: np.mean(embs, axis=0) for cat, embs in embeddings_by_cat.items()}

    for item in extracted_features:
        cat = item['Category']
        cx = centroids[cat]
        item['Distance'] = np.linalg.norm(item['emb'] - cx)

    df_results = pd.DataFrame(extracted_features)
    
    # 변동성이 큰 Area 값을 처리하기 위해 Log 변환 열 추가
    df_results['Log_Area'] = np.log10(df_results['Area'])

    # 4. t-SNE 차원 축소
    print("[Info] Running t-SNE for visualization...")
    X = np.stack(df_results['emb'].values)
    perp = max(1, min(30, len(X) - 1))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=args.seed)
    X_2d = tsne.fit_transform(X)
    
    df_results['tsne_1'] = X_2d[:, 0]
    df_results['tsne_2'] = X_2d[:, 1]

    # 5. 시각화 (연속형 산점도 & t-SNE)
    print("[Info] Generating continuous plots...")
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [1, 1.1]})
    
    # [왼쪽] Regression Plot (Area vs Distance)
    ax_reg = axes[0]
    # scatter 속성과 선 속성을 분리하여 시각적 명확성을 높임
    sns.regplot(data=df_results, x="Log_Area", y="Distance", ax=ax_reg,
                scatter_kws={"alpha": 0.3, "s": 20, "color": "#3498db"},
                line_kws={"color": "#e74c3c", "linewidth": 3})
    
    ax_reg.set_title("Embedding Distance vs Continuous BBox Area", fontsize=16, fontweight='bold')
    ax_reg.set_ylabel("Euclidean Distance to Category Centroid", fontsize=14)
    ax_reg.set_xlabel("Log10(Object Area)", fontsize=14)

    # 상관계수 계산 및 표기
    corr = df_results['Log_Area'].corr(df_results['Distance'])
    ax_reg.text(0.05, 0.95, f"Pearson Corr: {corr:.3f}", transform=ax_reg.transAxes, 
                fontsize=13, fontweight='bold', va='top', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # [오른쪽] t-SNE Scatter Plot (연속적인 크기 변화 시각화)
    ax_tsne = axes[1]
    # hue를 연속형 변수인 Log_Area로 설정하고, 팔레트를 viridis로 사용
    scatter = sns.scatterplot(data=df_results, x="tsne_1", y="tsne_2", hue="Log_Area", 
                              palette="viridis", s=40, alpha=0.7, edgecolor=None, ax=ax_tsne)
    
    ax_tsne.set_title("t-SNE Space: Feature Distribution by Area", fontsize=16, fontweight='bold')
    
    # 기존 범례를 제거하고 연속형 컬러바(Colorbar)로 교체
    ax_tsne.get_legend().remove()
    norm = plt.Normalize(df_results['Log_Area'].min(), df_results['Log_Area'].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_tsne, pad=0.02)
    cbar.set_label("Log10(Object Area)", fontsize=12, rotation=270, labelpad=15)

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