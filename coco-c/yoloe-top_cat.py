import argparse
import torch
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycocotools.coco import COCO 
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
import random
from tqdm import tqdm

try:
    from imagecorruptions import corrupt, get_corruption_names
except ImportError:
    print("[Error] 'imagecorruptions' is not installed. Please run: pip install imagecorruptions")
    exit()

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOE Embeddings All Corruptions Evaluation")
    parser.add_argument("--model", type=str, default="yoloe-11l-seg.pt")
    parser.add_argument("--ann_file", type=str, default="../data/coco/annotations/instances_val2017.json")
    parser.add_argument("--img_dir", type=str, default="../data/coco/val2017")
    parser.add_argument("--samples_per_class", type=int, default=1, help="Number of images to sample PER CLASS")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./analysis")
    parser.add_argument("--save_prefix", type=str, default="all_corruptions_shift")
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
    
    # 평가할 15가지 모든 노이즈 가져오기
    corruptions_list = get_corruption_names()
    severities = [1, 2, 3, 4, 5]
    
    print(f"[Info] Evaluating 15 Corruptions across {len(all_cat_ids)} Categories...")
    
    # 각 카테고리별로 샘플링된 데이터 저장
    sampled_data = []
    
    for cat_id in all_cat_ids:
        cat_name = coco.loadCats(cat_id)[0]['name']
        ann_ids = coco.getAnnIds(catIds=cat_id)
        if not ann_ids: continue
        
        selected_ann_ids = random.sample(ann_ids, min(len(ann_ids), args.samples_per_class))
        for ann_id in selected_ann_ids:
            try:
                ann = coco.loadAnns(ann_id)[0]
                img_info = coco.loadImgs(ann['image_id'])[0]
                img_path = os.path.join(args.img_dir, img_info['file_name'])
                x, y, w, h = ann['bbox']
                if w > 0 and h > 0 and os.path.exists(img_path):
                    sampled_data.append({
                        'cat_name': cat_name, 'img_path': img_path,
                        'bbox': [float(x), float(y), float(x + w), float(y + h)]
                    })
            except Exception:
                continue

    # 추론 함수 (중복 코드 최소화)
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

    # Step 1: 카테고리별 Clean 임베딩 추출 및 중심점(Centroid) 계산
    print("[Info] Step 1: Calculating Clean Centroids...")
    clean_embeddings_by_cat = {}
    
    for data in sampled_data:
        img_bgr = cv2.imread(data['img_path'])
        if img_bgr is None: continue
        
        emb = extract_embedding(img_bgr, data['bbox'])
        if emb is not None:
            if data['cat_name'] not in clean_embeddings_by_cat:
                clean_embeddings_by_cat[data['cat_name']] = []
            clean_embeddings_by_cat[data['cat_name']].append(emb)
            
    clean_centroids = {}
    for cat, embs in clean_embeddings_by_cat.items():
        clean_centroids[cat] = np.mean(embs, axis=0)

    # Step 2: 모든 노이즈, 강도에 대한 추론 및 거리 계산
    print(f"[Info] Step 2: Applying {len(corruptions_list)} Corruptions (This will take a while)...")
    results = []
    
    total_images = len(sampled_data)
    for idx, data in tqdm(enumerate(sampled_data)):
        if idx % 10 == 0:
            print(f"       Processing image {idx}/{total_images}")
            
        cat = data['cat_name']
        if cat not in clean_centroids: continue
        cx = clean_centroids[cat]
        
        img_bgr = cv2.imread(data['img_path'])
        if img_bgr is None: continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        for corruption in corruptions_list:
            for sev in severities:
                try:
                    img_corrupted_rgb = corrupt(img_rgb, corruption_name=corruption, severity=sev)
                    img_corrupted_bgr = cv2.cvtColor(img_corrupted_rgb, cv2.COLOR_RGB2BGR)
                    
                    emb = extract_embedding(img_corrupted_bgr, data['bbox'])
                    if emb is not None:
                        # Clean 중심점과의 유클리드 거리 계산
                        dist = np.linalg.norm(emb - cx)
                        results.append({
                            'Category': cat,
                            'Corruption': corruption,
                            'Severity': sev,
                            'Distance': dist
                        })
                except Exception:
                    continue

    if not results:
        print("[Error] No evaluation results were generated.")
        return

    df_results = pd.DataFrame(results)
    corruption_mapping = {
            # Noise
            'gaussian_noise': 'Noise', 'shot_noise': 'Noise', 'impulse_noise': 'Noise', 'speckle_noise': 'Noise',
            # Blur
            'defocus_blur': 'Blur', 'glass_blur': 'Blur', 'motion_blur': 'Blur', 'zoom_blur': 'Blur', 'gaussian_blur': 'Blur',
            # Weather
            'snow': 'Weather', 'frost': 'Weather', 'fog': 'Weather', 'brightness': 'Weather', 'spatter': 'Weather',
            # Digital
            'contrast': 'Digital', 'elastic_transform': 'Digital', 'pixelate': 'Digital', 'jpeg_compression': 'Digital', 'saturate': 'Digital'
        }
    
    # 데이터프레임에 새로운 'Corruption_Group' 열 추가 (매핑에 없는 경우 'Other'로 분류)
    df_results['Corruption_Group'] = df_results['Corruption'].map(corruption_mapping).fillna('Other')

    # Step 3: 시각화 (그룹화 적용)
    print("[Info] Generating grouped plots...")
    sns.set_theme(style="whitegrid")
    # 카테고리가 줄었으므로 그래프 세로 크기를 살짝 줄이면 비율이 더 좋습니다.
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [1.2, 1]})
    
    # [왼쪽] Line Plot: 큰 카테고리(Group)에 따른 이탈 거리 추세
    ax_line = axes[0]
    # hue를 'Corruption_Group'으로 변경, 색상 팔레트 단순화
    sns.lineplot(data=df_results, x="Severity", y="Distance", hue="Corruption_Group", 
                 marker="o", linewidth=3, markersize=10, ax=ax_line, palette="tab10")
    
    ax_line.set_title("Feature Shift by Corruption Group", fontsize=18, fontweight='bold')
    ax_line.set_ylabel("Mean Distance from Clean Centroid", fontsize=14)
    ax_line.set_xlabel("Severity Level", fontsize=14)
    ax_line.set_xticks(severities)
    # 범례 위치를 깔끔하게 그래프 안쪽이나 우측 상단으로 조정
    ax_line.legend(title="Corruption Groups", fontsize=12, title_fontsize=14)

    # [오른쪽] Heatmap: 그룹별, 강도별 평균 이탈 거리를 한눈에 파악
    ax_heat = axes[1]
    # 데이터를 피벗 테이블로 변환 시 index를 'Corruption_Group'으로 설정
    # 각 노이즈 그룹과 강도에 해당하는 거리 값들의 평균(mean)이 자동으로 계산됩니다.
    heatmap_data = df_results.pivot_table(index='Corruption_Group', columns='Severity', values='Distance', aggfunc='mean')
    
    # 거리가 멀수록(취약할수록) 붉은색으로 표시
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=.5, ax=ax_heat, cbar_kws={'label': 'Mean Distance'})
    ax_heat.set_title("Heatmap: Vulnerability by Corruption Group", fontsize=18, fontweight='bold')
    ax_heat.set_ylabel("Corruption Group", fontsize=14)
    ax_heat.set_xlabel("Severity Level", fontsize=14)

    plt.tight_layout()
    
    # 저장 파일명도 변경 (선택사항)
    save_filename = f"{args.save_prefix}_grouped.png"
    final_save_path = os.path.join(args.save_dir, save_filename)
    os.makedirs(args.save_dir, exist_ok=True)
    plt.savefig(final_save_path, dpi=300, bbox_inches='tight')
    print(f"[Success] Grouped Corruptions evaluation plot saved to {final_save_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)