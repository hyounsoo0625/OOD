import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from pycocotools.coco import COCO 
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
import random

# ==========================================
# 1. 시각화 함수 정의 (Figures)
# ==========================================

def plot_correlation_scatter(df_vis, target_class, save_dir='./analysis'):
    """Figure 1: 객체 크기(Area)와 중심점 거리 간의 상관관계 (Scatter Plot)"""
    df_subset = df_vis[df_vis['Category'] == target_class].copy()
    if len(df_subset) < 5: return
    
    df_subset['Log_Bbox_Area'] = np.log1p(df_subset['Bbox_Area'])
    corr, p_value = pearsonr(df_subset['Log_Bbox_Area'], df_subset['Distance_to_Centroid'])
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.regplot(
        data=df_subset, x='Log_Bbox_Area', y='Distance_to_Centroid',
        scatter_kws={'alpha':0.6, 's':40, 'edgecolor':'w'}, 
        line_kws={'color':'red', 'linewidth':2}, ax=ax
    )
    
    textstr = f'Pearson r: {corr:.3f}\np-value: {p_value:.2e}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    ax.set_title(f'Correlation: Bbox Area vs. Variance ({target_class.capitalize()})', fontweight='bold')
    ax.set_xlabel('Log(Bounding Box Area)')
    ax.set_ylabel('Cosine Distance to Class Centroid')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'fig1_correlation_{target_class}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_variance_distribution(df_vis, save_dir='./analysis'):
    """Figure 2: 객체 크기 그룹(Small, Medium, Large)에 따른 분산 비교 (Violin Plot)"""
    def categorize_size(area):
        if area < 32 * 32: return 'Small'
        elif area < 96 * 96: return 'Medium'
        else: return 'Large'
            
    df_vis['Size_Category'] = df_vis['Bbox_Area'].apply(categorize_size)
    order = ['Small', 'Medium', 'Large']
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.violinplot(data=df_vis, x='Size_Category', y='Distance_to_Centroid', order=order, inner="quartile", palette="pastel", ax=ax)
    sns.stripplot(data=df_vis, x='Size_Category', y='Distance_to_Centroid', order=order, color='black', alpha=0.3, size=3, jitter=True, ax=ax)
    
    ax.set_title('Embedding Variance by Object Size (All Classes)', fontweight='bold')
    ax.set_xlabel('Object Size Category (COCO standard)')
    ax.set_ylabel('Cosine Distance to Class Centroid')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig2_variance_violin.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_inliers_vs_outliers(df_vis, target_class, top_k=5, save_dir='./analysis'):
    """Figure 3: 거리가 가까운 샘플(Inlier) vs 먼 샘플(Outlier) 시각적 비교"""
    df_subset = df_vis[df_vis['Category'] == target_class].sort_values(by='Distance_to_Centroid')
    if len(df_subset) < top_k * 2: return

    inliers = df_subset.head(top_k)
    outliers = df_subset.tail(top_k)
    
    fig, axes = plt.subplots(2, top_k, figsize=(top_k * 2.5, 5.5))
    fig.suptitle(f'Variance Drivers for Class: "{target_class.capitalize()}"', fontsize=16, fontweight='bold', y=1.05)
    
    for i, (_, row) in enumerate(inliers.iterrows()):
        axes[0, i].imshow(row['Thumbnail'])
        axes[0, i].axis('off')
        title = f"Inliers (Low Variance)\nDist: {row['Distance_to_Centroid']:.2f}" if i == 0 else f"Dist: {row['Distance_to_Centroid']:.2f}"
        axes[0, i].set_title(title, fontsize=11)

    for i, (_, row) in enumerate(outliers.iterrows()):
        axes[1, i].imshow(row['Thumbnail'])
        axes[1, i].axis('off')
        title = f"Outliers (High Variance)\nDist: {row['Distance_to_Centroid']:.2f}" if i == 0 else f"Dist: {row['Distance_to_Centroid']:.2f}"
        axes[1, i].set_title(title, fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'fig3_inliers_outliers_{target_class}.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# 2. 메인 파이프라인 (추출 및 분석)
# ==========================================

def main():
    save_dir = "./analysis"
    os.makedirs(save_dir, exist_ok=True)
    
    # 모델 로드 및 설정
    print("Loading YOLOe model...")
    model = YOLOE("yoloe-11l-seg.pt")
    target_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'truck', 'bench', 'backpack', 'handbag', 'chair']
    num_samples = 200 
    
    # 🔹 너무 작은 Bounding Box를 걸러내기 위한 최소 면적 기준 (COCO small 객체 기준인 32x32 사용)
    MIN_BBOX_AREA = 32 * 32 

    # Hook 등록 (SAVPE 모듈의 출력 추출)
    visual_features = {}
    def savpe_hook_fn(module, input, output):
        visual_features['savpe_emb'] = output

    hook_attached = False
    for name, module in model.model.named_modules():
        if module.__class__.__name__ == 'SAVPE':
            module.register_forward_hook(savpe_hook_fn)
            hook_attached = True
            break
            
    if not hook_attached:
        print("Error: SAVPE module not found in the model.")
        return

    # COCO 데이터셋 로드
    data_dir = "../data/coco"
    ann_file = f"{data_dir}/annotations/instances_val2017.json"
    img_dir = f"{data_dir}/val2017"
    
    try:
        coco = COCO(ann_file)
    except Exception as e:
        print(f"Error loading COCO annotations: {e}")
        return

    # 데이터 저장을 위한 리스트
    extracted_data = []
    random.seed(42)

    # 1단계: 임베딩 및 메타데이터 추출
    for cls in target_classes:
        print(f"\nProcessing class: {cls}")
        catIds = coco.getCatIds(catNms=[cls])
        if not catIds: continue
        annIds = coco.getAnnIds(catIds=catIds)
        if not annIds: continue
        
        selected_annIds = random.sample(annIds, min(len(annIds), num_samples * 5))
        anns = coco.loadAnns(selected_annIds)
        
        count = 0
        for ann in tqdm(anns, desc=f"Extracting {cls}"):
            if count >= num_samples: break
            
            try:
                img_info = coco.loadImgs(ann['image_id'])[0]
                img_path = os.path.join(img_dir, img_info['file_name'])
                if not os.path.exists(img_path): continue
                
                x, y, w, h = ann['bbox']
                
                # 🔹 필터링 조건 추가: Bbox 면적이 MIN_BBOX_AREA보다 작으면 스킵
                bbox_area = w * h
                if bbox_area < MIN_BBOX_AREA:
                    continue
                
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = int(x + w), int(y + h)
                if x2 <= x1 or y2 <= y1: continue

                # 메타데이터 계산 (종횡비)
                aspect_ratio = w / h if h > 0 else 0

                visual_prompts = dict(
                    bboxes=np.array([[float(x), float(y), float(x + w), float(y + h)]], dtype=np.float32),
                    cls=np.array([target_classes.index(cls)], dtype=np.int32)
                )

                visual_features.clear()
                # Inference 실행
                _ = model.predict(source=img_path, visual_prompts=visual_prompts, predictor=YOLOEVPSegPredictor, save=False, verbose=False)
                
                if 'savpe_emb' in visual_features:
                    savpe_out = visual_features['savpe_emb']
                    if isinstance(savpe_out, tuple): savpe_out = savpe_out[0]
                    
                    v_vec = savpe_out.squeeze()
                    if v_vec.numel() == 0 or v_vec.dim() != 1: continue
                    
                    # L2 정규화
                    v_vec = F.normalize(v_vec, dim=0, p=2)
                    emb_array = v_vec.detach().cpu().numpy()
                    
                    # Thumbnail 추출
                    img_bgr = cv2.imread(img_path)
                    if img_bgr is not None:
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        patch = img_rgb[y1:y2, x1:x2]
                        patch_resized = cv2.resize(patch, (80, 80)) # 시각화를 위해 약간 크게 리사이즈
                        
                        extracted_data.append({
                            'Category': cls,
                            'High_Dim_Emb': emb_array,
                            'Bbox_Area': bbox_area,
                            'Aspect_Ratio': aspect_ratio,
                            'Thumbnail': patch_resized
                        })
                        count += 1
            except Exception:
                continue

    if len(extracted_data) == 0:
        print("Error: No valid data extracted.")
        return

    # DataFrame 생성
    df_vis = pd.DataFrame(extracted_data)

    # 2단계: 고차원 공간 중심점(Centroid) 계산 및 거리 도출
    print("\nCalculating Distances to Class Centroids...")
    df_vis['Distance_to_Centroid'] = 0.0
    valid_classes = df_vis['Category'].unique()
    
    for cat in valid_classes:
        cat_mask = df_vis['Category'] == cat
        # 해당 클래스의 전체 임베딩 가져오기
        embeddings = np.vstack(df_vis.loc[cat_mask, 'High_Dim_Emb'].values)
        # 중심점 계산
        centroid = np.mean(embeddings, axis=0)
        # 중심점과의 코사인 거리 계산
        df_vis.loc[cat_mask, 'Distance_to_Centroid'] = [cosine(emb, centroid) for emb in embeddings]

    # 3단계: 분석 플롯 생성 (논문용 증명)
    print("\nGenerating Analysis Figures...")
    
    # 전체 분포 비교 (Violin Plot)
    plot_variance_distribution(df_vis, save_dir)
    print(" - Figure 2 (Variance Violin Plot) Created.")
    
    # 각 클래스별 상관관계 및 정성 분석 시각화
    for cls in valid_classes:
        plot_correlation_scatter(df_vis, target_class=cls, save_dir=save_dir)
        plot_inliers_vs_outliers(df_vis, target_class=cls, top_k=5, save_dir=save_dir)
        print(f" - Figures 1 & 3 for '{cls}' Created.")

    print(f"\nAll processes completed successfully! Check the '{save_dir}' folder for the figures.")

if __name__ == "__main__":
    main()