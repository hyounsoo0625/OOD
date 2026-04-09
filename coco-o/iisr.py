import argparse
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from pycocotools.coco import COCO 
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
import random
from tqdm import tqdm

# ==========================================
# 1. IISR 계산 및 시각화 함수 정의
# ==========================================
def calculate_similarities_and_iisr(features, labels):
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    intra_sims_all = [] 
    intra_means_per_class = [] 
    prototypes = []
    
    print("  -> Calculating Intra-Similarities and Prototypes...")
    for label in tqdm(unique_labels, leave=False):
        mask = (np.array(labels) == label)
        class_features = features[mask]
        
        if len(class_features) == 0:
            continue
            
        prototype = class_features.mean(axis=0)
        prototypes.append(prototype)
        
        n_c = len(class_features)
        if n_c > 1:
            sim_matrix = cosine_similarity(class_features)
            idx = np.triu_indices(n_c, k=1)
            sims = sim_matrix[idx]
            
            intra_sims_all.extend(sims)
            intra_means_per_class.append(sims.mean())

    numerator_iisr = np.mean(intra_means_per_class) if intra_means_per_class else 0.0
    
    print("  -> Calculating Inter-Similarities...")
    inter_sims_all = [] 
    
    for i in tqdm(range(num_classes), leave=False):
        mask_i = (np.array(labels) == unique_labels[i])
        feats_i = features[mask_i]
        if len(feats_i) == 0: continue
            
        for j in range(i + 1, num_classes):
            mask_j = (np.array(labels) == unique_labels[j])
            feats_j = features[mask_j]
            if len(feats_j) == 0: continue
            
            sim_matrix_ij = cosine_similarity(feats_i, feats_j)
            inter_sims_all.extend(sim_matrix_ij.flatten())

    if len(prototypes) > 1:
        proto_matrix = np.array(prototypes)
        sim_matrix_proto = cosine_similarity(proto_matrix)
        idx_proto = np.triu_indices(len(prototypes), k=1)
        inter_proto_sims = sim_matrix_proto[idx_proto]
        denominator_iisr = inter_proto_sims.mean()
    else:
        denominator_iisr = 1.0

    iisr_value = numerator_iisr / denominator_iisr if denominator_iisr != 0 else 0.0
    return np.array(intra_sims_all), np.array(inter_sims_all), iisr_value

def plot_dual_histogram(intra_sims, inter_sims, title, save_path):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0.0, 1.0, 100)

    color1 = 'skyblue'
    ax1.hist(inter_sims, bins=bins, color=color1, alpha=0.7, edgecolor='dimgray', linewidth=0.5)
    ax1.set_xlabel('Similarity Range', fontsize=16)
    ax1.set_ylabel('Fre. of Inter- Similarity', color=color1, fontsize=16, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='black') 
    
    ax2 = ax1.twinx()
    color2 = 'sandybrown'
    ax2.hist(intra_sims, bins=bins, color=color2, alpha=0.7, edgecolor='dimgray', linewidth=0.5)
    ax2.set_ylabel('Fre. of Intra- Similarity', color='orange', fontsize=16, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='black')

    plt.title(title, fontsize=14)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# 2. 메인 로직: 모델 로드 및 특징 추출
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yoloe-v8m-seg.pt")
    parser.add_argument("--domains", type=str, nargs='+', default=["cartoon", "handmake", "painting", "sketch", "tattoo", "weather"])
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./analysis")
    parser.add_argument("--num_classes", type=int, default=10) 
    return parser.parse_args()

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(args.seed)

    print("[Info] Loading YOLOE Model...")
    try:
        model = YOLOE(args.model)
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
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
    labels_class = []

    print("[Info] Extracting Features using Visual Prompts...")
    for dom in args.domains:
        ann_file = os.path.join(args.data_dir, "ood_coco", dom, "annotations", "instances_val2017.json")
        img_dir = os.path.join(args.data_dir, "ood_coco", dom, "val2017")
        
        if not os.path.exists(ann_file):
            continue
            
        coco = COCO(ann_file)
        all_cat_ids = coco.getCatIds()
        target_cat_ids = all_cat_ids[:args.num_classes] 
        
        ann_ids = coco.getAnnIds(catIds=target_cat_ids)
        if not ann_ids: continue
        
        print(f"  -> Domain: {dom} (Found {len(ann_ids)} annotations)")
        for ann_id in tqdm(ann_ids, leave=False):
            try:
                ann = coco.loadAnns(ann_id)[0]
                img_info = coco.loadImgs(ann['image_id'])[0]
                img_path = os.path.join(img_dir, img_info['file_name'])
            except: continue
            
            if not os.path.exists(img_path): continue
            
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0: continue
            category_id = ann['category_id']

            visual_prompts = dict(
                bboxes=np.array([[float(x), float(y), float(x + w), float(y + h)]], dtype=np.float32),
                cls=np.array([category_id], dtype=np.int32)
            )

            visual_features.clear()
            try:
                _ = model.predict(source=img_path, visual_prompts=visual_prompts, predictor=YOLOEVPSegPredictor, save=False, verbose=False)
            except: continue
            
            if 'savpe_emb' in visual_features:
                savpe_out = visual_features['savpe_emb']
                if isinstance(savpe_out, tuple): savpe_out = savpe_out[0]
                
                v_vec = savpe_out.squeeze()
                if v_vec.numel() == 0 or v_vec.dim() != 1:
                    v_vec = v_vec.flatten()
                
                all_visual_embeddings.append(v_vec.detach().cpu().numpy())
                labels_domain.append(dom)
                labels_class.append(category_id)

    if not all_visual_embeddings:
        print("[Error] No visual embeddings extracted.")
        return

    features_np = np.array(all_visual_embeddings)

    # ==========================================
    # 3. IISR 분석 및 저장
    # ==========================================
    print(f"\n[Info] Total Features Extracted: {len(features_np)}")
    
    # --- Class Level IISR ---
    print("\n[Analysis 1] Class-Level IISR")
    intra_c, inter_c, iisr_c = calculate_similarities_and_iisr(features_np, labels_class)
    print(f"  -> Final Class IISR Score: {iisr_c:.4f}")
    plot_path_c = os.path.join(args.save_dir, "yoloe_iisr_class.png")
    plot_dual_histogram(intra_c, inter_c, f"YOLOE Class-level IISR: {iisr_c:.4f}", plot_path_c)
    print(f"  -> Saved plot: {plot_path_c}")

    # --- Domain Level IISR ---
    print("\n[Analysis 2] Domain-Level IISR")
    intra_d, inter_d, iisr_d = calculate_similarities_and_iisr(features_np, labels_domain)
    print(f"  -> Final Domain IISR Score: {iisr_d:.4f}")
    plot_path_d = os.path.join(args.save_dir, "yoloe_iisr_domain.png")
    plot_dual_histogram(intra_d, inter_d, f"YOLOE Domain-level IISR: {iisr_d:.4f}", plot_path_d)
    print(f"  -> Saved plot: {plot_path_d}")

    print("\n[Success] All tasks completed!")

if __name__ == "__main__":
    args = parse_args()
    main(args)