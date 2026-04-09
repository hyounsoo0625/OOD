import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from pycocotools.coco import COCO 
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
import random
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOE Embedding Channel Sensitivity Analysis")
    parser.add_argument("--model", type=str, default="yoloe-v8m-seg.pt") # 모델명 맞춤
    parser.add_argument("--domains", type=str, nargs='+', default=["cartoon", "handmake", "painting", "sketch", "tattoo", "weather"])
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./analysis")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes to evaluate")
    return parser.parse_args()

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("[Info] Loading YOLOE Model...")
    try:
        model = YOLOE(args.model)
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        return

    # Hook을 통한 SAVPE Feature 추출
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

    all_features = []
    labels_domain = []
    labels_class = []

    print("[Info] Extracting Features using Visual Prompts...")
    for dom in args.domains:
        ann_file = os.path.join(args.data_dir, "ood_coco", dom, "annotations", "instances_val2017.json")
        img_dir = os.path.join(args.data_dir, "ood_coco", dom, "val2017")
        
        if not os.path.exists(ann_file):
            print(f"[Warning] Not found: {ann_file}")
            continue
            
        coco = COCO(ann_file)
        target_cat_ids = coco.getCatIds()[:args.num_classes]
        ann_ids = coco.getAnnIds(catIds=target_cat_ids)
        
        if not ann_ids: continue
        
        print(f"  -> Processing Domain: {dom} (Annotations: {len(ann_ids)})")
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
                
                all_features.append(v_vec.detach().cpu().numpy())
                labels_domain.append(dom)
                labels_class.append(category_id)

    if not all_features:
        print("[Error] No features extracted.")
        return

    features = np.array(all_features)
    labels_domain = np.array(labels_domain)
    labels_class = np.array(labels_class)
    
    num_channels = features.shape[1]
    print(f"\n[Info] Total extracted samples: {features.shape[0]}")
    print(f"[Info] Embedding channel dimension: {num_channels}")

    # ==========================================
    # 1. Overall Domain Std (전체 도메인 평균 간의 Std)
    # ==========================================
    unique_domains = np.unique(labels_domain)
    domain_means = [features[labels_domain == d].mean(axis=0) for d in unique_domains]
    domain_stds = np.std(domain_means, axis=0)

    # ==========================================
    # 2. Overall Class Std (전체 클래스 평균 간의 Std)
    # ==========================================
    unique_classes = np.unique(labels_class)
    class_means = [features[labels_class == c].mean(axis=0) for c in unique_classes]
    class_stds = np.std(class_means, axis=0)

    # ==========================================
    # 3. Intra-Class Domain Std (같은 클래스 내에서 도메인 간의 Std) - **NEW**
    # ==========================================
    intra_class_domain_stds_list = []
    
    for c in unique_classes:
        mask_c = (labels_class == c)
        feats_c = features[mask_c]
        doms_c = labels_domain[mask_c]
        
        unique_doms_in_c = np.unique(doms_c)
        # 특정 클래스가 최소 2개 이상의 도메인에 존재해야 분산 비교 가능
        if len(unique_doms_in_c) > 1:
            domain_means_for_c = []
            for d in unique_doms_in_c:
                mask_d = (doms_c == d)
                domain_means_for_c.append(feats_c[mask_d].mean(axis=0))
            
            # 클래스 c에서의 도메인별 평균들 간의 표준편차
            std_c = np.std(domain_means_for_c, axis=0)
            intra_class_domain_stds_list.append(std_c)
            
    # 모든 클래스에 대한 결과를 평균 내어 최종 채널별 민감도 도출
    if intra_class_domain_stds_list:
        intra_class_domain_stds = np.mean(intra_class_domain_stds_list, axis=0)
    else:
        intra_class_domain_stds = np.zeros(num_channels)

    # ==========================================
    # 4. 시각화 (1x3 서브플롯)
    # ==========================================
    print("[Info] Plotting Histograms...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Overall Domain Std
    axes[0].hist(domain_stds, bins=50, alpha=0.6, color='steelblue', label='YOLOE', edgecolor='none')
    axes[0].set_title('Overall Domain Std', fontsize=12)
    axes[0].set_xlabel('(a)', fontsize=12)
    axes[0].set_ylabel('Number of Channels', fontsize=11)
    axes[0].grid(True, linestyle='-', alpha=0.7)

    # (b) Overall Class Std
    axes[1].hist(class_stds, bins=50, alpha=0.6, color='forestgreen', label='YOLOE', edgecolor='none')
    axes[1].set_title('Overall Class Std', fontsize=12)
    axes[1].set_xlabel('(b)', fontsize=12)
    axes[1].grid(True, linestyle='-', alpha=0.7)

    # (c) Intra-Class Domain Std (새로 추가된 지표)
    axes[2].hist(intra_class_domain_stds, bins=50, alpha=0.6, color='darkorange', label='YOLOE', edgecolor='none')
    axes[2].set_title('Intra-Class Domain Std\n(Style Shift within Same Object)', fontsize=12)
    axes[2].set_xlabel('(c)', fontsize=12)
    axes[2].grid(True, linestyle='-', alpha=0.7)

    plt.tight_layout()
    
    save_path = os.path.join(args.save_dir, "channel_sensitivity_extended.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Success] Saved extended plot to {save_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)