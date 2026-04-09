import argparse
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE 
from pycocotools.coco import COCO 
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
import random
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOE Feature Distribution by Class and Domain")
    parser.add_argument("--model", type=str, default="yoloe-v8m-seg.pt")
    parser.add_argument("--domains", type=str, nargs='+', default=["cartoon", "handmake", "painting", "sketch", "tattoo", "weather"])
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./analysis")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes to visualize") # 클래스 개수 지정
    return parser.parse_args()

def main(args):
    # 모델 로드 및 Hook 설정
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
    labels_class = []

    random.seed(args.seed)

    # 1. 데이터 및 특징(Feature) 추출
    for dom in args.domains:
        ann_file = os.path.join(args.data_dir, "ood_coco", dom, "annotations", "instances_val2017.json")
        img_dir = os.path.join(args.data_dir, "ood_coco", dom, "val2017")
        
        if not os.path.exists(ann_file):
            continue
            
        coco = COCO(ann_file)
        
        # 지정된 개수만큼의 클래스(Category ID)만 필터링
        all_cat_ids = coco.getCatIds()
        target_cat_ids = all_cat_ids[:args.num_classes] 
        
        ann_ids = coco.getAnnIds(catIds=target_cat_ids)
        if not ann_ids: continue
        
        print(f"[Info] Extracting features for {dom} (Limited to {args.num_classes} classes)...")
        
        for ann_id in tqdm(ann_ids):
            try:
                ann = coco.loadAnns(ann_id)[0]
                img_info = coco.loadImgs(ann['image_id'])[0]
                img_path = os.path.join(img_dir, img_info['file_name'])
            except Exception:
                continue
            
            if not os.path.exists(img_path): continue
            
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0: continue
            
            category_id = ann['category_id']

            # [핵심] YOLOE Visual Prompt 구성 (실제 클래스 ID 주입)
            visual_prompts = dict(
                bboxes=np.array([[float(x), float(y), float(x + w), float(y + h)]], dtype=np.float32),
                cls=np.array([category_id], dtype=np.int32) 
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
                labels_class.append(category_id)

    if not all_visual_embeddings:
        print("[Error] No visual embeddings extracted.")
        return

    # 2. t-SNE 차원 축소
    print("[Info] Running t-SNE...")
    X_vis = np.array(all_visual_embeddings)
    tsne_vis = TSNE(n_components=2, metric='euclidean', perplexity=30, random_state=args.seed)
    vis_2d = tsne_vis.fit_transform(X_vis)

    # 문자열 도메인을 컬러맵핑을 위해 정수 인덱스로 변환
    unique_domains = list(set(labels_domain))
    domain_to_idx = {dom: i for i, dom in enumerate(unique_domains)}
    domain_indices = [domain_to_idx[d] for d in labels_domain]

    # 클래스 ID도 순차적 정수로 맵핑 (컬러바 표시용)
    unique_classes = list(set(labels_class))
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    class_indices = [class_to_idx[c] for c in labels_class]

    # 3. 첨부된 이미지 스타일의 시각화 (서브플롯 + 컬러바)
    print("[Info] Plotting distributions...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6)) # 필요시 1, 4 로 늘려서 세부 도메인별 플롯 추가 가능

    # [Plot 1] 클래스별 분포 (Class Distribution)
    scatter1 = axes[0].scatter(vis_2d[:, 0], vis_2d[:, 1], c=class_indices, cmap='Spectral', s=15, alpha=0.7)
    axes[0].set_title(f"t-SNE: By Class (Total {len(unique_classes)} Classes)")
    cbar1 = fig.colorbar(scatter1, ax=axes[0])
    cbar1.set_ticks(range(len(unique_classes)))
    cbar1.set_ticklabels(unique_classes)

    # [Plot 2] 도메인별 분포 (Domain Distribution)
    scatter2 = axes[1].scatter(vis_2d[:, 0], vis_2d[:, 1], c=domain_indices, cmap='Set2', s=15, alpha=0.7)
    axes[1].set_title(f"t-SNE: By Domain (Total {len(unique_domains)} Domains)")
    cbar2 = fig.colorbar(scatter2, ax=axes[1])
    cbar2.set_ticks(range(len(unique_domains)))
    cbar2.set_ticklabels(unique_domains)

    plt.tight_layout()
    
    # 저장
    os.makedirs(args.save_dir, exist_ok=True)
    save_filename = os.path.join(args.save_dir, f"tsne_distribution_{args.num_classes}classes.png")
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"[Success] Plot saved to {save_filename}")

if __name__ == "__main__":
    args = parse_args()
    main(args)