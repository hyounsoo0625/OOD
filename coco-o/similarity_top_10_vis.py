import argparse
import torch
import numpy as np
import cv2
import os
import random
import pickle # 🌟 추가: pkl 파일 처리를 위한 라이브러리
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO 
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOE Visual Prompt Top-10 Retrieval (PKL Cache & Multi-Query)")
    parser.add_argument("--model", type=str, default="yoloe-v8m-seg.pt")
    parser.add_argument("--device", type=str, default="0", help="사용할 GPU 번호 (예: '0') 또는 'cpu'")
    
    parser.add_argument("--target_domains", type=str, nargs='+', default=["cartoon", "handmake", "painting", "sketch", "tattoo", "weather"])
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--query_domain", type=str, default="cartoon")
    
    # 🌟 다중 쿼리를 위한 설정 추가
    parser.add_argument("--num_queries", type=int, default=10, help="테스트할 쿼리의 개수")
    parser.add_argument("--sample_size_per_domain", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="./analysis")
    
    # 🌟 DB 파일 이름 추가
    parser.add_argument("--db_filename", type=str, default="embedding_db.pkl", help="캐싱할 DB 파일명")
    return parser.parse_args()

def draw_bbox(ax, img_path, bbox, color='red', title=""):
    img = cv2.imread(img_path)
    if img is None: return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    x, y, w, h = bbox
    rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"[Info] PyTorch CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[Info] Using GPU device: {args.device} ({torch.cuda.get_device_name(int(args.device) if args.device.isdigit() else 0)})")
    else:
        print("[Warning] CUDA is not available. Using CPU. This will be slow!")

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

    def get_embedding(img_path, bbox):
        x, y, w, h = bbox
        visual_prompts = dict(
            bboxes=np.array([[float(x), float(y), float(x + w), float(y + h)]], dtype=np.float32),
            cls=np.array([0], dtype=np.int32)
        )
        visual_features.clear()
        try:
            _ = model.predict(
                source=img_path, 
                visual_prompts=visual_prompts, 
                predictor=YOLOEVPSegPredictor, 
                save=False, 
                verbose=False,
                device=args.device 
            )
        except Exception:
            return None
        
        if 'savpe_emb' in visual_features:
            savpe_out = visual_features['savpe_emb']
            if isinstance(savpe_out, tuple): savpe_out = savpe_out[0]
            v_vec = savpe_out.squeeze()
            if v_vec.numel() == 0 or v_vec.dim() != 1:
                v_vec = v_vec.flatten()
            return v_vec.detach().cpu().numpy()
        return None

    # ==========================================
    # 1. Target Database 구축 (PKL 캐싱 적용)
    # ==========================================
    db_path = os.path.join(args.save_dir, args.db_filename)
    database = []

    if os.path.exists(db_path):
        print(f"[Info] Found existing embedding DB at {db_path}. Loading data...")
        with open(db_path, 'rb') as f:
            database = pickle.load(f)
        print(f"[Info] Loaded {len(database)} embeddings from PKL.")
    else:
        print(f"[Info] No existing DB found. Extracting embeddings to create {db_path}...")
        for dom in args.target_domains:
            target_ann_file = os.path.join(args.data_dir, "ood_coco", dom, "annotations", "instances_val2017.json")
            target_img_dir = os.path.join(args.data_dir, "ood_coco", dom, "val2017")
            
            if not os.path.exists(target_ann_file):
                continue

            print(f"\n[Info] Processing target domain: {dom}")
            try:
                target_coco = COCO(target_ann_file)
            except Exception:
                continue

            target_ann_ids = target_coco.getAnnIds()
            if args.sample_size_per_domain > 0 and len(target_ann_ids) > args.sample_size_per_domain:
                target_ann_ids = random.sample(target_ann_ids, args.sample_size_per_domain)

            print(f"       Extracting embeddings for {len(target_ann_ids)} objects...")
            for ann_id in tqdm(target_ann_ids, desc=f"Domain: {dom}"):
                ann = target_coco.loadAnns(ann_id)[0]
                x, y, w, h = ann['bbox']
                if w <= 0 or h <= 0: continue

                img_info = target_coco.loadImgs(ann['image_id'])[0]
                img_path = os.path.join(target_img_dir, img_info['file_name'])
                
                emb = get_embedding(img_path, ann['bbox'])
                if emb is not None:
                    database.append({
                        'domain': dom,
                        'ann_id': ann_id,
                        'img_path': img_path,
                        'bbox': ann['bbox'],
                        'embedding': emb
                    })
        
        if not database:
            print("[Error] No valid embeddings extracted. Exiting.")
            return

        # PKL 파일로 저장
        with open(db_path, 'wb') as f:
            pickle.dump(database, f)
        print(f"\n[Info] Successfully saved {len(database)} embeddings to {db_path}.")

    # ==========================================
    # 2. Query 10개 랜덤 선택 및 탐색 루프
    # ==========================================
    query_ann_file = os.path.join(args.data_dir, "ood_coco", args.query_domain, "annotations", "instances_val2017.json")
    query_img_dir = os.path.join(args.data_dir, "ood_coco", args.query_domain, "val2017")

    try:
        query_coco = COCO(query_ann_file)
    except Exception as e:
        print(f"[Error] Failed to load Query COCO annotations: {e}")
        return

    # 쿼리 후보 추출 (너무 작은 bbox 제외)
    all_query_ids = query_coco.getAnnIds()
    valid_query_ids = []
    for q_id in all_query_ids:
        ann = query_coco.loadAnns(q_id)[0]
        if ann['bbox'][2] > 10 and ann['bbox'][3] > 10: # w, h가 너무 작은 객체는 패스
            valid_query_ids.append(q_id)

    if len(valid_query_ids) < args.num_queries:
        print(f"[Warning] Not enough valid queries. Using {len(valid_query_ids)} instead of {args.num_queries}.")
        selected_queries = valid_query_ids
    else:
        selected_queries = random.sample(valid_query_ids, args.num_queries)

    print(f"\n[Info] Starting search for {len(selected_queries)} queries...")

    # 지정된 쿼리 개수만큼 루프
    for idx, query_ann_id in enumerate(selected_queries):
        print(f"\n--- Processing Query {idx+1}/{len(selected_queries)} (Ann ID: {query_ann_id}) ---")
        
        query_ann = query_coco.loadAnns(query_ann_id)[0]
        query_img_info = query_coco.loadImgs(query_ann['image_id'])[0]
        query_img_path = os.path.join(query_img_dir, query_img_info['file_name'])
        query_bbox = query_ann['bbox']

        query_emb = get_embedding(query_img_path, query_bbox)
        if query_emb is None:
            print(f"[Warning] Failed to extract embedding for Query {query_ann_id}. Skipping.")
            continue

        # 3. 거리 계산 (자기 자신 제외)
        results = []
        for item in database:
            # Query와 동일한 객체가 DB에 있다면 건너뛰기
            if item['domain'] == args.query_domain and item['ann_id'] == query_ann_id:
                continue
            
            dist = np.linalg.norm(query_emb - item['embedding'])
            # 기존 DB 딕셔너리를 수정하지 않고 새 딕셔너리로 묶어서 리스트에 추가
            results.append({**item, 'distance': dist})

        if not results:
            continue

        # 거리가 짧은 순으로 정렬
        results_sorted = sorted(results, key=lambda x: x['distance'])
        top_10_results = results_sorted[:10]

        # 4. 시각화 및 개별 파일 저장
        fig = plt.figure(figsize=(20, 13))
        
        ax_query = plt.subplot2grid((3, 5), (0, 2))
        query_title = f"[QUERY]\nDomain: {args.query_domain}\nID: {query_ann_id}"
        draw_bbox(ax_query, query_img_path, query_bbox, color='red', title=query_title)

        for i, result in enumerate(top_10_results):
            row = (i // 5) + 1
            col = i % 5
            ax_res = plt.subplot2grid((3, 5), (row, col))
            title_text = f"Rank {i+1} [{result['domain']}]\nDist: {result['distance']:.3f}"
            draw_bbox(ax_res, result['img_path'], result['bbox'], color='lime', title=title_text)

        plt.tight_layout()

        save_name = f"query_{idx+1}_id_{query_ann_id}.png"
        final_save_path = os.path.join(args.save_dir, save_name)
        plt.savefig(final_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Success] Saved result to {final_save_path}")

    print("\n[Info] All queries processed completely!")

if __name__ == "__main__":
    args = parse_args()
    main(args)