import argparse
import numpy as np
import os
import pickle
from pycocotools.coco import COCO 
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yoloe-v8m-seg.pt")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--num_samples", type=int, default=500, help="평가할 샘플 수")
    return parser.parse_args()

def main(args):
    model = YOLOE(args.model)
    
    visual_features = {}
    def savpe_hook_fn(module, input, output):
        visual_features['savpe_emb'] = output

    for name, module in model.model.named_modules():
        if 'savpe' in module.__class__.__name__.lower():
            module.register_forward_hook(savpe_hook_fn)
            break

    ann_file = os.path.join(args.data_dir, "annotations", "instances_val2017.json")
    coco = COCO(ann_file)
    ann_ids = coco.getAnnIds()[:args.num_samples] # 빠른 테스트를 위해 샘플 수 제한

    corruptions = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 'shot_noise', 'snow', 'zoom_blur'] # 논문용 대표 4종
    severities = ['1', '2', '3', '4', '5']
    
    # 데이터 저장용 딕셔너리 구조: results[img_id][corruption][severity] = embedding
    results = {}

    print("[Info] Extracting Features...")
    for ann_id in tqdm(ann_ids):
        ann = coco.loadAnns(ann_id)[0]
        img_info = coco.loadImgs(ann['image_id'])[0]
        file_name = img_info['file_name']
        x, y, w, h = ann['bbox']
        if w <= 0 or h <= 0: continue
        
        category_id = ann['category_id']
        visual_prompts = dict(
            bboxes=np.array([[float(x), float(y), float(x + w), float(y + h)]], dtype=np.float32),
            cls=np.array([category_id], dtype=np.int32)
        )

        results[ann_id] = {'category_id': category_id, 'clean': None, 'corrupted': {}}
        
        # 1. Clean 이미지 추출 (val2017 원본이 있다고 가정)
        clean_path = os.path.join(args.data_dir, "val2017", file_name)
        if os.path.exists(clean_path):
            visual_features.clear()
            try:
                model.predict(source=clean_path, visual_prompts=visual_prompts, predictor=YOLOEVPSegPredictor, save=False, verbose=False)
                results[ann_id]['clean'] = visual_features['savpe_emb'][0].squeeze().detach().cpu().numpy()
            except: pass

        # 2. Corrupted 이미지 추출
        for corr in corruptions:
            results[ann_id]['corrupted'][corr] = {}
            for sev in severities:
                corr_path = os.path.join(args.data_dir, "coco-c", corr, sev, file_name)
                if not os.path.exists(corr_path): continue
                
                visual_features.clear()
                try:
                    model.predict(source=corr_path, visual_prompts=visual_prompts, predictor=YOLOEVPSegPredictor, save=False, verbose=False)
                    results[ann_id]['corrupted'][corr][sev] = visual_features['savpe_emb'][0].squeeze().detach().cpu().numpy()
                except: pass

    # 결과 저장
    os.makedirs("./analysis", exist_ok=True)
    with open("./analysis/cococ_features.pkl", "wb") as f:
        pickle.dump(results, f)
    print("[Success] Features saved to ./analysis/cococ_features.pkl")

if __name__ == "__main__":
    main(parse_args())