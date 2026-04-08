import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE 
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pycocotools.coco import COCO 
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
import random

def main():
    model = YOLOE("yoloe-11l-seg.pt")
    
    target_classes = ['person', 'car', 'dog', 'cat', 'chair', 'bird', 'horse', 'bottle', 'backpack', 'umbrella']
    num_samples = 30
    
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    dummy_vp = dict(bboxes=np.array([[0, 0, 100, 100]], dtype=np.float32), cls=np.array([0]))
    
    _ = model.predict(source=dummy_img, visual_prompts=dummy_vp, predictor=YOLOEVPSegPredictor, verbose=False)

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

    data_dir = "../data/coco"
    ann_file = f"{data_dir}/annotations/instances_val2017.json"
    img_dir = f"{data_dir}/val2017"
    
    try:
        coco = COCO(ann_file)
    except Exception as e:
        print(f"Error loading COCO annotations: {e}")
        return

    all_visual_embeddings = []
    labels_vis = []
    thumbnails_vis = []

    random.seed(42)

    for cls in tqdm(target_classes, desc="Processing classes"):
        try:
            catIds = coco.getCatIds(catNms=[cls])
            if not catIds: continue
            annIds = coco.getAnnIds(catIds=catIds)
            if not annIds: continue
            
            selected_annIds = random.sample(annIds, min(len(annIds), num_samples * 5))
            anns = coco.loadAnns(selected_annIds)
        except Exception:
            continue
        
        temp_embeddings = []
        temp_thumbnails = []
        count = 0
        
        for ann in anns:
            if count >= num_samples: break
            
            try:
                img_info = coco.loadImgs(ann['image_id'])[0]
                img_path = os.path.join(img_dir, img_info['file_name'])
                
                if not os.path.exists(img_path): continue
                
                x, y, w, h = ann['bbox']
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = int(x + w), int(y + h)
                
                if x2 <= x1 or y2 <= y1: continue

                visual_prompts = dict(
                    bboxes=np.array([[float(x), float(y), float(x + w), float(y + h)]], dtype=np.float32),
                    cls=np.array([target_classes.index(cls)], dtype=np.int32)
                )

                visual_features.clear()
                
                _ = model.predict(source=img_path, visual_prompts=visual_prompts, predictor=YOLOEVPSegPredictor, save=False, verbose=False)
                
                if 'savpe_emb' in visual_features:
                    savpe_out = visual_features['savpe_emb']
                    if isinstance(savpe_out, tuple): savpe_out = savpe_out[0]
                    
                    v_vec = savpe_out.squeeze()
                    if v_vec.numel() == 0 or v_vec.dim() != 1: continue
                    
                    v_vec = F.normalize(v_vec, dim=0, p=2)
                    
                    img_bgr = cv2.imread(img_path)
                    if img_bgr is not None:
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        patch = img_rgb[y1:y2, x1:x2]
                        patch_resized = cv2.resize(patch, (40, 40)) 
                        
                        temp_thumbnails.append(patch_resized)
                        temp_embeddings.append(v_vec.detach().cpu().numpy())
                        count += 1
            except Exception:
                continue
        
        if count > 1:
            all_visual_embeddings.extend(temp_embeddings)
            labels_vis.extend([cls] * count)
            thumbnails_vis.extend(temp_thumbnails)

    if len(all_visual_embeddings) == 0:
        print("Error: No visual data extracted.")
        return

    valid_classes = sorted(list(set(labels_vis)))
    X_vis = np.array(all_visual_embeddings)

    perp_vis = max(1, min(30, len(X_vis) - 1))
    tsne_vis = TSNE(n_components=2, metric='euclidean', perplexity=perp_vis, random_state=42, init='random')
    vis_2d = tsne_vis.fit_transform(X_vis)

    df_vis = pd.DataFrame({'t-SNE-1': vis_2d[:, 0], 't-SNE-2': vis_2d[:, 1], 'Category': labels_vis})

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 10))
    
    palette = sns.color_palette("husl", len(valid_classes))
    color_map = dict(zip(valid_classes, palette))

    sns.scatterplot(x="t-SNE-1", y="t-SNE-2", hue="Category", palette=color_map, data=df_vis, ax=ax, s=10, alpha=0.1, edgecolor='none', legend=False)
    
    for idx, row in df_vis.iterrows():
        thumb = thumbnails_vis[idx]
        color = color_map[row['Category']]
        
        imagebox = OffsetImage(thumb, zoom=0.6, alpha=0.85)
        ab = AnnotationBbox(imagebox, (row['t-SNE-1'], row['t-SNE-2']), frameon=True, 
                            pad=0.1, bboxprops=dict(edgecolor=color, lw=2, alpha=0.7))
        ax.add_artist(ab)

    cluster_centers = df_vis.groupby('Category')[['t-SNE-1', 't-SNE-2']].mean().reset_index()

    for cat in valid_classes:
        cx, cy = cluster_centers[cluster_centers['Category'] == cat][['t-SNE-1', 't-SNE-2']].values[0]
        cat_points = df_vis[df_vis['Category'] == cat]
        
        for _, row in cat_points.iterrows():
            px, py = row['t-SNE-1'], row['t-SNE-2']
            ax.plot([cx, px], [cy, py], color='gray', linestyle='--', linewidth=0.5, alpha=0.4, zorder=2)

    for idx, row in cluster_centers.iterrows():
        cat = row['Category']
        color = color_map[cat]
        ax.scatter(row['t-SNE-1'], row['t-SNE-2'], marker='*', s=1000, facecolor=color, edgecolor='black', linewidth=1.5, zorder=10, label=cat)

    ax.set_title("Visual Prompt Space with Thumbnails (10 COCO Classes)", fontsize=18, fontweight='bold')

    plt.tight_layout()
    os.makedirs("./analysis", exist_ok=True)
    plt.savefig("./analysis/yoloe_tsne_coco_10_classes.png", dpi=300, bbox_inches='tight')
    print("Success: Plot saved to './analysis/yoloe_tsne_coco_10_classes.png'")

if __name__ == "__main__":
    main()