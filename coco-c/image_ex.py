import argparse
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

try:
    from imagecorruptions import corrupt, get_corruption_names
except ImportError:
    print("[Error] 'imagecorruptions' is not installed. Please run: pip install imagecorruptions")
    exit()

def parse_args():
    parser = argparse.ArgumentParser(description="Generate and Save Image Corruption Examples")
    parser.add_argument("--img_path", type=str, default="../data/coco/val2017/000000000139.jpg", help="Path to the sample image")
    parser.add_argument("--save_dir", type=str, default="./corruption_examples", help="Directory to save generated images")
    parser.add_argument("--img_size", type=int, default=256, help="Resize image for the summary grid (Individual files keep original size)")
    return parser.parse_args()

def main(args):
    if not os.path.exists(args.img_path):
        print(f"[Error] Image not found at {args.img_path}")
        print("Please provide a valid image path using --img_path")
        return

    # 1. 이미지 로드 및 전처리 (OpenCV는 BGR이므로 RGB로 변환)
    img_bgr = cv2.imread(args.img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 그리드용 리사이즈 이미지 (원본 비율 무시하고 정사각형으로 맞춤, 보기 편하게)
    img_grid_base = cv2.resize(img_rgb, (args.img_size, args.img_size))
    corruptions_list = get_corruption_names()
    severities = [1, 2, 3, 4, 5]

    print(f"[Info] Base image loaded: {args.img_path} (Shape: {img_rgb.shape})")
    print(f"[Info] Generating examples for {len(corruptions_list)} corruptions...")

    # 저장할 최상위 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)
    individual_save_dir = os.path.join(args.save_dir, "individual_files")
    os.makedirs(individual_save_dir, exist_ok=True)

    # Matplotlib 그리드 설정 (15행 x 6열: Clean + Sev 1~5)
    fig, axes = plt.subplots(len(corruptions_list), len(severities) + 1, 
                             figsize=(16, 2.5 * len(corruptions_list)))
    plt.subplots_adjust(wspace=0.05, hspace=0.3)

    for row_idx, corruption in enumerate(corruptions_list):
        # 개별 저장용 폴더 생성 (예: ./corruption_examples/individual_files/gaussian_noise/)
        corr_folder = os.path.join(individual_save_dir, corruption)
        os.makedirs(corr_folder, exist_ok=True)

        print(f"  -> Processing: {corruption}")

        # [Grid 0열] Clean 이미지 배치
        axes[row_idx, 0].imshow(img_grid_base)
        axes[row_idx, 0].axis('off')
        if row_idx == 0:
            axes[row_idx, 0].set_title("Clean (Original)", fontsize=14, fontweight='bold')
        
        # 행(Row)의 왼쪽에 노이즈 이름 표시
        axes[row_idx, 0].text(-0.1, 0.5, corruption, transform=axes[row_idx, 0].transAxes,
                              fontsize=14, fontweight='bold', va='center', ha='right')

        # Clean 이미지도 개별 저장
        clean_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(corr_folder, "clean.png"), clean_bgr)

        # [Grid 1~5열] 강도별 노이즈 이미지 생성 및 배치
        for col_idx, sev in enumerate(severities, start=1):
            # 1. 원본 크기 이미지에 노이즈 적용 (개별 저장용)
            corrupted_img_full = corrupt(img_rgb, corruption_name=corruption, severity=sev)
            corrupted_bgr_full = cv2.cvtColor(corrupted_img_full, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(corr_folder, f"sev_{sev}.png"), corrupted_bgr_full)

            # 2. 리사이즈된 이미지에 노이즈 적용 (그리드 시각화용)
            corrupted_img_grid = corrupt(img_grid_base, corruption_name=corruption, severity=sev)
            
            axes[row_idx, col_idx].imshow(corrupted_img_grid)
            axes[row_idx, col_idx].axis('off')
            
            # 첫 번째 행에만 강도(Severity) 타이틀 달기
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(f"Severity {sev}", fontsize=14, fontweight='bold')

    # 전체 그리드 이미지 저장
    grid_save_path = os.path.join(args.save_dir, "corruption_grid_summary.png")
    plt.savefig(grid_save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print("\n[Success] Process completed!")
    print(f"  1. Summary Grid saved at: {grid_save_path}")
    print(f"  2. Individual high-res files saved in: {individual_save_dir}/")

if __name__ == "__main__":
    args = parse_args()
    main(args)