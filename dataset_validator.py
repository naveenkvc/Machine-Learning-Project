# dataset_validator.py
"""
AI-TREAD Dataset Validator & Background Segmenter.
Audits datasets for corruption, duplicates, label anomalies, removes backgrounds using GrabCut, 
and generates procedurally textured synthetic tyre images for test pipelines.
"""

import os
import hashlib
import argparse
import cv2
import numpy as np
from PIL import Image

def get_image_hash(filepath):
    """Computes MD5 hash of file bytes to find exact duplicates."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def segment_tyre_grabcut(image_path, output_path=None):
    """
    Applies OpenCV GrabCut segmentation to isolate the tyre carcass from vehicle body,
    shadows, and roadway backgrounds. Blackens all background pixels.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        h, w = img.shape[:2]
        
        # Define rectangle bounding the tyre (assuming centered carcass in standard scans)
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
        
        # Initialize masks and GrabCut model arrays
        mask = np.zeros((h, w), dtype=np.uint8)
        bgdModel = np.zeros((1, 65), dtype=np.float64)
        fgdModel = np.zeros((1, 65), dtype=np.float64)
        
        # Execute GrabCut iterations
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
        
        # Create mask where background (GC_BGD, GC_PR_BGD) is 0 and foreground is 1
        mask2 = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype('uint8')
        
        # Apply mask to image
        segmented_img = img * mask2[:, :, np.newaxis]
        
        if output_path:
            cv2.imwrite(output_path, segmented_img)
            
        return segmented_img
    except Exception as e:
        print(f"[ERROR] GrabCut failed on {image_path}: {e}")
        return None

def generate_synthetic_dataset(data_dir="data"):
    """
    Procedurally creates a synthetic tyre dataset under data/good, data/worn, data/damaged.
    Simulates high-fidelity rubber textures, tread blocks, and fracture defects.
    """
    print(f"\n[SYNTHETIC GENERATOR] Creating test directories under '{data_dir}/'...")
    classes = ["good", "worn", "damaged"]
    for c in classes:
        os.makedirs(os.path.join(data_dir, c), exist_ok=True)
        
    num_samples = 220
    print(f"[SYNTHETIC GENERATOR] Generating {num_samples} procedural samples per class...")
    
    np.random.seed(42)
    
    for c in classes:
        for idx in range(1, num_samples + 1):
            # 1. Base dark rubber texture canvas
            img = np.random.normal(30, 3, (400, 400, 3)).astype(np.uint8)
            
            # Draw tyre contour background circle (dark grey carcass)
            cx, cy = 200, 200
            cv2.circle(img, (cx, cy), 160, (45, 45, 45), -1)
            cv2.circle(img, (cx, cy), 80, (20, 20, 20), -1) # inner rim space
            
            # 2. Draw tread block structures
            if c == "good":
                # Healthy parallel treads (prominent, high contrast channels)
                for r in range(95, 150, 12):
                    cv2.circle(img, (cx, cy), r, (15, 15, 15), 3)
                # Repeating transverse tread ribs
                for ang in range(0, 360, 15):
                    rad = np.deg2rad(ang)
                    x1 = int(cx + 90 * np.cos(rad))
                    y1 = int(cy + 90 * np.sin(rad))
                    x2 = int(cx + 155 * np.cos(rad))
                    y2 = int(cy + 155 * np.sin(rad))
                    cv2.line(img, (x1, y1), (x2, y2), (10, 10, 10), 3)
                    
            elif c == "worn":
                # Flat, smooth tread lines (blurry, extremely low contrast)
                for r in range(95, 150, 12):
                    cv2.circle(img, (cx, cy), r, (38, 38, 38), 1)
                # Extremely thin, faded transverse ridges
                for ang in range(0, 360, 20):
                    rad = np.deg2rad(ang)
                    x1 = int(cx + 90 * np.cos(rad))
                    y1 = int(cy + 90 * np.sin(rad))
                    x2 = int(cx + 155 * np.cos(rad))
                    y2 = int(cy + 155 * np.sin(rad))
                    cv2.line(img, (x1, y1), (x2, y2), (32, 32, 32), 1)
                # Apply Gaussian Blur to simulate smooth, bald surfaces
                img = cv2.GaussianBlur(img, (15, 15), 0)
                
            elif c == "damaged":
                # Base tread lines (can be normal or worn)
                for r in range(95, 150, 12):
                    cv2.circle(img, (cx, cy), r, (15, 15, 15), 2)
                for ang in range(0, 360, 15):
                    rad = np.deg2rad(ang)
                    x1 = int(cx + 90 * np.cos(rad))
                    y1 = int(cy + 90 * np.sin(rad))
                    x2 = int(cx + 155 * np.cos(rad))
                    y2 = int(cy + 155 * np.sin(rad))
                    cv2.line(img, (x1, y1), (x2, y2), (10, 10, 10), 2)
                
                # Overlay prominent localized fractures / jagged cracks (high variance cuts)
                # Crack 1: Main structural tear
                points = np.array([
                    [220, 200], [240, 215], [230, 235], [260, 255], [250, 275]
                ], dtype=np.int32)
                cv2.polylines(img, [points], False, (5, 5, 5), 5)
                # Add highlighting white exposure boundaries (steel belts showing)
                cv2.polylines(img, [points], False, (220, 220, 220), 1)
                
                # Crack 2: Sidewall cut
                cv2.line(img, (110, 110), (130, 130), (5, 5, 5), 4)
                cv2.line(img, (110, 110), (130, 130), (200, 200, 200), 1)
                
            # Add Gaussian noise for realistic camera grain texture
            noise = np.random.normal(0, 1.5, img.shape).astype(np.int32)
            img = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)
            
            # Save file
            filepath = os.path.join(data_dir, c, f"synthetic_{c}_{idx}.png")
            cv2.imwrite(filepath, img)
            
    print("[SUCCESS] Synthetic Tyre Dataset generated cleanly!")

def validate_dataset(data_dir="data"):
    """
    Audits dataset folders. Identifies corruption, exact duplicate hashes,
    displays balance distributions, and flags label integrity warnings.
    """
    if not os.path.exists(data_dir):
        print(f"[WARNING] Directory '{data_dir}' not found. Cannot validate.")
        return False
        
    print(f"\n[DATASET AUDITOR] Launching validation scans on '{data_dir}/'...")
    
    classes = ["good", "worn", "damaged"]
    metrics = {c: {"total": 0, "corrupt": [], "duplicates": [], "labeled_ok": 0, "warnings": []} for c in classes}
    
    seen_hashes = {}
    total_scanned = 0
    
    for c in classes:
        folder_path = os.path.join(data_dir, c)
        if not os.path.exists(folder_path):
            continue
            
        for file in os.listdir(folder_path):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            filepath = os.path.join(folder_path, file)
            total_scanned += 1
            metrics[c]["total"] += 1
            
            # 1. Audit for Corruption (can PIL open and load pixel data?)
            try:
                with Image.open(filepath) as pil_img:
                    pil_img.verify() # check file integrity
                # Load again with CV2 to confirm pixel accessibility
                cv_img = cv2.imread(filepath)
                if cv_img is None or cv_img.size == 0:
                    metrics[c]["corrupt"].append(file)
                    continue
            except Exception:
                metrics[c]["corrupt"].append(file)
                continue
                
            # 2. Audit for Duplicates via MD5 checks
            fhash = get_image_hash(filepath)
            if fhash in seen_hashes:
                metrics[c]["duplicates"].append((file, seen_hashes[fhash]))
                continue
            seen_hashes[fhash] = f"{c}/{file}"
            
            # 3. Audit for Label Integrity anomalies
            # Good tyres should have distinct edge density. Damaged tyres should have cracks (std deviation)
            cv_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(cv_gray, (5, 5), 0)
            canny = cv2.Canny(blurred, 30, 100)
            mean_density = np.mean(canny)
            
            if c == "good" and mean_density < 15.0:
                metrics[c]["warnings"].append((file, f"Anomalously low edge density ({mean_density:.1f}%) for a 'Good' label. Flat profile wear suspected."))
            elif c == "worn" and mean_density > 38.0:
                metrics[c]["warnings"].append((file, f"Anomalously high edge density ({mean_density:.1f}%) for a 'Worn' label. Check for wrong folder."))
            elif c == "damaged":
                # Check standard dev in grid block sectors
                bh, bw = canny.shape
                block_h, block_w = bh // 3, bw // 3
                block_densities = []
                for r in range(3):
                    for idx_c in range(3):
                        block = canny[r*block_h:(r+1)*block_h, idx_c*block_w:(idx_c+1)*block_w]
                        block_densities.append(np.mean(block))
                std_dev = np.std(block_densities)
                if std_dev < 8.0:
                    metrics[c]["warnings"].append((file, f"Highly uniform tread ribs (Std Dev {std_dev:.1f}) for a 'Damaged' label. Missing structural crack defects?"))
            
            metrics[c]["labeled_ok"] += 1

    # 4. Generate Dataset Quality Markdown Report
    report_path = "dataset_report.md"
    with open(report_path, 'w', encoding='utf-8') as rf:
        rf.write("# AI-TREAD // Machine Learning Dataset Audit Report\n\n")
        rf.write(f"**Target Directory scanned:** `{data_dir}/`  \n")
        rf.write(f"**Total Files Discovered:** `{total_scanned}`  \n")
        rf.write(f"**Audit Status:** Complete  \n\n")
        
        rf.write("## 📊 Class Balance Distributions\n")
        rf.write("| Category Label | Total Samples | Checked OK | Corrupted Files | Duplicate Detections | Warnings |\n")
        rf.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for c in classes:
            rf.write(f"| **{c.upper()}** | {metrics[c]['total']} | {metrics[c]['labeled_ok']} | {len(metrics[c]['corrupt'])} | {len(metrics[c]['duplicates'])} | {len(metrics[c]['warnings'])} |\n")
            
        rf.write("\n## 🚨 Critical Integrity Flags\n")
        
        # Write Corrupted
        corrupted_found = False
        rf.write("### 🟥 Corrupted Files (Immediate Deletion Required)\n")
        for c in classes:
            for file in metrics[c]["corrupt"]:
                corrupted_found = True
                rf.write(f"* **{c.upper()}:** `{file}` - Fails standard decoding checks.\n")
        if not corrupted_found:
            rf.write("_No corrupted image files found. Decoders OK._\n")
            
        # Write Duplicates
        dups_found = False
        rf.write("\n### 🟨 Exact Hash Duplicates (Action: Remove to prevent test leakage)\n")
        for c in classes:
            for file, original in metrics[c]["duplicates"]:
                dups_found = True
                rf.write(f"* **{c.upper()}:** `{file}` - Exact hash match of `{original}`.\n")
        if not dups_found:
            rf.write("_No duplicate file hashes detected. Dataset is clean._\n")
            
        # Write Label Integrity
        warnings_found = False
        rf.write("\n### 🟧 Label Contradiction Warnings (Suspected Mislabeling)\n")
        for c in classes:
            for file, msg in metrics[c]["warnings"]:
                warnings_found = True
                rf.write(f"* **{c.upper()}:** `{file}` - {msg}\n")
        if not warnings_found:
            rf.write("_Label validations passed. Pixel textures match categories._\n")
            
    print(f"\n[SUCCESS] Dataset Audit Report successfully exported to '{report_path}'!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset validator utilities.")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data folder")
    parser.add_argument("--generate_synthetic", action="store_true", help="Launch procedural tyre data generator")
    parser.add_argument("--segment_path", type=str, default="", help="Single file image to run GrabCut background removal")
    args = parser.parse_args()
    
    if args.generate_synthetic:
        generate_synthetic_dataset(args.data_dir)
        validate_dataset(args.data_dir)
    elif args.segment_path:
        segment_tyre_grabcut(args.segment_path, "segmented_tyre_output.png")
        print("[SUCCESS] Segmented tyre saved as 'segmented_tyre_output.png'")
    else:
        validate_dataset(args.data_dir)
