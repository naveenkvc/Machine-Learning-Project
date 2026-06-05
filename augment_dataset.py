# augment_dataset.py
"""
AI-TREAD TensorFlow & OpenCV Hybrid Dataset Augmenter.
Multiplies minor datasets by generating randomized spatial and color permutations,
including perspective warps, Gaussian blur/noise, rotations, flips, and zoom crops.
"""

import os
import argparse
import random
import cv2
import numpy as np

# Hide TensorFlow init logs for cleaner console output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    import tensorflow as tf
    HAS_TF = True
except Exception:
    HAS_TF = False

def apply_opencv_spatial_transforms(img):
    """Applies rotation, blur, perspective warp, and Gaussian noise via OpenCV."""
    h, w = img.shape[:2]
    
    # 1. Random Rotation (±25 degrees)
    angle = random.uniform(-25, 25)
    rot_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img_aug = cv2.warpAffine(img, rot_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # 2. Random Perspective Transform (simulates camera view angles)
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    px_shift = int(min(h, w) * 0.06)
    s1 = random.randint(-px_shift, px_shift)
    s2 = random.randint(-px_shift, px_shift)
    s3 = random.randint(-px_shift, px_shift)
    s4 = random.randint(-px_shift, px_shift)
    pts2 = np.float32([
        [s1, s2], 
        [w - s1, s2], 
        [s3, h - s4], 
        [w - s3, h - s4]
    ])
    pers_matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_aug = cv2.warpPerspective(img_aug, pers_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # 3. Random Gaussian Blur
    if random.random() > 0.4:
        kernel = random.choice([3, 5, 7])
        img_aug = cv2.GaussianBlur(img_aug, (kernel, kernel), 0)
        
    # 4. Add Gaussian Noise (high-ISO road-camera grain simulation)
    if random.random() > 0.4:
        noise = np.random.normal(0, random.uniform(1.0, 4.0), img_aug.shape).astype(np.int32)
        img_aug = np.clip(img_aug.astype(np.int32) + noise, 0, 255).astype(np.uint8)
        
    return img_aug

def apply_tensorflow_color_transforms(img_np):
    """Applies horizontal flips, brightness, contrast, and zoom crops via TensorFlow Eager execution (fallback to OpenCV/NumPy)."""
    h, w = img_np.shape[:2]
    
    if HAS_TF:
        try:
            # Convert BGR OpenCV image to RGB tensor [0.0, 1.0]
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            tensor = tf.convert_to_tensor(img_rgb, dtype=tf.float32) / 255.0
            
            # 1. Random Horizontal Flip (50% probability)
            if random.random() > 0.5:
                tensor = tf.image.flip_left_right(tensor)
                
            # 2. Random Brightness Variation
            delta = random.uniform(-0.15, 0.15)
            tensor = tf.image.adjust_brightness(tensor, delta)
            
            # 3. Random Contrast Adjustments
            contrast_factor = random.uniform(0.75, 1.35)
            tensor = tf.image.adjust_contrast(tensor, contrast_factor)
            
            # 4. Random Zoom Crop & Resize
            crop_scale = random.uniform(0.80, 0.95)
            crop_h = int(h * crop_scale)
            crop_w = int(w * crop_scale)
            
            # Slice random bounding coordinates
            offset_h = random.randint(0, h - crop_h)
            offset_w = random.randint(0, w - crop_w)
            
            tensor_cropped = tf.image.crop_to_bounding_box(tensor, offset_h, offset_w, crop_h, crop_w)
            # Resize back to model input size
            tensor_resized = tf.image.resize(tensor_cropped, (h, w))
            
            # Clip back to standard RGB pixel range and convert back to BGR numpy array
            tensor_clipped = tf.clip_by_value(tensor_resized * 255.0, 0.0, 255.0)
            augmented_np = tensor_clipped.numpy().astype(np.uint8)
            
            augmented_bgr = cv2.cvtColor(augmented_np, cv2.COLOR_RGB2BGR)
            return augmented_bgr
        except Exception:
            pass # Fallback to OpenCV/NumPy below if TensorFlow Eager fails dynamically
            
    # --- OpenCV & NumPy Graceful Robust Fallback ---
    img_aug = img_np.copy()
    
    # 1. Random Horizontal Flip
    if random.random() > 0.5:
        img_aug = cv2.flip(img_aug, 1)
        
    # 2. Random Brightness (±35 pixel values)
    delta = random.randint(-35, 35)
    img_aug = np.clip(img_aug.astype(np.int32) + delta, 0, 255).astype(np.uint8)
    
    # 3. Random Contrast Adjustment (multiply brightness values)
    contrast_factor = random.uniform(0.75, 1.35)
    img_aug = np.clip(img_aug.astype(np.float32) * contrast_factor, 0, 255).astype(np.uint8)
    
    # 4. Random Zoom Crop & Resize back
    crop_scale = random.uniform(0.80, 0.95)
    crop_h = int(h * crop_scale)
    crop_w = int(w * crop_scale)
    
    offset_h = random.randint(0, h - crop_h)
    offset_w = random.randint(0, w - crop_w)
    
    cropped = img_aug[offset_h : offset_h + crop_h, offset_w : offset_w + crop_w]
    img_aug = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return img_aug

def expand_dataset(input_dir, output_dir, factor=20):
    """Iterates through classes, loading raw files and writing balanced augmentations."""
    print(f"\n[DATA AUGMENTER] Target Expansion Factor: {factor}x per image")
    print(f"[DATA AUGMENTER] Scanning input directory '{input_dir}'...")
    
    classes = ["good", "worn", "damaged"]
    total_written = 0
    total_originals = 0
    
    # Prepare directories
    for c in classes:
        os.makedirs(os.path.join(output_dir, c), exist_ok=True)
        
    for c in classes:
        folder_path = os.path.join(input_dir, c)
        if not os.path.exists(folder_path):
            print(f"[WARNING] Input folder '{folder_path}' missing. Skipping class '{c}'.")
            continue
            
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"  * Class '{c.upper()}': Found {len(files)} original files.")
        
        for file in files:
            filepath = os.path.join(folder_path, file)
            total_originals += 1
            
            # Load original
            orig_img = cv2.imread(filepath)
            if orig_img is None:
                continue
                
            base_name, ext = os.path.splitext(file)
            
            # Generate factor-many unique augmented variants
            for idx in range(1, factor + 1):
                # Apply OpenCV spatial matrices
                cv_aug = apply_opencv_spatial_transforms(orig_img)
                # Apply TensorFlow color tensors
                tf_aug = apply_tensorflow_color_transforms(cv_aug)
                
                # Save augmented file
                out_name = f"{base_name}_aug_{idx}{ext}"
                out_path = os.path.join(output_dir, c, out_name)
                
                cv2.imwrite(out_path, tf_aug)
                total_written += 1

    print("\n" + "="*50)
    print("[SUCCESS] DATASET AUGMENTATION COMPLETED")
    print("="*50)
    print(f"Total Original Images Processed: {total_originals}")
    print(f"Total Augmented Images Written:   {total_written}")
    print(f"Output Dataset Directory:        '{output_dir}/'")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TensorFlow & OpenCV dataset augmenter.")
    parser.add_argument("--input_dir", type=str, default="data", help="Input dataset folder")
    parser.add_argument("--output_dir", type=str, default="dataset_augmented", help="Target output folder")
    parser.add_argument("--factor", type=int, default=20, help="Number of augmented images to generate per original")
    args = parser.parse_args()
    
    expand_dataset(args.input_dir, args.output_dir, args.factor)
