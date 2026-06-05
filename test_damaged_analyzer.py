# test_damaged_analyzer.py
import cv2
import numpy as np
from PIL import Image

img_path = r"c:\Users\navee\OneDrive\Desktop\ML2\snapshots\good_20260529_030203.png"

# Load image and resize to 128x128 to match Streamlit preprocessing
pil_image = Image.open(img_path)
pil_resized = pil_image.resize((128, 128))

# Core calculations
img_resized = pil_resized.resize((400, 400))
cv_img = np.array(img_resized.convert("RGB"))
cv_gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
blurred = cv2.GaussianBlur(cv_gray, (5, 5), 0)
canny = cv2.Canny(blurred, 30, 100)

h, w = canny.shape
center_box = canny[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
mean_density = np.mean(center_box)

bh, bw = center_box.shape
block_h, block_w = bh // 3, bw // 3
block_densities = []
for r in range(3):
    for c in range(3):
        sub_block = center_box[r*block_h:(r+1)*block_h, c*block_w:(c+1)*block_w]
        block_densities.append(np.mean(sub_block))
tread_variance = np.std(block_densities)

print("\n--- Damaged Tyre Snapshot Metrics ---")
print("Canny Mean Density:", mean_density)
print("Canny Tread Variance:", tread_variance)
print("Block Densities:", block_densities)
