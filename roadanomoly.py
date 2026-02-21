# --------- Single Colab Cell: Upload -> Better Pothole Detection -> Show ---------
!pip -q install opencv-python

import cv2
import numpy as np
from google.colab import files
from google.colab.patches import cv2_imshow

# 1) Upload
uploaded = files.upload()
img_path = list(uploaded.keys())[0]
print("Selected:", img_path)

# 2) Read
img = cv2.imread(img_path)
if img is None:
    raise RuntimeError("❌ Could not read the uploaded image. Upload a JPG/PNG again.")

h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3) Normalize lighting (important)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
g = clahe.apply(gray)

# 4) Smooth
blur = cv2.GaussianBlur(g, (5, 5), 0)

# 5) Darkness cue (blackhat highlights dark blobs relative to neighborhood)
k = max(31, ((min(h, w) // 6) | 1))       # adaptive-ish kernel, must be odd
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)
bh = cv2.normalize(blackhat, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

# 6) Texture cue (potholes usually have high texture/edges)
lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
lap = np.abs(lap)
lap = cv2.normalize(lap, None, 0, 1, cv2.NORM_MINMAX)

# 7) Combined score (tune weights if needed)
score = 0.6 * bh + 0.4 * lap
score_u8 = (score * 255).astype(np.uint8)

# 8) Threshold (auto)
_, mask = cv2.threshold(score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 9) Clean mask (reduce speckles + fill holes)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)

# 10) Keep ONLY the largest connected component (main pothole)
num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

if num <= 1:
    print("❌ No pothole-like region found. Try changing weights or kernel size.")
    output = img.copy()
else:
    # stats: [label] -> x,y,w,h,area ; label 0 is background
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + np.argmax(areas)

    pothole_mask = np.zeros_like(mask)
    pothole_mask[labels == largest_label] = 255

    # Optional: tighten by finding contour on largest region
    cnts, _ = cv2.findContours(pothole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = img.copy()

    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        x, y, bw, bhh = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x + bw, y + bhh), (0, 0, 255), 2)
        print("✅ Detected pothole area:", int(cv2.contourArea(cnt)))
    else:
        print("❌ Contour not found after component extraction.")

# 11) Show results + debug
print("\nResult (bounding box):")
cv2_imshow(output)

print("\nDebug: score map (brighter = more pothole-like)")
cv2_imshow(score_u8)

print("\nDebug: final mask (largest region extracted if any)")
if num > 1:
    cv2_imshow(pothole_mask)
else:
    cv2_imshow(mask)