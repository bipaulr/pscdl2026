import sys, os
sys.path.insert(0, os.path.abspath('.'))
import cv2
import numpy as np
from utils.evaluator import evaluate_dataset

# Save our synthetic GT mask first so the harness can find it
gt_mask = np.zeros((480, 640), dtype=np.uint8)
gt_mask[300:380, 280:360] = 255
os.makedirs("data/masks", exist_ok=True)
os.makedirs("data/videos", exist_ok=True)

# Copy test video into data/videos/
import shutil
shutil.copy("data/test_video.mp4", "data/videos/test_video.mp4")
cv2.imwrite("data/masks/test_video.png", gt_mask)

print("Running evaluation harness...\n")
results = evaluate_dataset(
    video_dir  = "data/videos",
    gt_dir     = "data/masks",
    output_dir = "outputs/predictions",
)

print(f"\nSummary: {results}")