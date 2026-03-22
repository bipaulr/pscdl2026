import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from utils.visualize import overlay_mask, side_by_side

# ---------------------------------------------------------------------------
# Load video
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture("data/test_video.mp4")
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()
print(f"Loaded {len(frames)} frames")

baseline = frames[:150]
eval_frames = frames[150:]

# ---------------------------------------------------------------------------
# Build MOG2 background model on baseline
# ---------------------------------------------------------------------------
mog2 = cv2.createBackgroundSubtractorMOG2(
    history=200,
    varThreshold=16.0,
    detectShadows=False
)
for frame in baseline:
    mog2.apply(frame)
print("MOG2 trained on baseline")

# ---------------------------------------------------------------------------
# Get background mean image
# ---------------------------------------------------------------------------
bg_mean = np.mean(
    [f.astype(np.float32) for f in baseline], axis=0
).astype(np.uint8)
cv2.imwrite("outputs/bg_mean.png", bg_mean)
print("Background mean saved to outputs/bg_mean.png")

# ---------------------------------------------------------------------------
# Run detection on eval frames
# ---------------------------------------------------------------------------
raw_masks = []
for frame in eval_frames:
    # MOG2 mask
    fg_mog = mog2.apply(frame)
    _, fg_mog = cv2.threshold(fg_mog, 200, 255, cv2.THRESH_BINARY)

    # Frame diff from bg mean
    diff = cv2.absdiff(frame, bg_mean)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, fg_diff = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)

    # Combine both
    combined = cv2.bitwise_and(fg_mog, fg_diff)
    raw_masks.append(combined)

print(f"Generated {len(raw_masks)} raw masks")

# ---------------------------------------------------------------------------
# Temporal persistence filter
# ---------------------------------------------------------------------------
h, w = raw_masks[0].shape
counter = np.zeros((h, w), dtype=np.int32)
filtered_masks = []

for mask in raw_masks:
    active = (mask > 0).astype(np.int32)
    counter = (counter + active) * active
    persistent = (counter >= 20).astype(np.uint8) * 255
    filtered_masks.append(persistent)

print("Temporal filter applied")

# ---------------------------------------------------------------------------
# Save visual results at 3 timepoints
# ---------------------------------------------------------------------------
os.makedirs("outputs", exist_ok=True)

for idx in [10, 50, 150]:
    if idx >= len(eval_frames):
        continue
    frame = eval_frames[idx]
    raw  = raw_masks[idx]
    filt = filtered_masks[idx]

    vis_raw  = overlay_mask(frame, raw,  color=(0, 255, 100))
    vis_filt = overlay_mask(frame, filt, color=(0, 100, 255))
    combined_vis = np.hstack([frame, vis_raw, vis_filt])

    out_path = f"outputs/frame_{idx:03d}_comparison.png"
    cv2.imwrite(out_path, combined_vis)
    print(f"Saved {out_path}")

# ---------------------------------------------------------------------------
# Final union mask
# ---------------------------------------------------------------------------
union_mask = np.zeros_like(filtered_masks[0])
for m in filtered_masks:
    union_mask = cv2.bitwise_or(union_mask, m)

cv2.imwrite("outputs/final_union_mask.png", union_mask)
print("Final union mask saved to outputs/final_union_mask.png")

# ---------------------------------------------------------------------------
# Quick evaluation against known ground truth
# ---------------------------------------------------------------------------
from utils.metrics import evaluate

gt_mask = np.zeros((h, w), dtype=np.uint8)
gt_mask[300:380, 280:360] = 255  # known object location from make_test_video.py

metrics = evaluate(union_mask, gt_mask)
print(f"\nResults vs ground truth:")
print(f"  Precision : {metrics['precision']}")
print(f"  Recall    : {metrics['recall']}")
print(f"  F1        : {metrics['f1']}")
print(f"  TP={metrics['TP']}  FP={metrics['FP']}  FN={metrics['FN']}")