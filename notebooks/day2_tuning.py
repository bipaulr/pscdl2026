import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from utils.metrics import evaluate

# Load video
cap = cv2.VideoCapture("data/test_video.mp4")
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

baseline   = frames[:150]
eval_frames = frames[150:]
h, w = eval_frames[0].shape[:2]

# Ground truth
gt_mask = np.zeros((h, w), dtype=np.uint8)
gt_mask[300:380, 280:360] = 255

# Build MOG2
mog2 = cv2.createBackgroundSubtractorMOG2(
    history=200, varThreshold=16.0, detectShadows=False
)
for frame in baseline:
    mog2.apply(frame)

bg_mean = np.mean(
    [f.astype(np.float32) for f in baseline], axis=0
).astype(np.uint8)

# Raw masks
raw_masks = []
for frame in eval_frames:
    fg_mog = mog2.apply(frame)
    _, fg_mog = cv2.threshold(fg_mog, 200, 255, cv2.THRESH_BINARY)
    diff = cv2.absdiff(frame, bg_mean)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, fg_diff = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
    raw_masks.append(cv2.bitwise_and(fg_mog, fg_diff))

# ---------------------------------------------------------------------------
# Sweep persist_thresh values and see effect on F1
# ---------------------------------------------------------------------------
print(f"{'persist_thresh':<18} {'Precision':<12} {'Recall':<10} {'F1':<8}")
print("-" * 50)

best_f1 = 0
best_thresh = 0

for thresh in [1, 3, 5, 8, 10, 15, 20, 30, 40, 50]:
    counter = np.zeros((h, w), dtype=np.int32)
    filtered = []
    for mask in raw_masks:
        active = (mask > 0).astype(np.int32)
        counter = (counter + active) * active
        filtered.append((counter >= thresh).astype(np.uint8) * 255)

    union = np.zeros((h, w), dtype=np.uint8)
    for m in filtered:
        union = cv2.bitwise_or(union, m)

    m = evaluate(union, gt_mask)
    print(f"{thresh:<18} {m['precision']:<12} {m['recall']:<10} {m['f1']:<8}")

    if m['f1'] > best_f1:
        best_f1 = m['f1']
        best_thresh = thresh

print(f"\nBest persist_thresh: {best_thresh}  →  F1: {best_f1}")