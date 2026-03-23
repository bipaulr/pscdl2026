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

baseline    = frames[:150]
eval_frames = frames[150:]
h, w        = eval_frames[0].shape[:2]

gt_mask = np.zeros((h, w), dtype=np.uint8)
gt_mask[300:380, 280:360] = 255

bg_mean = np.mean(
    [f.astype(np.float32) for f in baseline], axis=0
).astype(np.uint8)

# Build raw union mask (no postprocessing)
mog2 = cv2.createBackgroundSubtractorMOG2(
    history=200, varThreshold=16.0, detectShadows=False
)
for frame in baseline:
    mog2.apply(frame)

counter = np.zeros((h, w), dtype=np.int32)
filtered = []
for frame in eval_frames:
    fg = mog2.apply(frame)
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
    diff = cv2.absdiff(frame, bg_mean)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, fg_diff = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
    combined = cv2.bitwise_and(fg, fg_diff)
    active = (combined > 0).astype(np.int32)
    counter = (counter + active) * active
    filtered.append((counter >= 10).astype(np.uint8) * 255)

union = np.zeros((h, w), dtype=np.uint8)
for m in filtered:
    union = cv2.bitwise_or(union, m)

# Baseline — no postprocessing
base_metrics = evaluate(union, gt_mask)
print(f"No postprocessing:  F1={base_metrics['f1']}  P={base_metrics['precision']}  R={base_metrics['recall']}")
print()

# ---------------------------------------------------------------------------
# Sweep kernel sizes for morphological ops
# ---------------------------------------------------------------------------
print(f"{'Kernel':<8} {'Close→Open':<14} {'Open→Close':<14} {'Close only':<14} {'Open only'}")
print("-" * 65)

for k in [3, 5, 7, 9, 11]:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # Close then Open
    m1 = cv2.morphologyEx(union, cv2.MORPH_CLOSE, kernel)
    m1 = cv2.morphologyEx(m1,    cv2.MORPH_OPEN,  kernel)
    f1_co = evaluate(m1, gt_mask)['f1']

    # Open then Close
    m2 = cv2.morphologyEx(union, cv2.MORPH_OPEN,  kernel)
    m2 = cv2.morphologyEx(m2,    cv2.MORPH_CLOSE, kernel)
    f1_oc = evaluate(m2, gt_mask)['f1']

    # Close only
    m3 = cv2.morphologyEx(union, cv2.MORPH_CLOSE, kernel)
    f1_c = evaluate(m3, gt_mask)['f1']

    # Open only
    m4 = cv2.morphologyEx(union, cv2.MORPH_OPEN, kernel)
    f1_o = evaluate(m4, gt_mask)['f1']

    print(f"{k:<8} {f1_co:<14} {f1_oc:<14} {f1_c:<14} {f1_o}")

# ---------------------------------------------------------------------------
# Blob area filter sweep
# ---------------------------------------------------------------------------
print("\nBlob area filter sweep (Close→Open kernel=5):")
print(f"{'Min area':<12} {'Precision':<12} {'Recall':<10} {'F1'}")
print("-" * 45)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
morph = cv2.morphologyEx(union, cv2.MORPH_CLOSE, kernel)
morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN,  kernel)

for min_area in [0, 100, 300, 500, 800, 1200, 2000]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(morph, connectivity=8)
    clean = np.zeros_like(morph)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    m = evaluate(clean, gt_mask)
    print(f"{min_area:<12} {m['precision']:<12} {m['recall']:<10} {m['f1']}")

# ---------------------------------------------------------------------------
# Save best postprocessed mask
# ---------------------------------------------------------------------------
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
best = cv2.morphologyEx(union, cv2.MORPH_CLOSE, kernel)
best = cv2.morphologyEx(best,  cv2.MORPH_OPEN,  kernel)
cv2.imwrite("outputs/day3_best_mask.png", best)
print("\nBest mask saved to outputs/day3_best_mask.png")