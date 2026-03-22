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

def run_subtractor(subtractor, name):
    for frame in baseline:
        subtractor.apply(frame)

    raw_masks = []
    for frame in eval_frames:
        fg = subtractor.apply(frame)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        diff = cv2.absdiff(frame, bg_mean)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, fg_diff = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
        raw_masks.append(cv2.bitwise_and(fg, fg_diff))

    counter = np.zeros((h, w), dtype=np.int32)
    filtered = []
    for mask in raw_masks:
        active = (mask > 0).astype(np.int32)
        counter = (counter + active) * active
        filtered.append((counter >= 10).astype(np.uint8) * 255)

    union = np.zeros((h, w), dtype=np.uint8)
    for m in filtered:
        union = cv2.bitwise_or(union, m)

    metrics = evaluate(union, gt_mask)
    print(f"{name:<10} Precision={metrics['precision']}  Recall={metrics['recall']}  F1={metrics['f1']}")
    return union

print("Comparing MOG2 vs KNN:\n")

mog2 = cv2.createBackgroundSubtractorMOG2(
    history=200, varThreshold=16.0, detectShadows=False
)
knn = cv2.createBackgroundSubtractorKNN(
    history=200, dist2Threshold=400.0, detectShadows=False
)

run_subtractor(mog2, "MOG2")
run_subtractor(knn,  "KNN")