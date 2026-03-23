import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath('.'))

from utils.sam_refine import load_sam, refine_union_mask
from utils.evaluator import run_pipeline_on_video
from utils.metrics import evaluate

# Load video frames
cap = cv2.VideoCapture("data/test_video.mp4")
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

eval_frames = frames[150:]
h, w = eval_frames[0].shape[:2]

gt_mask = np.zeros((h, w), dtype=np.uint8)
gt_mask[300:380, 280:360] = 255

# Step 1: Get union mask from classical pipeline
print("Running classical pipeline...")
union_mask = run_pipeline_on_video("data/test_video.mp4")
m_classical = evaluate(union_mask, gt_mask)
print(f"Classical pipeline: P={m_classical['precision']}  R={m_classical['recall']}  F1={m_classical['f1']}")

# Step 2: Load SAM
print("\nLoading SAM...")
predictor = load_sam()

# Step 3: Use last eval frame as representative
rep_frame = eval_frames[-1]

# Step 4: Refine with SAM
print("Refining mask with SAM...")
refined_mask = refine_union_mask(predictor, rep_frame, union_mask)
m_sam = evaluate(refined_mask, gt_mask)
print(f"SAM refined:        P={m_sam['precision']}  R={m_sam['recall']}  F1={m_sam['f1']}")

# Step 5: Save comparison
os.makedirs("outputs", exist_ok=True)
cv2.imwrite("outputs/day4_classical_mask.png", union_mask)
cv2.imwrite("outputs/day4_sam_refined_mask.png", refined_mask)

# Side by side visual
from utils.visualize import overlay_mask
vis_classical = overlay_mask(rep_frame, union_mask, color=(0, 255, 100))
vis_sam       = overlay_mask(rep_frame, refined_mask, color=(0, 100, 255))
comparison    = np.hstack([rep_frame, vis_classical, vis_sam])
cv2.imwrite("outputs/day4_comparison.png", comparison)
print("\nOutputs saved:")
print("  outputs/day4_classical_mask.png")
print("  outputs/day4_sam_refined_mask.png")
print("  outputs/day4_comparison.png")

print(f"\nF1 change: {m_classical['f1']} → {m_sam['f1']}")