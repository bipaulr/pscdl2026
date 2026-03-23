import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath('.'))

from utils.grounded_sam import load_grounding_dino, load_sam, run_grounded_sam
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

eval_frames = frames[150:]
h, w = eval_frames[0].shape[:2]

gt_mask = np.zeros((h, w), dtype=np.uint8)
gt_mask[300:380, 280:360] = 255

# Use last eval frame as representative
rep_frame = eval_frames[-1]

# Load models
print("Loading models...")
dino_model    = load_grounding_dino()
sam_predictor = load_sam()

# Run Grounded SAM
print("Running Grounded SAM...")
result_mask = run_grounded_sam(
    frame_bgr   = rep_frame,
    dino_model  = dino_model,
    sam_predictor = sam_predictor,
    text_prompt = "bag . box . object . debris",
    box_threshold = 0.3,
)

m = evaluate(result_mask, gt_mask)
print(f"\nGrounded SAM: P={m['precision']}  R={m['recall']}  F1={m['f1']}")

# Save outputs
os.makedirs("outputs", exist_ok=True)
cv2.imwrite("outputs/day3_grounded_sam_mask.png", result_mask)

from utils.visualize import overlay_mask
vis = overlay_mask(rep_frame, result_mask, color=(255, 100, 0))
cv2.imwrite("outputs/day3_grounded_sam_overlay.png", vis)
print("Saved outputs/day3_grounded_sam_mask.png")