import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from utils.metrics import evaluate
from utils.evaluator import run_pipeline_on_video

h, w = 480, 640
gt_mask = np.zeros((h, w), dtype=np.uint8)
gt_mask[300:380, 280:360] = 255

def apply_brightness_shift(video_path, output_path, shift):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 25, (w, h))
    for i, frame in enumerate(frames):
        # Apply shift only to eval frames (after 150)
        if i >= 150:
            frame = np.clip(frame.astype(np.int16) + shift, 0, 255).astype(np.uint8)
        out.write(frame)
    out.release()

def apply_noise(video_path, output_path, noise_std):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 25, (w, h))
    for frame in frames:
        noise = np.random.normal(0, noise_std, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        out.write(frame)
    out.release()

os.makedirs("data/stress", exist_ok=True)

print("Running robustness stress tests...\n")
print(f"{'Condition':<30} {'P':<10} {'R':<10} {'F1'}")
print("-" * 55)

# Baseline
pred = run_pipeline_on_video("data/test_video.mp4")
m = evaluate(pred, gt_mask)
print(f"{'Clean (baseline)':<30} {m['precision']:<10} {m['recall']:<10} {m['f1']}")

# Brightness shifts
for shift in [-40, -20, +20, +40]:
    path = f"data/stress/bright_{shift:+d}.mp4"
    apply_brightness_shift("data/test_video.mp4", path, shift)
    pred = run_pipeline_on_video(path)
    m = evaluate(pred, gt_mask)
    label = f"Brightness {shift:+d}"
    print(f"{label:<30} {m['precision']:<10} {m['recall']:<10} {m['f1']}")

# Gaussian noise
for std in [10, 20, 35]:
    path = f"data/stress/noise_{std}.mp4"
    apply_noise("data/test_video.mp4", path, std)
    pred = run_pipeline_on_video(path)
    m = evaluate(pred, gt_mask)
    label = f"Noise std={std}"
    print(f"{label:<30} {m['precision']:<10} {m['recall']:<10} {m['f1']}")