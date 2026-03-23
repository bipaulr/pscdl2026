import cv2
import numpy as np
from pathlib import Path
from utils.metrics import evaluate

def apply_clahe(frame):
    """Apply CLAHE histogram equalization to normalize illumination."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def run_pipeline_on_video(video_path: str, baseline_frames: int = 150,
                           persist_thresh: int = 10, diff_threshold: float = 30.0,
                           morph_kernel: int = 5, min_blob_area: int = 500,
                           use_clahe: bool = True) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) <= baseline_frames:
        h, w = frames[0].shape[:2]
        return np.zeros((h, w), dtype=np.uint8)

    if use_clahe:
        frames = [apply_clahe(f) for f in frames]

    baseline    = frames[:baseline_frames]
    eval_frames = frames[baseline_frames:]
    h, w        = eval_frames[0].shape[:2]

    # MOG2
    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=200, varThreshold=16.0, detectShadows=False
    )
    for frame in baseline:
        mog2.apply(frame)

    # Temporal median background — more robust than mean for illumination shifts
    bg_median = np.median(
        [f.astype(np.float32) for f in baseline], axis=0
    ).astype(np.uint8)

    # Detection + temporal filter
    counter = np.zeros((h, w), dtype=np.int32)
    filtered = []
    for frame in eval_frames:
        fg = mog2.apply(frame)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        diff = cv2.absdiff(frame, bg_median)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, fg_diff = cv2.threshold(diff_gray, diff_threshold, 255, cv2.THRESH_BINARY)

        combined = cv2.bitwise_and(fg, fg_diff)
        active = (combined > 0).astype(np.int32)
        counter = (counter + active) * active
        filtered.append((counter >= persist_thresh).astype(np.uint8) * 255)

    union = np.zeros((h, w), dtype=np.uint8)
    for m in filtered:
        union = cv2.bitwise_or(union, m)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, kernel)
    union = cv2.morphologyEx(union, cv2.MORPH_OPEN,  kernel)

    # Blob filter
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(union, connectivity=8)
    clean = np.zeros_like(union)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_blob_area:
            clean[labels == i] = 255

    return clean


def evaluate_dataset(video_dir: str, gt_dir: str, output_dir: str,
                     **pipeline_kwargs) -> dict:
    """
    Run pipeline on all videos in video_dir, compare against GT masks in gt_dir.
    Saves predicted masks to output_dir.
    Returns summary dict with per-video and mean metrics.
    """
    video_dir  = Path(video_dir)
    gt_dir     = Path(gt_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    video_files = sorted(video_dir.glob("*.mp4"))

    if not video_files:
        print(f"No .mp4 files found in {video_dir}")
        return {}

    print(f"Found {len(video_files)} videos\n")
    print(f"{'Video':<30} {'P':<8} {'R':<8} {'F1':<8}")
    print("-" * 55)

    for vf in video_files:
        gt_path  = gt_dir  / (vf.stem + ".png")
        out_path = output_dir / (vf.stem + "_pred.png")

        pred = run_pipeline_on_video(str(vf), **pipeline_kwargs)
        cv2.imwrite(str(out_path), pred)

        if gt_path.exists():
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            m  = evaluate(pred, gt)
            results.append({**m, "video": vf.name})
            print(f"{vf.name:<30} {m['precision']:<8} {m['recall']:<8} {m['f1']:<8}")
        else:
            print(f"{vf.name:<30} no GT found — skipping eval")

    if not results:
        return {}

    mean_p  = round(float(np.mean([r['precision'] for r in results])), 4)
    mean_r  = round(float(np.mean([r['recall']    for r in results])), 4)
    mean_f1 = round(float(np.mean([r['f1']        for r in results])), 4)

    print("-" * 55)
    print(f"{'MEAN':<30} {mean_p:<8} {mean_r:<8} {mean_f1:<8}")

    return {
        "per_video": results,
        "mean_precision": mean_p,
        "mean_recall":    mean_r,
        "mean_f1":        mean_f1
    }