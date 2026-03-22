"""
PSCDL 2026 - Persistent Scene Change Detection and Localization
Baseline pipeline. Fill in TODOs after dataset drops (March 30).
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    baseline_frames: int = 150       # How many frames to use for BG modelling
    persist_thresh: int = 20         # Frames a change must persist to count
    diff_threshold: float = 30.0     # Pixel diff threshold (0-255)
    morph_kernel: int = 5            # Morphological op kernel size
    min_blob_area: int = 500         # Ignore tiny noise blobs (px²)
    gmm_history: int = 200
    gmm_var_threshold: float = 16.0
    gmm_detect_shadows: bool = False


cfg = Config()


# ---------------------------------------------------------------------------
# Step 1: Load video and split segments
# ---------------------------------------------------------------------------

def load_video_frames(video_path: str) -> list:
    """Load all frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"Loaded {len(frames)} frames from {video_path}")
    return frames


def split_segments(frames: list, baseline_count: int) -> tuple:
    """
    Split frames into baseline (clean) and evaluation (change) segments.
    In the actual dataset the split point may be labeled — adjust accordingly.
    """
    baseline = frames[:baseline_count]
    evaluation = frames[baseline_count:]
    print(f"Baseline: {len(baseline)} frames | Evaluation: {len(evaluation)} frames")
    return baseline, evaluation


# ---------------------------------------------------------------------------
# Step 2: Background modelling
# ---------------------------------------------------------------------------

def build_background_model(baseline_frames: list) -> tuple:
    """
    Fit an adaptive GMM on the baseline frames.
    Returns the trained MOG2 subtractor and a mean background image.
    """
    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=cfg.gmm_history,
        varThreshold=cfg.gmm_var_threshold,
        detectShadows=cfg.gmm_detect_shadows,
    )
    for frame in baseline_frames:
        mog2.apply(frame)

    # Mean background image for frame differencing
    bg_mean = np.mean(
        [f.astype(np.float32) for f in baseline_frames], axis=0
    ).astype(np.uint8)

    print("Background model built.")
    return mog2, bg_mean


# ---------------------------------------------------------------------------
# Step 3: Change detection (classical)
# ---------------------------------------------------------------------------

def detect_changes_classical(
    eval_frames: list,
    mog2,
    bg_mean: np.ndarray,
) -> list:
    """
    Per-frame binary foreground mask using MOG2 + frame differencing.
    Returns a list of raw binary masks (uint8, 0/255).
    """
    raw_masks = []
    for frame in eval_frames:
        # MOG2 mask
        fg_mog = mog2.apply(frame)
        _, fg_mog = cv2.threshold(fg_mog, 200, 255, cv2.THRESH_BINARY)

        # Absolute frame difference from background mean
        diff = cv2.absdiff(frame, bg_mean)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, fg_diff = cv2.threshold(
            diff_gray, cfg.diff_threshold, 255, cv2.THRESH_BINARY
        )

        # Combine: pixel must be flagged by both
        combined = cv2.bitwise_and(fg_mog, fg_diff)
        raw_masks.append(combined)

    print(f"Change detection done: {len(raw_masks)} masks generated.")
    return raw_masks


# ---------------------------------------------------------------------------
# Step 4: Temporal persistence filter
# ---------------------------------------------------------------------------

def temporal_filter(raw_masks: list, persist_thresh: int) -> list:
    """
    A pixel is only marked as a persistent change if it stays active
    for at least `persist_thresh` consecutive frames.
    """
    if not raw_masks:
        return []

    h, w = raw_masks[0].shape
    counter = np.zeros((h, w), dtype=np.int32)
    filtered = []

    for mask in raw_masks:
        active = (mask > 0).astype(np.int32)
        counter = (counter + active) * active  # resets to 0 when pixel disappears
        persistent = (counter >= persist_thresh).astype(np.uint8) * 255
        filtered.append(persistent)

    print("Temporal filtering done.")
    return filtered


# ---------------------------------------------------------------------------
# Step 5: Post-processing / morphological cleanup
# ---------------------------------------------------------------------------

def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    """
    Clean up noise with morphological ops and remove tiny blobs.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Remove blobs smaller than min_blob_area
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    clean = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= cfg.min_blob_area:
            clean[labels == i] = 255

    return clean


# ---------------------------------------------------------------------------
# Step 6: Full pipeline for one video
# ---------------------------------------------------------------------------

def predict_mask(video_path: str, output_path: Optional[str] = None) -> np.ndarray:
    """
    Full pipeline: video → final binary change mask.
    The mask is the union of persistent change regions across all eval frames.
    """
    frames = load_video_frames(video_path)
    baseline, eval_frames = split_segments(frames, cfg.baseline_frames)

    mog2, bg_mean = build_background_model(baseline)
    raw_masks = detect_changes_classical(eval_frames, mog2, bg_mean)
    filtered_masks = temporal_filter(raw_masks, cfg.persist_thresh)

    if not filtered_masks:
        h, w = frames[0].shape[:2]
        print("Warning: no persistent changes detected.")
        return np.zeros((h, w), dtype=np.uint8)

    # Union across time — any pixel that was persistently changed
    union_mask = np.zeros_like(filtered_masks[0])
    for m in filtered_masks:
        union_mask = cv2.bitwise_or(union_mask, m)

    final_mask = postprocess_mask(union_mask)

    if output_path:
        cv2.imwrite(output_path, final_mask)
        print(f"Mask saved to {output_path}")

    return final_mask


# ---------------------------------------------------------------------------
# Step 7: Batch over dataset
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from utils.metrics import evaluate

    DATA_DIR   = Path("data/videos")    # TODO: update after dataset drops
    GT_DIR     = Path("data/masks")     # TODO: update after dataset drops
    OUTPUT_DIR = Path("outputs/masks")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scores = []
    video_files = sorted(DATA_DIR.glob("*.mp4"))

    if not video_files:
        print("No videos found in data/videos/ — add some test videos to run.")
    
    for video_file in video_files:
        gt_path = GT_DIR / (video_file.stem + ".png")
        out_path = OUTPUT_DIR / (video_file.stem + "_pred.png")

        print(f"\nProcessing: {video_file.name}")
        pred = predict_mask(str(video_file), str(out_path))

        if gt_path.exists():
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            metrics = evaluate(pred, gt)
            scores.append(metrics)
            print(f"Metrics: {metrics}")
        else:
            print(f"No GT mask found for {video_file.name} — skipping eval.")

    if scores:
        avg_f1 = np.mean([s["f1"] for s in scores])
        avg_p  = np.mean([s["precision"] for s in scores])
        avg_r  = np.mean([s["recall"] for s in scores])
        print(f"\n{'='*40}")
        print(f"Mean Precision : {avg_p:.4f}")
        print(f"Mean Recall    : {avg_r:.4f}")
        print(f"Mean F1        : {avg_f1:.4f}")
        print(f"{'='*40}")