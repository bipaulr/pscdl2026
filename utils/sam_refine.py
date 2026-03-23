import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor


def load_sam(checkpoint: str = "models/sam_vit_b.pth", device: str = None):
    """Load SAM model onto GPU if available."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
    sam.to(device)
    print(f"SAM loaded on: {device}")
    return SamPredictor(sam)


def mask_to_bbox(mask: np.ndarray, padding: int = 10):
    """
    Convert a binary mask to a bounding box [x1, y1, x2, y2].
    Returns None if mask is empty.
    """
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    h_img, w_img = mask.shape
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_img, x + w + padding)
    y2 = min(h_img, y + h + padding)
    return np.array([x1, y1, x2, y2])


def refine_mask_with_sam(predictor: SamPredictor,
                          frame: np.ndarray,
                          rough_mask: np.ndarray,
                          padding: int = 10) -> np.ndarray:
    """
    Given a frame and a rough binary mask from MOG2,
    use SAM to produce a clean refined mask.

    Args:
        predictor   : loaded SamPredictor
        frame       : BGR frame (H, W, 3)
        rough_mask  : binary mask (H, W) uint8 0/255
        padding     : bbox padding in pixels

    Returns:
        refined binary mask (H, W) uint8 0/255
    """
    bbox = mask_to_bbox(rough_mask, padding=padding)
    if bbox is None:
        return rough_mask  # nothing detected, return as-is

    # SAM expects RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictor.set_image(frame_rgb)

    masks, scores, _ = predictor.predict(
        box=bbox,
        multimask_output=True  # SAM returns 3 mask candidates
    )

    # Pick the mask with highest confidence score
    best_idx  = np.argmax(scores)
    best_mask = masks[best_idx].astype(np.uint8) * 255

    return best_mask


def refine_union_mask(predictor: SamPredictor,
                       representative_frame: np.ndarray,
                       union_mask: np.ndarray) -> np.ndarray:
    """
    Refine the final union mask using SAM on a representative frame.
    Use the last eval frame as representative (most likely to show the object clearly).
    """
    return refine_mask_with_sam(predictor, representative_frame, union_mask)