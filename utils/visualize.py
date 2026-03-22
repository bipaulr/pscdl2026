import cv2
import numpy as np

def overlay_mask(frame, mask, color=(0, 255, 100), alpha=0.4):
    overlay = frame.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

def side_by_side(frame, pred_mask, gt_mask=None):
    pred_vis = overlay_mask(frame, pred_mask, color=(0, 255, 100))
    panels = [frame, pred_vis]
    if gt_mask is not None:
        gt_vis = overlay_mask(frame, gt_mask, color=(0, 100, 255))
        panels.append(gt_vis)
    return np.hstack(panels)