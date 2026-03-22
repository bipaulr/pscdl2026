import numpy as np

def evaluate(pred_mask, gt_mask):
    pred = (pred_mask > 0).astype(bool)
    gt   = (gt_mask   > 0).astype(bool)

    TP = np.logical_and(pred,  gt).sum()
    FP = np.logical_and(pred, ~gt).sum()
    FN = np.logical_and(~pred,  gt).sum()

    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "precision": round(float(precision), 4),
        "recall":    round(float(recall),    4),
        "f1":        round(float(f1),        4),
        "TP": int(TP), "FP": int(FP), "FN": int(FN)
    }