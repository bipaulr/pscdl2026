import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from segment_anything import sam_model_registry, SamPredictor


def load_grounding_dino(
    config_path="models/GroundingDINO_SwinT_OGC.py",
    checkpoint_path="models/groundingdino_swint_ogc.pth",
    device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    from groundingdino.util.inference import load_model
    model = load_model(config_path, checkpoint_path, device=device)
    print(f"Grounding DINO loaded on: {device}")
    return model


def load_sam(checkpoint="models/sam_vit_b.pth", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
    sam.to(device)
    print(f"SAM loaded on: {device}")
    return SamPredictor(sam)


def run_grounded_sam(frame_bgr, dino_model, sam_predictor,
                     text_prompt="bag . box . object . debris . luggage",
                     box_threshold=0.3, text_threshold=0.25):
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    transform = T.Compose([
        T.Resize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    pil_img = Image.fromarray(frame_rgb)
    img_tensor = transform(pil_img).unsqueeze(0)

    device = next(dino_model.parameters()).device
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = dino_model(img_tensor, captions=[text_prompt])

    logits = outputs["pred_logits"].sigmoid()[0]
    boxes  = outputs["pred_boxes"][0]
    scores = logits.max(dim=-1).values
    keep   = scores > box_threshold

    if keep.sum() == 0:
        print("Grounding DINO: no objects detected above threshold")
        return np.zeros((h, w), dtype=np.uint8)

    filtered_boxes = boxes[keep].cpu().numpy()

    union_mask = np.zeros((h, w), dtype=np.uint8)
    sam_predictor.set_image(frame_rgb)

    for box in filtered_boxes:
        cx, cy, bw, bh = box
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        bbox = np.array([x1, y1, x2, y2])
        masks, scores_sam, _ = sam_predictor.predict(
            box=bbox,
            multimask_output=True
        )
        best = masks[np.argmax(scores_sam)].astype(np.uint8) * 255
        union_mask = cv2.bitwise_or(union_mask, best)

    return union_mask