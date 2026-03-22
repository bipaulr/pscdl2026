import cv2
import numpy as np
import torch
import torchvision
from segment_anything import sam_model_registry, SamPredictor
from utils.metrics import evaluate
from utils.visualize import overlay_mask

print("OpenCV     :", cv2.__version__)
print("NumPy      :", np.__version__)
print("PyTorch    :", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("SAM        : OK")
print("Custom utils: OK")
print("\nAll imports successful.")