import torch
from torchvision.ops import nms

boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=torch.float32)
scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
iou_threshold = 0.5

try:
    indices = nms(boxes, scores, iou_threshold)
    print("NMS indices:", indices)
except Exception as e:
    print("Error:", e)