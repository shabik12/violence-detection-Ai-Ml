import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# Allow YOLO model class (trusted source)
torch.serialization.add_safe_globals([DetectionModel])

# Override torch.load temporarily for YOLO loading
original_torch_load = torch.load
def safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False  # Trusted source
    return original_torch_load(*args, **kwargs)

torch.load = safe_torch_load

# Load model
model = YOLO("yolov8n.pt")

# Restore original torch.load
torch.load = original_torch_load

print("YOLO LOADED SUCCESSFULLY")
