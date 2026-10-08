#!/usr/bin/env python3
"""
Clean YOLO Loader - Inference Only, No Training
"""

import torch
import os
import time
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

print("=" * 50)
print("CLEAN YOLO INFERENCE-ONLY LOADER")
print("=" * 50)

# Clear all environment variables that might trigger training
os.environ.pop('ULTRALYTICS_DATASETS_DIR', None)
os.environ.pop('YOLO_CONFIG_DIR', None)
os.environ.pop('HOME', None)
os.environ['ULTRALYTICS_DATASETS_DIR'] = '/dev/null'
os.environ['YOLO_CONFIG_DIR'] = '/dev/null'

print("Starting YOLO model load...")
start_time = time.time()

try:
    # Add safe globals for PyTorch 2.0+
    import torch.nn
    torch.serialization.add_safe_globals([
        DetectionModel,
        torch.nn.modules.container.Sequential,
        torch.nn.modules.conv.Conv2d,
        torch.nn.modules.batchnorm.BatchNorm2d,
        torch.nn.modules.activation.ReLU,
        torch.nn.modules.activation.SiLU,
        torch.nn.modules.pooling.MaxPool2d,
        torch.nn.modules.upsampling.Upsample,
        torch.nn.modules.linear.Linear
    ])
    print("✓ Safe globals configured")
except Exception as e:
    print(f"! Safe globals failed: {e}")

# Override torch.load
original_torch_load = torch.load
def safe_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    kwargs['map_location'] = 'cpu'
    return original_torch_load(*args, **kwargs)

torch.load = safe_torch_load

try:
    print("Loading YOLO weights directly...")

    # Load weights directly without Ultralytics wrapper
    weights_path = 'yolov8n.pt'
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model file not found: {weights_path}")

    # Create YOLO model with explicit inference mode
    model = YOLO(weights_path)

    # Force to CPU immediately
    device = torch.device('cpu')
    model.to(device)
    model.eval()

    # Clear any training-related attributes
    model.trainer = None
    model.ckpt = None

    print("✓ Model loaded, testing inference...")

    # Quick inference test
    import numpy as np
    test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    results = model.predict(source=test_img, verbose=False, conf=0.25)

    load_time = time.time() - start_time
    print(f"✅ YOLO model loaded successfully!")
    print(f"⏱️  Load time: {load_time:.2f} seconds")
    print(f"🎯 Model classes: {len(model.names)}")
    print(f"📱 Device: {next(model.parameters()).device}")
    print(f"🔍 Inference test: {len(results[0].boxes) if results[0].boxes else 0} detections")

except Exception as e:
    print(f"❌ YOLO load failed: {e}")
finally:
    torch.load = original_torch_load

print("=" * 50)
