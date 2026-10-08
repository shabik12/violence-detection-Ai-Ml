#!/usr/bin/env python3
"""
Quick YOLO Load Test - Verify if YOLO loads successfully without training
"""

import torch
import os
import time
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

print("=" * 50)
print("YOLO LOAD VERIFICATION TEST")
print("=" * 50)

# Set environment to prevent dataset loading
os.environ['ULTRALYTICS_DATASETS_DIR'] = ''
os.environ['YOLO_CONFIG_DIR'] = ''
os.environ['HOME'] = ''

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
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    if 'map_location' not in kwargs:
        kwargs['map_location'] = 'cpu'
    return original_torch_load(*args, **kwargs)

torch.load = safe_torch_load

try:
    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt')

    # Force CPU
    device = torch.device('cpu')
    model.to(device)
    model.eval()

    # Test inference
    import numpy as np
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model.predict(source=dummy_frame, verbose=False)

    load_time = time.time() - start_time
    print(f"✅ YOLO model loaded successfully!")
    print(f"⏱️  Load time: {load_time:.2f} seconds")
    print(f"🎯 Model classes: {len(model.names)}")
    print(f"📱 Device: {next(model.parameters()).device}")

except Exception as e:
    print(f"❌ YOLO load failed: {e}")
finally:
    torch.load = original_torch_load

print("=" * 50)
