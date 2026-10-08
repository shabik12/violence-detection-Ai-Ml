#!/usr/bin/env python3
"""
Simple YOLO loader that bypasses dataset requirements
"""

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

def load_yolo_simple():
    """Load YOLO model without dataset dependencies"""
    
    # Add DetectionModel to safe globals
    torch.serialization.add_safe_globals([
        DetectionModel,
        'ultralytics.nn.tasks.DetectionModel'
    ])
    
    # Override torch.load temporarily
    original_torch_load = torch.load
    def safe_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    
    torch.load = safe_torch_load
    
    try:
        # Load YOLO model directly
        model = YOLO('yolov8n.pt')
        
        # Force CPU
        device = torch.device('cpu')
        model.to(device)
        model.eval()
        
        # Restore torch.load
        torch.load = original_torch_load
        
        print("✅ YOLO loaded successfully!")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load YOLO: {e}")
        torch.load = original_torch_load
        return None

if __name__ == "__main__":
    model = load_yolo_simple()
    if model:
        print("🎯 YOLO ready for object detection!")
    else:
        print("❌ YOLO loading failed")
