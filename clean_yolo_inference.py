#!/usr/bin/env python3
"""
Clean YOLO Loader - Inference Only, No Training
"""

import torch
import os
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

def load_yolo_inference_only():
    """Load YOLO model for inference only - no training"""

    print("🚀 Loading YOLOv8n for inference only...")

    # Setup safe loading
    try:
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
    except Exception:
        pass

    # Override torch.load
    original_torch_load = torch.load
    def safe_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        kwargs['map_location'] = 'cpu'
        return original_torch_load(*args, **kwargs)
    torch.load = safe_torch_load

    try:
        # Clear any dataset paths
        os.environ['ULTRALYTICS_DATASETS_DIR'] = ''
        os.environ['HOME'] = ''

        # Load model directly
        model = YOLO('yolov8n.pt')
        model.to('cpu')
        model.eval()

        # Test with dummy image
        import numpy as np
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = model.predict(dummy, verbose=False)

        print("✅ YOLO loaded successfully!")
        return model

    except Exception as e:
        print(f"❌ Failed to load YOLO: {e}")
        return None
    finally:
        torch.load = original_torch_load

def filter_detections(results, allowed_classes):
    """Filter YOLO results for specific classes"""
    if not results:
        return results

    # Class IDs: 0=person, 49=knife, 67=cell phone
    class_map = {0: 'person', 49: 'knife', 67: 'cell phone'}

    for result in results:
        if result.boxes is not None:
            filtered_indices = []
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                if class_id in allowed_classes:
                    filtered_indices.append(i)

            if filtered_indices:
                result.boxes = result.boxes[filtered_indices]
            else:
                result.boxes = None

    return results

if __name__ == "__main__":
    model = load_yolo_inference_only()
    if model:
        print("🎯 YOLO ready for inference!")
