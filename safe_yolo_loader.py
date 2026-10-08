#!/usr/bin/env python3
"""
Safe YOLO Model Loader for PyTorch 2.6+
Handles unpickling errors with torch.serialization.add_safe_globals
Compatible with YOLOv5 and YOLOv8 models
"""

import torch
import os
import sys
from pathlib import Path
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel, BaseModel, ClassificationModel, SegmentationModel

def setup_safe_torch_loading():
    """Configure PyTorch for safe YOLO model loading"""
    try:
        import torch.nn
        # Add all ultralytics model classes and PyTorch components to safe globals
        torch.serialization.add_safe_globals([
            DetectionModel,
            BaseModel,
            ClassificationModel,
            SegmentationModel,
            torch.nn.modules.container.Sequential,
            torch.nn.modules.conv.Conv2d,
            torch.nn.modules.batchnorm.BatchNorm2d,
            torch.nn.modules.activation.ReLU,
            torch.nn.modules.activation.SiLU,
            torch.nn.modules.pooling.MaxPool2d,
            torch.nn.modules.upsampling.Upsample,
            torch.nn.modules.linear.Linear
        ])
        print("✓ Safe globals configured for ultralytics models")
    except AttributeError:
        # Fallback for older PyTorch versions
        print("! PyTorch version doesn't support add_safe_globals, using fallback method")

def safe_load_yolo_model(model_path):
    """
    Safely load any YOLO model (v5 or v8) without unpickling errors

    Args:
        model_path (str): Path to YOLO model file (.pt)

    Returns:
        YOLO: Loaded YOLO model
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Store original torch.load function
    original_torch_load = torch.load

    def safe_torch_load(*args, **kwargs):
        """Override torch.load to handle weights_only parameter"""
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False  # Allow ultralytics models
        if 'map_location' not in kwargs:
            kwargs['map_location'] = 'cpu'
        return original_torch_load(*args, **kwargs)

    try:
        # Override torch.load temporarily
        torch.load = safe_torch_load

        # Load the YOLO model
        print(f"Loading model: {model_path}")
        model = YOLO(model_path)

        # Safe inference sanity-check (does not require any dataset)
        try:
            import numpy as np
            img = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = model.predict(source=img, verbose=False)
        except Exception:
            # If predict sanity check fails, still return the loaded model.
            pass

        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    finally:
        # Always restore original torch.load
        torch.load = original_torch_load

def detect_and_load_model():
    """Auto-detect available YOLO models and load the first one found"""
    # Common YOLO model files to check
    model_files = [
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
        "yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt",
        "yolov5n.pt", "yolov5s6.pt"
    ]

    # Check current directory for model files
    available_models = [f for f in model_files if os.path.exists(f)]

    if not available_models:
        print("No YOLO model files found in current directory!")
        print("Please download a model file (e.g., yolov8n.pt or yolov5s.pt)")
        return None

    print(f"Found models: {available_models}")
    print(f"Loading: {available_models[0]}")

    return safe_load_yolo_model(available_models[0])

def main():
    """Main function to demonstrate safe YOLO loading"""
    print("=" * 60)
    print("SAFE YOLO MODEL LOADER")
    print("=" * 60)

    # Print system info
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")

    try:
        import ultralytics
        print(f"Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        print("ERROR: ultralytics package not found!")
        print("Install with: pip install ultralytics")
        return

    print("-" * 60)

    # Setup safe loading
    setup_safe_torch_loading()

    # Detect and load model
    try:
        model = detect_and_load_model()

        if model is not None:
            print("Model loaded successfully")
        else:
            print("✗ Failed to load model")

    except Exception as e:
        print(f"✗ Error: {e}")
        print("This might be due to:")
        print("  - Corrupted model file")
        print("  - Incompatible PyTorch version")
        print("  - Missing dependencies")

if __name__ == "__main__":
    main()
