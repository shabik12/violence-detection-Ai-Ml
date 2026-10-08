#!/usr/bin/env python3
"""
Simple YOLO Inference - Based on Working Implementation
Author: Senior Computer Vision Engineer

Clean YOLOv8 inference for person, knife, and cell phone detection.
"""

import torch
import os
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from typing import List, Dict, Union

# YOLOv8n class IDs for target objects
CLASS_MAPPING = {
    0: 'person',
    49: 'knife',
    67: 'cell phone',
}

ALLOWED_CLASSES = list(CLASS_MAPPING.keys())

def load_yolo_model(model_path: str = 'yolov8n.pt'):
    """
    Load YOLO model safely without triggering training

    Args:
        model_path: Path to YOLO model file

    Returns:
        YOLO model or None if failed
    """
    try:
        print(f"🚀 Loading YOLO model: {model_path}")

        # Setup PyTorch safety
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

        # Override torch.load temporarily
        original_load = torch.load
        def safe_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False
            kwargs['map_location'] = 'cpu'
            return original_load(*args, **kwargs)

        torch.load = safe_torch_load

        try:
            # Clear environment to prevent auto-training
            os.environ['ULTRALYTICS_DATASETS_DIR'] = ''
            os.environ['HOME'] = ''

            # Load model
            model = YOLO(model_path)
            model.to('cpu')
            model.eval()

            print("✅ YOLO model loaded successfully")
            return model

        finally:
            # Restore original torch.load
            torch.load = original_load

    except Exception as e:
        print(f"❌ Failed to load YOLO model: {e}")
        return None

def detect_objects(frame: Union[np.ndarray, str], model=None) -> List[Dict[str, Union[str, float]]]:
    """
    Detect objects in frame and return filtered results

    Args:
        frame: Input frame (numpy array or image path)
        model: Pre-loaded YOLO model (optional)

    Returns:
        List of detected objects with label and confidence
        Format: [{"label": "knife", "confidence": 0.87}, ...]
    """
    # Load model if not provided
    if model is None:
        model = load_yolo_model()
        if model is None:
            return []

    try:
        # Handle different input types
        if isinstance(frame, str):
            # Load image from path
            img = cv2.imread(frame)
            if img is None:
                print(f"❌ Could not load image: {frame}")
                return []
            frame = img

        # Run inference
        results = model.predict(source=frame, verbose=False)

        # Process and filter results
        detections = []

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Get class ID and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    # Filter for allowed classes
                    if class_id in ALLOWED_CLASSES:
                        label = CLASS_MAPPING.get(class_id, f"class_{class_id}")

                        # Only include detections with reasonable confidence
                        if confidence > 0.3:  # 30% confidence threshold
                            detections.append({
                                "label": label,
                                "confidence": round(confidence, 3)
                            })

                            # Debug output
                            print(f"🎯 Detected: {label} (confidence: {confidence:.3f})")

        return detections

    except Exception as e:
        print(f"❌ Detection failed: {e}")
        return []

def get_supported_classes():
    """Get list of supported object classes"""
    return list(CLASS_MAPPING.values())

def test_yolo_detector():
    """Test the YOLO detector"""
    print("🧪 Testing YOLO Detector")
    print("=" * 50)

    model = load_yolo_model()

    if model is None:
        print("❌ Model loading failed")
        return

    print(f"✅ Supported classes: {get_supported_classes()}")

    # Test with dummy image
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detect_objects(dummy_frame, model)

    print(f"📊 Test result: {len(detections)} objects detected")
    for detection in detections:
        print(f"   - {detection['label']}: {detection['confidence']:.3f}")

if __name__ == "__main__":
    test_yolo_detector()
