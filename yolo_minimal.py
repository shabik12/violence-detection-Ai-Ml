#!/usr/bin/env python3
"""
Minimal YOLO Loader - Bypasses All Training
Direct model loading without any Ultralytics training triggers
"""

import torch
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Union

# YOLOv8n class IDs for target objects
CLASS_MAPPING = {
    0: 'person',
    43: 'knife',      # CORRECT: Knife is ID 43
    67: 'cell phone',
    # Additional related objects
    42: 'fork',       # Related utensil
    44: 'spoon',      # Related utensil
    76: 'scissors',   # Sharp object
}

ALLOWED_CLASSES = list(CLASS_MAPPING.keys())

# Global model variable
_yolo_model = None

def get_yolo_model():
    """Get or load YOLO model (singleton pattern)"""
    global _yolo_model
    if _yolo_model is None:
        try:
            print("Loading YOLO model directly...")
            
            # Fix PyTorch 2.6+ weights_only issue
            try:
                from ultralytics.nn.tasks import DetectionModel
                torch.serialization.add_safe_globals([DetectionModel])
            except Exception:
                pass
            
            # Override torch.load temporarily
            original_load = torch.load
            def safe_load(*args, **kwargs):
                kwargs['weights_only'] = False
                kwargs['map_location'] = 'cpu'
                return original_load(*args, **kwargs)
            
            torch.load = safe_load
            
            try:
                # Direct load
                _yolo_model = YOLO('yolov8n.pt')
                _yolo_model.to('cpu')
                print("✅ YOLO model loaded")
            finally:
                # Restore original
                torch.load = original_load
                
        except Exception as e:
            print(f"❌ YOLO load failed: {e}")
            _yolo_model = None
    return _yolo_model

def detect_objects(frame: Union[np.ndarray, str]) -> List[Dict[str, Union[str, float]]]:
    """
    Detect objects in frame and return filtered results
    
    Args:
        frame: Input frame (numpy array or image path)
        
    Returns:
        List of detected objects with label and confidence
        Format: [{"label": "knife", "confidence": 0.87}, ...]
    """
    model = get_yolo_model()
    if model is None:
        return []
    
    try:
        # Handle different input types
        if isinstance(frame, str):
            img = cv2.imread(frame)
            if img is None:
                return []
            frame = img
        
        # Run inference
        results = model.predict(source=frame, verbose=False)
        
        # Process and filter results
        detections = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Filter for allowed classes
                    if class_id in ALLOWED_CLASSES:
                        label = CLASS_MAPPING.get(class_id, f"class_{class_id}")
                        
                        # Only include detections with reasonable confidence
                        if confidence > 0.2:  # Lowered threshold for better detection
                            detections.append({
                                "label": label,
                                "confidence": round(confidence, 3)
                            })
        
        return detections
        
    except Exception as e:
        print(f"❌ Detection failed: {e}")
        return []

def get_supported_classes():
    """Get list of supported object classes"""
    return list(CLASS_MAPPING.values())

# Test function
def test():
    """Test the detector"""
    print("Testing minimal YOLO detector...")
    model = get_yolo_model()
    if model:
        print(f"✅ Model loaded, supports: {get_supported_classes()}")
    else:
        print("❌ Model failed to load")

if __name__ == "__main__":
    test()
