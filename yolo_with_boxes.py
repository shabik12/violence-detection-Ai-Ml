#!/usr/bin/env python3
"""
YOLO Detector with Bounding Boxes
Returns full YOLO results for proper boxing like original app.py
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

def detect_objects_with_boxes(frame: Union[np.ndarray, str]) -> List:
    """
    Detect objects and return full YOLO results with bounding boxes
    Enhanced for continuous detection and multiple objects
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
        
        # Run inference with very low confidence to catch everything
        results = model.predict(source=frame, verbose=False, conf=0.1)  # Very low threshold
        
        # Filter results to only include our target classes
        filtered_results = []
        for result in results:
            if result.boxes is not None:
                # Create new filtered result
                filtered_boxes = []
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Only keep allowed classes with very low threshold
                    if class_id in ALLOWED_CLASSES and confidence > 0.1:
                        filtered_boxes.append(box)
                
                # Replace boxes with filtered ones
                if filtered_boxes:
                    result.boxes = filtered_boxes
                    filtered_results.append(result)
        
        # Print detections
        if filtered_results:
            for result in filtered_results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        label = CLASS_MAPPING.get(class_id, f"class_{class_id}")
                        print(f"🎯 YOLO detected: {label} ({confidence:.3f})")
        
        return filtered_results
        
    except Exception as e:
        print(f"❌ Detection failed: {e}")
        return []

def detect_objects(frame: Union[np.ndarray, str]) -> List[Dict[str, Union[str, float]]]:
    """
    Detect objects and return simplified results (for compatibility)
    
    Args:
        frame: Input frame (numpy array or image path)
        
    Returns:
        List of detected objects with label and confidence
    """
    results = detect_objects_with_boxes(frame)
    
    detections = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = CLASS_MAPPING.get(class_id, f"class_{class_id}")
                
                detections.append({
                    "label": label,
                    "confidence": round(confidence, 3)
                })
    
    return detections

def get_supported_classes():
    """Get list of supported object classes"""
    return list(CLASS_MAPPING.values())

# Test function
def test():
    """Test detector"""
    print("Testing YOLO detector with bounding boxes...")
    model = get_yolo_model()
    if model:
        print(f"✅ Model loaded, supports: {get_supported_classes()}")
        print("🔪 Knife detection with proper bounding boxes")
        print("📦 Returns full YOLO results for boxing")
    else:
        print("❌ Model failed to load")

if __name__ == "__main__":
    test()
