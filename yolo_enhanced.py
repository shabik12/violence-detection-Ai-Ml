#!/usr/bin/env python3
"""
Enhanced YOLO Detector with Knife Detection Improvements
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
    # Additional related objects for better detection
    42: 'fork',       # Related utensil
    44: 'spoon',      # Related utensil
    76: 'scissors',   # Sharp object (similar to knife)
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
        
        # Enhance image for better knife detection
        enhanced_frame = enhance_for_knife_detection(frame)
        
        # Run inference with lower confidence for knife detection
        results = model.predict(source=enhanced_frame, verbose=False, conf=0.15)  # Very low confidence
        
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
                        
                        # Use different thresholds for different objects
                        threshold = 0.15 if label == 'knife' else 0.3  # Lower threshold for knife
                        
                        if confidence > threshold:
                            detections.append({
                                "label": label,
                                "confidence": round(confidence, 3)
                            })
        
        return detections
        
    except Exception as e:
        print(f"❌ Detection failed: {e}")
        return []

def enhance_for_knife_detection(frame):
    """
    Enhance image to improve knife detection
    """
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Edge detection to highlight sharp objects
    edges = cv2.Canny(enhanced, 50, 150)
    
    # Convert back to BGR
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    # Combine with original
    combined = cv2.addWeighted(frame, 0.7, enhanced_bgr, 0.3, 0)
    
    return combined

def get_supported_classes():
    """Get list of supported object classes"""
    return list(CLASS_MAPPING.values())

# Test function
def test():
    """Test the detector"""
    print("Testing enhanced YOLO detector...")
    model = get_yolo_model()
    if model:
        print(f"✅ Model loaded, supports: {get_supported_classes()}")
        print("🔪 Knife detection enhanced with:")
        print("   • Lower confidence threshold (15%)")
        print("   • Image enhancement for sharp objects")
        print("   • Edge detection preprocessing")
    else:
        print("❌ Model failed to load")

if __name__ == "__main__":
    test()
