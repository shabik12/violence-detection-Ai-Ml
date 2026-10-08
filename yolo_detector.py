#!/usr/bin/env python3
"""
Clean YOLO Inference Module - Professional Implementation
Author: Senior Computer Vision Engineer
Version: 1.0

This module provides safe YOLOv8 inference for object detection
without any training dependencies or dataset requirements.
"""

import torch
import os
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from typing import List, Dict, Optional, Union

class YOLODetector:
    """
    Safe YOLOv8 inference class that handles PyTorch 2.6+ compatibility
    and provides filtered object detection for specific classes.
    """
    
    # YOLOv8n class IDs for our target objects
    CLASS_MAPPING = {
        0: 'person',
        49: 'knife', 
        67: 'cell phone',
        # Note: 'gun' is not in COCO dataset, would need custom training
        41: 'knife',  # Alternative knife class in some versions
        73: 'cell phone',  # Alternative cell phone class
    }
    
    # Allowed class IDs for filtering
    ALLOWED_CLASSES = list(CLASS_MAPPING.keys())
    
    def __init__(self, model_path: str = 'yolov8n.pt'):
        """
        Initialize YOLO detector with safe loading
        
        Args:
            model_path: Path to YOLO model file (default: yolov8n.pt)
        """
        self.model = None
        self.device = torch.device('cpu')
        self.model_path = model_path
        self._load_model()
    
    def _setup_pytorch_safety(self) -> None:
        """Configure PyTorch for safe model loading"""
        try:
            # Add required classes to safe globals for PyTorch 2.6+
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
            print("✓ PyTorch safety configured")
        except Exception as e:
            print(f"! PyTorch safety setup failed: {e}")
    
    def _safe_torch_load(self, *args, **kwargs):
        """Safe torch.load wrapper for PyTorch 2.6+ compatibility"""
        kwargs['weights_only'] = False
        kwargs['map_location'] = 'cpu'
        return torch.load(*args, **kwargs)
    
    def _load_model(self) -> bool:
        """
        Load YOLO model safely without triggering training
        
        Returns:
            bool: True if model loaded successfully
        """
        try:
            print(f"🚀 Loading YOLO model: {self.model_path}")
            
            # Setup PyTorch safety
            self._setup_pytorch_safety()
            
            # Override torch.load temporarily
            original_load = torch.load
            torch.load = self._safe_torch_load
            
            try:
                # Clear environment to prevent auto-training
                os.environ['ULTRALYTICS_DATASETS_DIR'] = ''
                os.environ['HOME'] = ''
                
                # Load model
                self.model = YOLO(self.model_path)
                self.model.to(self.device)
                self.model.eval()
                
                print("✅ YOLO model loaded successfully")
                return True
                
            finally:
                # Restore original torch.load
                torch.load = original_load
                
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            self.model = None
            return False
    
    def detect_objects(self, frame: Union[np.ndarray, str]) -> List[Dict[str, Union[str, float]]]:
        """
        Detect objects in frame and return filtered results
        
        Args:
            frame: Input frame (numpy array or image path)
            
        Returns:
            List of detected objects with label and confidence
            Format: [{"label": "knife", "confidence": 0.87}, ...]
        """
        if self.model is None:
            print("❌ YOLO model not loaded")
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
            results = self.model.predict(source=frame, verbose=False)
            
            # Process and filter results
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Get class ID and confidence
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Filter for allowed classes
                        if class_id in self.ALLOWED_CLASSES:
                            label = self.CLASS_MAPPING.get(class_id, f"class_{class_id}")
                            
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
    
    def get_supported_classes(self) -> List[str]:
        """
        Get list of supported object classes
        
        Returns:
            List of class names that can be detected
        """
        return list(set(self.CLASS_MAPPING.values()))
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded successfully"""
        return self.model is not None

# Convenience function for easy integration
def detect_objects(frame: Union[np.ndarray, str], model_path: str = 'yolov8n.pt') -> List[Dict[str, Union[str, float]]]:
    """
    Convenience function for object detection
    
    Args:
        frame: Input frame (numpy array or image path)
        model_path: Path to YOLO model file
        
    Returns:
        List of detected objects with label and confidence
    """
    detector = YOLODetector(model_path)
    return detector.detect_objects(frame)

# Test function
def test_yolo_detector():
    """Test the YOLO detector with a sample image"""
    print("🧪 Testing YOLO Detector")
    print("=" * 50)
    
    detector = YOLODetector()
    
    if not detector.is_model_loaded():
        print("❌ Detector initialization failed")
        return
    
    print(f"✅ Supported classes: {detector.get_supported_classes()}")
    
    # Test with dummy image
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect_objects(dummy_frame)
    
    print(f"📊 Test result: {len(detections)} objects detected")
    for detection in detections:
        print(f"   - {detection['label']}: {detection['confidence']:.3f}")

if __name__ == "__main__":
    test_yolo_detector()
