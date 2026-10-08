#!/usr/bin/env python3
"""
Test Custom Weapons Model
"""

import torch
import os
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

def test_custom_model():
    """Test the custom weapons detection model"""

    print("🧪 Testing Custom Weapons Model")
    print("=" * 40)

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
        # Load custom model
        model_path = 'runs/weights/best.pt'
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return

        print(f"📦 Loading custom model: {model_path}")
        model = YOLO(model_path)

        # Test on your images
        test_images = [
            'datasets/weapons/images/train/gun.png',
            'datasets/weapons/images/train/knife.jpg',
            'datasets/weapons/images/train/mobile.png'
        ]

        print("\n🎯 Testing on your images:")
        for img_path in test_images:
            if os.path.exists(img_path):
                results = model.predict(img_path, verbose=False, conf=0.1)  # Lower confidence
                print(f"\n📸 {os.path.basename(img_path)}:")

                for result in results:
                    boxes = result.boxes
                    if len(boxes) > 0:
                        for i, box in enumerate(boxes):
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = model.names[cls]
                            print(f"   ✅ {class_name} (confidence: {conf:.2f})")
                    else:
                        print("   ❌ No detections")
            else:
                print(f"   ❌ Image not found: {img_path}")

        print(f"\n✅ Custom model test completed!")
        print(f"🎯 Model classes: {list(model.names.values())}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        torch.load = original_torch_load

if __name__ == "__main__":
    test_custom_model()
