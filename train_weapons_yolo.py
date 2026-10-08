#!/usr/bin/env python3
"""
Train Custom YOLO Model for Weapons Detection
"""

import torch
import os
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

def train_custom_yolo():
    """Train YOLO model on weapons dataset"""

    print("🚀 Starting Custom YOLO Training for Weapons Detection")
    print("=" * 60)

    # Setup safe PyTorch loading
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
        print("✓ Safe globals configured")
    except Exception as e:
        print(f"! Safe globals failed: {e}")

    # Override torch.load temporarily
    original_torch_load = torch.load
    def safe_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        kwargs['map_location'] = 'cpu'
        return original_torch_load(*args, **kwargs)
    torch.load = safe_torch_load

    try:
        # Check dataset
        dataset_path = "datasets/weapons.yaml"
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset config not found: {dataset_path}")
            return None

        # Load pre-trained YOLOv8n model
        print("📦 Loading pre-trained YOLOv8n model...")
        model = YOLO('yolov8n.pt')

        # Train on custom dataset
        print("🎯 Training on weapons dataset...")
        print("   Classes: gun, knife, mobile")
        print("   Images: 3 training images")
        print("   This will take 10-20 minutes...")

        results = model.train(
            data=dataset_path,
            epochs=50,  # Small number for small dataset
            imgsz=640,
            batch=4,
            device='cpu',
            name='weapons_detector',
            save=True,
            plots=True,
            verbose=True
        )

        print("✅ Training completed!")
        print(f"📁 Model saved to: runs/detect/weapons_detector/")

        # Test the trained model
        print("🧪 Testing trained model...")
        model_path = "runs/detect/weapons_detector/weights/best.pt"
        if os.path.exists(model_path):
            trained_model = YOLO(model_path)

            # Test on one of your images
            test_image = "datasets/weapons/images/train/gun.png"
            if os.path.exists(test_image):
                results = trained_model.predict(test_image, verbose=False)
                print(f"🎯 Test on {test_image}:")
                for result in results:
                    boxes = result.boxes
                    if len(boxes) > 0:
                        for i, box in enumerate(boxes):
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            print(f"   Detection {i+1}: {model.names[cls]} (confidence: {conf:.2f})")
                    else:
                        print("   No detections")

        return model_path

    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None
    finally:
        # Always restore original torch.load
        torch.load = original_torch_load

if __name__ == "__main__":
    model_path = train_custom_yolo()
    if model_path:
        print(f"\n🎉 Success! Your custom model is ready: {model_path}")
        print("\n📝 Next steps:")
        print("1. Update app.py to use your custom model")
        print("2. Replace 'yolov8n.pt' with your model path")
    else:
        print("\n❌ Training failed. Check the error above.")
