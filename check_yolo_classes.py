#!/usr/bin/env python3
"""
Check YOLOv8n Class Names and IDs
"""

from ultralytics import YOLO
import torch

def check_yolo_classes():
    print("🔍 Checking YOLOv8n Class Names and IDs")
    print("=" * 60)

    # Load model
    try:
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
            model = YOLO('yolov8n.pt')
            print("✅ Model loaded successfully")

            # Print all class names and IDs
            print("\n📋 All YOLOv8n Classes:")
            print("-" * 40)

            knife_found = False
            for class_id, class_name in model.names.items():
                print(f"ID {class_id:3d}: {class_name}")
                if 'knife' in class_name.lower() or 'blade' in class_name.lower() or 'sword' in class_name.lower():
                    knife_found = True
                    print(f"  ⚔️  WEAPON DETECTED!")

            if not knife_found:
                print("\n❌ No knife/weapon classes found in YOLOv8n!")
                print("\n💡 This explains why knife detection isn't working.")
                print("📝 YOLOv8n is trained on COCO dataset which doesn't include knives")
                print("🔧 For knife detection, you need:")
                print("   1. Custom training with knife images")
                print("   2. Or use a different model that includes knives")

            # Check specific IDs we were using
            print("\n🔍 Checking our configured IDs:")
            configured_ids = [0, 48, 49, 50, 51, 67]
            for class_id in configured_ids:
                if class_id in model.names:
                    print(f"ID {class_id}: {model.names[class_id]} ✅")
                else:
                    print(f"ID {class_id}: NOT FOUND ❌")

        finally:
            torch.load = original_load

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_yolo_classes()
