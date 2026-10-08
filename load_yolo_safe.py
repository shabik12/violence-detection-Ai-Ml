import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# Allow DetectionModel as safe global to fix unpickling error (PyTorch 2.6+)
try:
    torch.serialization.add_safe_globals([DetectionModel])
except AttributeError:
    print("Using fallback for older PyTorch version")

# Override torch.load to handle weights_only parameter
original_torch_load = torch.load
def safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

torch.load = safe_torch_load

def load_yolo_model(weights_path):
    """
    Safely load YOLO model (v5 or v8) in PyTorch 2.0+

    Args:
        weights_path (str): Path to YOLO weights file (.pt)
    """
    try:
        # Load the model
        model = YOLO(weights_path)
        print("Model loaded successfully!")
        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    finally:
        # Restore original torch.load
        torch.load = original_torch_load

# Example usage
if __name__ == "__main__":
    # Specify your model path here
    weights_path = "yolov8n.pt"  # or "yolov5s.pt"

    # Load the model
    model = load_yolo_model(weights_path)

    if model:
        print(f"Model type: {type(model).__name__}")
        print(f"Model classes: {len(model.names)}")
