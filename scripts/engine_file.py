from ultralytics import YOLO

# Load your PyTorch model
model = YOLO('weapon.pt')

# Export the model to TensorRT format
model.export(format='engine', device=0)  # device=0 for first GPU