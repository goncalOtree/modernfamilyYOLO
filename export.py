from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("runs/detect/train/weights/best.pt")

# Export the model
model.export(format="onnx")

