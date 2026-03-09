from ultralytics import YOLO

# Load a model
model = YOLO("yolo/yolo12s.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="yolo/lvis.yaml", epochs=100, imgsz=640)