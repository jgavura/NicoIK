from ultralytics import YOLO


# Load the exported ONNX model
onnx_model = YOLO("yolo/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.onnx")

# Run inference
results = onnx_model(['num_dets', 'boxes', 'scores', 'labels'], {"images": "https://ultralytics.com/images/bus.jpg"})