from ultralytics import YOLO
import torch

# --- CONFIGURATION ---
MODEL_PATH = 'custom_dataset_models/yolov3_sppu_objects_1+2+3_hands_only_1+2.pt'
DATA_YAML = 'annotated_yolo_datasets/custom_objects_1+2+3_+_hands_only_1+2/data.yaml'
IMG_SIZE = 640
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(DEVICE)

def evaluate_with_class_details():
    # 1. Load the model
    model = YOLO(MODEL_PATH)

    # 2. Run Validation
    # verbose=True ensures the per-class table is printed to the console
    # save_json=True can be added if you want a detailed file output
    print(f"\n--- Starting Validation for {MODEL_PATH} ---")
    results = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        device=DEVICE,
        split='val',  # Use 'test' or 'val' based on your dataset
        verbose=True   # This gives you the per-class metrics table
    )

    # 3. Extract Timing Metrics
    # These are averages calculated during the validation run
    speed = results.speed
    preprocess = speed['preprocess']
    inference = speed['inference']
    postprocess = speed['postprocess']
    total_latency = preprocess + inference + postprocess

    # 4. Final Timing Summary
    print("\n" + "="*50)
    print("SPEED PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Pre-process:  {preprocess:.2f} ms")
    print(f"Inference:    {inference:.2f} ms")
    print(f"Post-process: {postprocess:.2f} ms")
    print(f"Total Latency:{total_latency:.2f} ms per image")
    print(f"Actual FPS:   {1000/total_latency:.1f}")
    print("="*50)

if __name__ == "__main__":
    evaluate_with_class_details()