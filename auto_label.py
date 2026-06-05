"""
auto_label.py
=============
Pseudo-labeling pipeline for Colombian ANPR.

What this script does:
1. Reads frames from cali_traffic.mp4 (every N seconds)
2. Runs anpr-demo-model.pt on each frame
3. Saves frames + YOLO label files ONLY for detections above a confidence threshold
4. Splits results into train/valid sets
5. Writes a data.yaml file ready for train.py

Usage:
    python auto_label.py
Then run:
    python train.py
"""

import cv2
import os
import random
import shutil
from pathlib import Path
from ultralytics import YOLO
import torch

# ─── SAFE GPU CONFIGURATION (RTX 4080 12GB Laptop) ──────────────────────────
GPU_MEMORY_FRACTION = 0.60  # 60% VRAM cap — raise if stable, lower if crashing
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, device=0)
    torch.backends.cudnn.benchmark = False
    DEVICE = 0
    print(f"[GPU] Using {torch.cuda.get_device_name(0)} at {int(GPU_MEMORY_FRACTION*100)}% VRAM")
else:
    DEVICE = 'cpu'
    print("[GPU] No CUDA found — using CPU")
# ─────────────────────────────────────────────────────────────────────────────


# ─── CONFIGURATION ──────────────────────────────────────────────────────────────
VIDEO_PATH       = "cali_traffic.mp4"         # Source video
MODEL_PATH       = "models/best.pt"            # Switched to best.pt — detected plates consistently
OUTPUT_DIR       = "dataset"                   # Where to save the dataset
CONF_THRESHOLD   = 0.40                        # Lowered to capture more detections
SAMPLE_EVERY_SEC = 1                           # Extract 1 frame every N seconds
TRAIN_SPLIT      = 0.85                        # 85% train, 15% valid
CLASS_NAMES      = ["license_plate"]           # Dataset class names
# ────────────────────────────────────────────────────────────────────────────────


def create_dataset_structure(output_dir: str):
    """Create the YOLOv8 dataset folder structure."""
    for split in ["train", "valid"]:
        Path(f"{output_dir}/{split}/images").mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/{split}/labels").mkdir(parents=True, exist_ok=True)
    print(f"[✓] Created dataset structure in '{output_dir}/'")


def write_data_yaml(output_dir: str, class_names: list):
    """Write the data.yaml config file for YOLO training."""
    yaml_content = f"""train: train/images
val: valid/images

nc: {len(class_names)}
names: {class_names}
"""
    yaml_path = f"{output_dir}/data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"[✓] Wrote dataset config to '{yaml_path}'")


def extract_and_label(video_path, model_path, output_dir, conf_threshold, sample_every_sec, train_split):
    """Main pipeline: extract frames, auto-label, save to disk."""
    
    if not os.path.exists(video_path):
        print(f"[✗] Video not found: {video_path}")
        return 0
    
    print(f"[→] Loading model: {model_path}")
    model = YOLO(model_path).to(DEVICE)
    
    print(f"[→] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(fps * sample_every_sec))
    
    print(f"[i] FPS: {fps:.1f} | Total frames: {total_frames} | Sampling every {frame_interval} frames")
    print(f"[i] Confidence threshold: {conf_threshold}")
    
    saved_frames = []
    frame_idx = 0
    sampled = 0
    kept = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            sampled += 1
            
            # Run plate detection
            results = model(frame, verbose=False, conf=conf_threshold)[0]
            boxes = results.boxes
            
            if boxes is not None and len(boxes) > 0:
                h, w = frame.shape[:2]
                label_lines = []
                
                # ⚡ Bolt: Extract all bounding box data efficiently using vectorized tensor operations
                # This avoids repeated synchronous CPU-GPU data transfers caused by accessing individual `box` objects.
                boxes_data = boxes.data.cpu().numpy()
                for row in boxes_data:
                    x1, y1, x2, y2, conf, cls = row
                    conf = float(conf)
                    cls = int(cls)
                    
                    # Normalize to YOLO format (cx, cy, bw, bh) in [0..1]
                    cx = (x1 + x2) / 2 / w
                    cy = (y1 + y2) / 2 / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    
                    # Class 0 = license_plate regardless of original class index
                    label_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                
                # Save frame and labels temporarily
                frame_name = f"frame_{frame_idx:07d}"
                saved_frames.append((frame.copy(), "\n".join(label_lines), frame_name))
                kept += 1
            
            if sampled % 50 == 0:
                print(f"[→] Sampled {sampled} frames | Kept {kept} with plates...")
        
        frame_idx += 1
    
    cap.release()
    print(f"\n[✓] Done scanning. {kept} / {sampled} frames had plates above {conf_threshold} confidence.")
    
    if kept == 0:
        print("[✗] No frames kept. Try lowering CONF_THRESHOLD (e.g. 0.5).")
        return 0
    
    # Shuffle and split into train/valid
    random.shuffle(saved_frames)
    split_idx = int(len(saved_frames) * train_split)
    train_set = saved_frames[:split_idx]
    valid_set = saved_frames[split_idx:]
    
    print(f"[i] Split: {len(train_set)} train | {len(valid_set)} valid")
    
    # Write to disk
    for split_name, dataset in [("train", train_set), ("valid", valid_set)]:
        for frame, labels, name in dataset:
            img_path = f"{output_dir}/{split_name}/images/{name}.jpg"
            lbl_path = f"{output_dir}/{split_name}/labels/{name}.txt"
            cv2.imwrite(img_path, frame)
            with open(lbl_path, "w") as f:
                f.write(labels)
    
    print(f"[✓] Dataset saved to '{output_dir}/'")
    return kept


def main():
    print("=" * 60)
    print("  ANPR Colombian Plates — Auto-Labeling Pipeline")
    print("=" * 60)
    
    # Check for existing dataset
    if os.path.exists(OUTPUT_DIR):
        answer = input(f"\n[!] '{OUTPUT_DIR}/' already exists. Overwrite? (y/n): ").strip().lower()
        if answer == 'y':
            shutil.rmtree(OUTPUT_DIR)
            print(f"[✓] Removed existing '{OUTPUT_DIR}/'")
        else:
            print("[→] Appending to existing dataset.")
    
    create_dataset_structure(OUTPUT_DIR)
    write_data_yaml(OUTPUT_DIR, CLASS_NAMES)
    
    kept = extract_and_label(
        video_path=VIDEO_PATH,
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        conf_threshold=CONF_THRESHOLD,
        sample_every_sec=SAMPLE_EVERY_SEC,
        train_split=TRAIN_SPLIT
    )
    
    if kept > 0:
        print("\n" + "=" * 60)
        print(f"  ✅  Dataset ready!  {kept} labeled frames generated.")
        print("  Run 'python train.py' to start fine-tuning.")
        print("=" * 60)


if __name__ == "__main__":
    main()
