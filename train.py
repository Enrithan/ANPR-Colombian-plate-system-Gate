from ultralytics import YOLO

def train_model():
    print("[System] Loading YOLOv10 Nano model (MIT License)...")
    # Load the YOLOv10 Nano model
    # Note: yolov10n is highly optimized for CPU/Edge devices
    model = YOLO('yolov10n.pt')

    print("[System] Starting Commercial-Safe Training on GPU (device 0)...")
    model.train(
        data='dataset/data.yaml',
        epochs=50,
        imgsz=1024,
        device=0,
        name='colombia_anpr_v10_mit'
    )

if __name__ == "__main__":
    train_model()