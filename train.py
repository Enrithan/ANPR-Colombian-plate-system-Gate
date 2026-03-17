from ultralytics import YOLO
import torch

# ─── SAFE GPU CONFIGURATION (RTX 4080 12GB Laptop) ──────────────────────────
GPU_MEMORY_FRACTION = 0.60

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, device=0)
    torch.backends.cudnn.benchmark = False
    print(f"[GPU] CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"[GPU] Memory cap set to {int(GPU_MEMORY_FRACTION * 100)}% of VRAM")
    DEVICE = 0
else:
    print("[GPU] No CUDA device found — falling back to CPU")
    DEVICE = 'cpu'
# ─────────────────────────────────────────────────────────────────────────────

# REQUIRED on Windows: multiprocessing workers must be launched from __main__
if __name__ == '__main__':
    model = YOLO('models/anpr-demo-model.pt')

    model.train(
        data='dataset/data.yaml',
        epochs=50,
        imgsz=640,
        device=DEVICE,
        batch=12,
        workers=2,
        amp=True,
        patience=10,
        cache=False,
        name='colombian_plates'
    )
