# 🚗 ANPR Colombian Plate — Automatic Number Plate Recognition

> AI-powered license plate detection and recognition system tailored for **Colombian vehicle plates** (cars and motorcycles), designed for residential entrance gate automation.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLO-v8%2Fv11-orange?logo=opencv)
![EasyOCR](https://img.shields.io/badge/OCR-EasyOCR-green)
![CUDA](https://img.shields.io/badge/GPU-CUDA%2012.4-76B900?logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📋 Overview

This system automatically:
- **Detects** vehicles and license plates from RTSP camera streams or video files
- **Reads** Colombian plate formats: `AAA123` (cars) and `AAA12A` (motorcycles)
- **Matches** detected plates against an authorized vehicle database
- **Triggers** a gate relay controller to open/close the access gate

Designed specifically for the Colombian license plate format as mandated by the *Ministerio de Transporte de Colombia*.

---

## 🏗️ Architecture

```
Video Source (RTSP / File)
        │
        ▼
 Vehicle Detector (YOLOv8)
        │
        ▼
 Vehicle Tracker (SORT)
        │
        ▼
 Plate Detector (YOLOv8 fine-tuned)
        │
   ┌────┴────┐
   │ Filter  │ ← Size check (< 35% frame width)
   │         │ ← Aspect ratio (1.8x – 6.0x)
   └────┬────┘
        │
        ▼
  EasyOCR Reader
        │
        ▼
 Authorization DB (SQLite)
        │
        ▼
 Gate Controller (USB Relay / HTTP / Mock)
```

---

## 🇨🇴 Colombian Plate Formats Supported

| Type | Format | Example |
|---|---|---|
| Car / Taxi | `AAA000` — 3 letters + 3 digits | `EOM370` |  
| Motorcycle | `AAA00A` — 3 letters + 2 digits + 1 letter | `ABC12X` |

---

## ⚡ Features

- ✅ **GPU-accelerated** inference and training (CUDA 12.4, tested on RTX 4080 Laptop)
- ✅ **Safe GPU limits** — configurable VRAM cap to prevent BSOD on laptops
- ✅ **Auto-labeling pipeline** — extract and pseudo-label plates directly from traffic video
- ✅ **Fine-tuning workflow** — train on your own footage in ~3 minutes on a modern GPU
- ✅ **False positive filtering** — geometric validation (size + aspect ratio) rejects whole-car detections
- ✅ **Dual plate format** — handles both car and motorcycle plates
- ✅ **Clean security posture** — no hardcoded secrets, no injection vulnerabilities (Bandit SAST verified)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- NVIDIA GPU with CUDA 12.4 (optional but recommended)
- Git

### 1. Clone and set up environment
```bash
git clone https://github.com/YOUR-USERNAME/anpr-colombian-plate.git
cd anpr-colombian-plate

python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # Linux/Mac

pip install -r requirements.txt

# For GPU support (NVIDIA CUDA 12.4):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2. Add your model weights
Download or train your plate detection model and place it in:
```
models/
└── anpr-demo-model.pt    # or your custom trained model
```

> 📌 Model weights are excluded from git. Download from [Roboflow Universe](https://universe.roboflow.com) or train your own (see Training section).

### 3. Configure
Edit `config.json`:
```json
{
  "camera": {
    "source": "rtsp://admin:password@192.168.1.x:554/ch01"
  },
  "models": {
    "vehicle_detector": "yolov8n.pt",
    "plate_detector": "./models/anpr-demo-model.pt"
  }
}
```

### 4. Run
```bash
python main.py
```

---

## 🏋️ Training Your Own Model

### Auto-label from traffic video
```bash
# Step 1: Extract and auto-label plate frames from your video
python auto_label.py

# Step 2: Fine-tune the model
python train.py
```

The trained model will be saved at:
```
runs/detect/colombian_plates/weights/best.pt
```

### GPU Safety Settings (Laptop)
Edit `GPU_MEMORY_FRACTION` in both `auto_label.py` and `train.py`:
```python
GPU_MEMORY_FRACTION = 0.60  # 60% VRAM cap — safe for RTX 4080 Laptop
```
Reduce to `0.50` if you experience crashes.

---

## 📁 Project Structure

```
anpr-colombian-plate/
├── core/
│   └── gate_system.py        # Main detection + gate control loop
├── vision/
│   ├── detectors.py          # YOLO-based vehicle & plate detectors
│   └── plate_reader.py       # EasyOCR + Colombian format validation
├── services/
│   └── gate_controller.py    # USB relay / HTTP / mock gate drivers
├── config/
│   └── app_config.py         # Configuration loader
├── models/                   # Place .pt model weights here (git-ignored)
├── dataset/                  # YOLOv8 dataset structure (images git-ignored)
│   └── data.yaml
├── auto_label.py             # Auto-labeling pipeline from video
├── train.py                  # Fine-tuning script
├── main.py                   # Application entry point
├── util.py                   # Helper functions
└── config.json               # Runtime configuration
```

---

## 🔒 Security

- **SAST scan**: 0 HIGH/MEDIUM findings in source code (Bandit)
- **Dependencies**: Regularly audited with `pip-audit`
- **No hardcoded credentials** — all configuration via `config.json` or environment variables
- Plate data stored **locally only** in SQLite — no external API calls
- Complies with Colombian **Ley 1581 de 2012** (Data Protection)

---

## 🗺️ Roadmap

- [ ] Multi-camera RTSP stream support (exit / entry gate zones)
- [ ] USB relay hardware gate controller integration
- [ ] Confidence voting across frames for higher accuracy
- [ ] Web dashboard for plate management
- [ ] DVR system integration audit

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Roboflow](https://roboflow.com) — dataset management
