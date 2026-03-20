import os
import time
import subprocess

CURATED_DIR = r"D:\recoveryuserorchi\documentos proyectos 2025\ANPR colombian plate\curated_dataset"
TRACKER_FILE = "last_trained_count.txt"
TRAIN_SCRIPT = r"D:\recoveryuserorchi\documentos proyectos 2025\ANPR colombian plate\LPRNet_Training_Pipeline\train.py"
PYTHON_EXE = r"D:\recoveryuserorchi\documentos proyectos 2025\ANPR colombian plate\venv\Scripts\python.exe"
THRESHOLD = 50

def get_total_images():
    count = 0
    if not os.path.exists(CURATED_DIR):
        return count
    for root, dirs, files in os.walk(CURATED_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                count += 1
    return count

def main():
    print("[AutoTrainer] Starting Continuous MLOps Background Monitor...")
    print(f"[AutoTrainer] Will compile new LPRNet model upon every {THRESHOLD} new curated plates.")
    while True:
        try:
            if not os.path.exists(TRACKER_FILE):
                with open(TRACKER_FILE, "w") as f:
                    f.write(str(get_total_images()))
                    
            with open(TRACKER_FILE, "r") as f:
                last_count = int(f.read().strip() or "0")
                
            current_count = get_total_images()
            
            if current_count >= last_count + THRESHOLD:
                print(f"\n[AutoTrainer] Threshold Reached! {current_count} plates detected (Up from {last_count}).")
                print(f"[AutoTrainer] 🔥 Spawning PyTorch Native Engine...")
                
                # Execute train.py synchronously to lock until finished
                subprocess.run([PYTHON_EXE, TRAIN_SCRIPT], check=True, cwd=r"D:\recoveryuserorchi\documentos proyectos 2025\ANPR colombian plate\LPRNet_Training_Pipeline")
                
                print(f"[AutoTrainer] Training completed successfully. Updating baseline to {current_count}.")
                with open(TRACKER_FILE, "w") as f:
                    f.write(str(current_count))
                    
        except Exception as e:
            print(f"[AutoTrainer] Error Intercepted: {e}")
            
        # Scan folder for changes every 60 seconds
        time.sleep(60)

if __name__ == "__main__":
    main()
