import sys
import os
os.environ["FLAGS_enable_pir_api"] = "0"

# IMPORTANT: Import PaddleOCR BEFORE YOLO/PyTorch to prevent Intel MKL thread conflicts
# The following line is commented out as PaddleOCR is no longer used.
# from vision.paddle_reader import PaddleOCRPlateReader

from config.app_config import AppConfig
from core.gate_system import GateEntrySystem

from vision.video_source import MockVideoSource, RTSPVideoSource
from vision.detectors import YOLOVehicleDetector, SORTVehicleTracker, YOLOPlateDetector
from vision.lprnet_reader import LPRNetPlateReader  # Custom PyTorch LPRNet

from services.auth_service import SQLiteAuthService
from services.gate_controller import MockGateController, HTTPRelayGateController

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

class ModelWatcher(FileSystemEventHandler):
    def __init__(self, reader):
        self.reader = reader
    def on_modified(self, event):
        # Trigger reload when the weights file is modified
        if "best_lprnet.pth" in event.src_path:
            time.sleep(1) # Wait brief moment for the fast AutoTrainer to finish its file lock
            self.reader.reload_weights()

def main():
    print("Initializing Gate Entry System Configuration...")
    # 1. Load Configuration (Poka-Yoke: validates self at startup)
    config = AppConfig("config.json")
    
    # 2. Instantiate Vision Components
    print("Loading AI Models...")
    
    if config.is_live_camera:
        video_source = RTSPVideoSource(config.camera_source)
    else:
        video_source = MockVideoSource(config.camera_source)
        
    vehicle_detector = YOLOVehicleDetector(config.vehicle_model_path)
    vehicle_tracker = SORTVehicleTracker()
    plate_detector = YOLOPlateDetector(config.plate_model_path)
    plate_reader = LPRNetPlateReader()
    
    # 2.5 Initialize Hot-Reloader Daemon Thread
    print("Initializing Hot-Reloader Daemon...")
    observer = Observer()
    observer.schedule(ModelWatcher(plate_reader), path="LPRNet_Training_Pipeline", recursive=False)
    observer.start()
    
    # 3. Instantiate External Services
    print("Loading Services...")
    auth_service = SQLiteAuthService(config.db_path)
    
    # Pre-seed auth DB for testing if using the sample mp4
    if not config.is_live_camera:
        auth_service.add_authorized_plate("NEM621", "Test Vehicle 1") # Known plate in sample.mp4
        
    if config.gate_type == "mock":
        gate_controller = MockGateController(config.gate_cooldown)
    elif config.gate_type == "http":
        gate_controller = HTTPRelayGateController(config.gate_trigger_url, config.gate_cooldown)
    else:
        raise ValueError(f"Unknown gate type in config: {config.gate_type}")
        
    # 4. Inject Dependencies into Core Orchestrator
    print("Starting Main Event Loop...")
    system = GateEntrySystem(
        config=config,
        video_source=video_source,
        vehicle_detector=vehicle_detector,
        vehicle_tracker=vehicle_tracker,
        plate_detector=plate_detector,
        plate_reader=plate_reader,
        auth_service=auth_service,
        gate_controller=gate_controller
    )
    
    # 5. Run
    system.start()

if __name__ == "__main__":
    main()