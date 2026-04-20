import cv2
import re
from collections import Counter
import sys
import os
import time
import logging
import threading
import queue

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the root util is found for get_car
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util import get_car

from config.app_config import AppConfig
from vision.interfaces import (IVideoSource, IVehicleDetector, 
                             IVehicleTracker, IPlateDetector, IPlateReader)
from services.interfaces import IAuthorizationService, IGateController

class GateEntrySystem:
    AUTH_CACHE_TTL = 30  # seconds
    MAX_CACHE_SIZE = 1000
    
    # Colombian Plate Patterns (Private, Public, Motorcycle)
    # ⚡ Bolt: Single combined regex pattern for 4x faster validation
    # instead of looping over multiple pre-compiled regex objects.
    COLOMBIAN_PATTERN_COMBINED = re.compile(r'^([A-Z]{3}[0-9]{3}|[A-Z]{3}[0-9]{2}[A-Z]|[A-Z]{2}[0-9]{4})$')

    def __init__(
        self,
        config: AppConfig,
        video_source: IVideoSource,
        vehicle_detector: IVehicleDetector,
        vehicle_tracker: IVehicleTracker,
        plate_detector: IPlateDetector,
        plate_reader: IPlateReader,
        auth_service: IAuthorizationService,
        gate_controller: IGateController
    ):
        self.config = config
        self.video_source = video_source
        self.vehicle_detector = vehicle_detector
        self.vehicle_tracker = vehicle_tracker
        self.plate_detector = plate_detector
        self.plate_reader = plate_reader
        self.auth_service = auth_service
        self.gate_controller = gate_controller
        
        # State & Performance Buffers
        self._auth_cache = {} 
        self._vehicle_last_read = {}  # {car_id: (text, timestamp)}
        self._vehicle_history = {}    # {car_id: [list of OCR guesses]}
        self._shared_ai_results = {"vehicles": [], "plates": []}
        self._ai_queue = queue.Queue(maxsize=1) 
        self._ocr_queue = queue.Queue(maxsize=5)
        
        # Dataset Generator
        self.save_dataset = True
        self.dataset_dir = "dataset"
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
            print(f"[System] Created dataset directory: {self.dataset_dir}")
        
        self._running = False
        self._lock = threading.Lock()

    def _get_authorized_status(self, plate_text: str) -> bool:
        """Checks cache before querying the database for authorization status."""
        now = time.monotonic()
        if plate_text in self._auth_cache:
            is_authorized, timestamp = self._auth_cache[plate_text]
            if now - timestamp < self.AUTH_CACHE_TTL:
                return is_authorized
        is_authorized = self.auth_service.is_authorized(plate_text)
        if len(self._auth_cache) >= self.MAX_CACHE_SIZE:
            self._auth_cache.clear()
        self._auth_cache[plate_text] = (is_authorized, now)
        return is_authorized

    def _validate_pattern(self, plate_text: str) -> bool:
        """Returns True if plate matches one of the Colombian patterns."""
        clean_text = plate_text.replace("-", "").replace(" ", "").upper()
        # ⚡ Bolt: Single match check against a combined regex pattern
        return bool(self.COLOMBIAN_PATTERN_COMBINED.match(clean_text))

    def _ai_worker(self):
        """Background thread for heavy YOLO detection and tracking."""
        print("[System] Background AI Detector Thread Started.")
        while self._running:
            try:
                frame = self._ai_queue.get(timeout=1.0)
                if frame is None: continue
                
                # 1. Detection & Tracking (The heavy uplift)
                detections = self.vehicle_detector.detect(frame)
                track_ids = self.vehicle_tracker.update(detections)
                plates = self.plate_detector.detect(frame)
                
                # 2. Update Shared State for UI
                with self._lock:
                    self._shared_ai_results["track_ids"] = track_ids
                    self._shared_ai_results["plates"] = plates
                
                # 3. Offload potential plates to OCR thread
                for plate in plates:
                    px1, py1, px2, py2, pconf, pcls = plate
                    
                    # Quick Geometric filter
                    pw, ph = px2-px1, py2-py1
                    # Less strict geometry filter so we don't accidentally drop weird angles
                    if pw <= 10 or ph <= 10: continue

                    car_x1, car_y1, car_x2, car_y2, car_id = get_car(plate, track_ids)
                    if car_id == -1:
                        car_id = 9999 # Allow orphaned plates if the car bounding box is tight
                        
                    if car_id not in self._vehicle_last_read:
                        if not self._ocr_queue.full():
                            crop = frame[int(py1):int(py2), int(px1):int(px2)]
                            self._ocr_queue.put_nowait((crop, car_id, plate))

                self._ai_queue.task_done()
            except queue.Empty: continue
            except Exception as e:
                logger.error(f"AI Worker Error: {e}")

    def _ocr_worker(self):
        """Background thread for the even heavier OCR processing."""
        print("[System] Background OCR Worker Started.")
        while self._running:
            try:
                item = self._ocr_queue.get(timeout=1.0)
                if item is None: continue
                crop, car_id, plate_bbox = item
                
                # Full OCR
                text, score = self.plate_reader.read_text(crop)
                
                # --- Dataset Collection ---
                if self.save_dataset:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"plate_{car_id}_{timestamp}_{text if text else 'unknown'}.jpg"
                    save_path = os.path.join(self.dataset_dir, filename)
                    cv2.imwrite(save_path, crop)
                
                if text:
                    text = text.upper().replace("-", "").replace(" ", "")
                    
                    # Store in history for voting
                    if car_id not in self._vehicle_history:
                        self._vehicle_history[car_id] = []
                    self._vehicle_history[car_id].append(text)
                    
                    # Update HUD with the latest guess (temporary)
                    self._vehicle_last_read[car_id] = (text, time.time())
                    
                    # Trigger the 'Voting' process if we have enough samples
                    if len(self._vehicle_history[car_id]) >= 3:
                        self._process_recognized_plate(car_id)
                
                self._ocr_queue.task_done()
            except queue.Empty: continue
            except Exception as e:
                logger.exception("OCR Worker Error")

    def start(self):
        print("Gate Entry System Started. Video decoupled from AI for 30+ FPS.")
        self._running = True
        
        # Start Threads
        threading.Thread(target=self._ai_worker, daemon=True).start()
        threading.Thread(target=self._ocr_worker, daemon=True).start()
        
        while True:
            # 1. Immediate Frame Acquisition
            ret, frame = self.video_source.read_frame()
            if not ret or frame is None:
                if not self.config.is_live_camera: break
                time.sleep(0.01)
                continue
            
            display_frame = frame.copy()
            
            # 2. Draw Latest Available AI Results
            with self._lock:
                plates = self._shared_ai_results.get("plates", [])
                track_ids = self._shared_ai_results.get("track_ids", [])

            for plate in plates:
                px1, py1, px2, py2, pconf, pcls = plate
                cv2.rectangle(display_frame, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 2)
                
                # Show OCR read if we have it
                car_res = get_car(plate, track_ids)
                car_id = car_res[4]
                if car_id != -1 and car_id in self._vehicle_last_read:
                    text, last_time = self._vehicle_last_read[car_id]
                    if time.time() - last_time < 3.0:
                        cv2.putText(display_frame, text, (int(px1), int(py1) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                if car_id != -1:
                    cx1, cy1, cx2, cy2 = car_res[0:4]
                    cv2.rectangle(display_frame, (int(cx1), int(cy1)), (int(cx2), int(cy2)), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"ID: {int(car_id)}", (int(cx1), int(cy1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 3. Offload current frame to AI queue for next update
            if not self._ai_queue.full():
                try: self._ai_queue.put_nowait(frame)
                except queue.Full: pass

            # 4. Zero-Latency Display
            cv2.imshow("Gate Entry System - ANPR", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self._running = False
                break

        print("Shutting down...")
        self._running = False
        self.video_source.release()
        cv2.destroyAllWindows()

    def _process_recognized_plate(self, car_id: int):
        """Analyzes history for a specific car to pick the winning plate identification."""
        history = self._vehicle_history.get(car_id, [])
        if not history: return
        
        # 1. Filter for valid Colombian patterns
        valid_candidates = [plate for plate in history if self._validate_pattern(plate)]
        if not valid_candidates: return
            
        # 2. Pick the most frequent valid candidate (Majority Voting)
        counter = Counter(valid_candidates)
        winner, count = counter.most_common(1)[0]
        
        # 3. Decision Logic: Trigger if winner has at least 2 consistent votes
        if count >= 2:
            now = time.time()
            # Prevent double-triggering for the same car too quickly
            if car_id in self._vehicle_last_read:
                prev_text, prev_time = self._vehicle_last_read[car_id]
                if winner == prev_text and (now - prev_time < 10):
                    return

            print(f"\n[VOTER] Winner for Car {car_id}: {winner} ({count} votes)")
            self._vehicle_last_read[car_id] = (winner, now)
            self._authorize_plate(winner)

    def _authorize_plate(self, plate_text: str):
        """Final stage: query DB and trigger physical gate."""
        is_authorized = self._get_authorized_status(plate_text)
        if is_authorized:
            print(f">>> ACCESS GRANTED: {plate_text} <<<")
            self.gate_controller.open_gate()
        else:
            print(f"--- ACCESS DENIED: {plate_text} ---")
