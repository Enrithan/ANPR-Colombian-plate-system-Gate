import cv2
import sys
import os
import time
import logging

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
        self._auth_cache = {} # Cache for authorization results: {plate_text: (is_authorized, timestamp)}

    def _get_authorized_status(self, plate_text: str) -> bool:
        """Checks cache before querying the database for authorization status."""
        now = time.monotonic()

        # 1. Check Cache
        if plate_text in self._auth_cache:
            is_authorized, timestamp = self._auth_cache[plate_text]
            if now - timestamp < self.AUTH_CACHE_TTL:
                logger.debug(f"Cache hit for plate {plate_text}")
                return is_authorized

        # 2. Database Query
        is_authorized = self.auth_service.is_authorized(plate_text)

        # 3. Update Cache (with simple size limit)
        if len(self._auth_cache) >= self.MAX_CACHE_SIZE:
            self._auth_cache.clear() # Simple eviction strategy

        self._auth_cache[plate_text] = (is_authorized, now)
        return is_authorized

    def start(self):
        print("Gate Entry System Started.")
        
        while True:
            ret, frame = self.video_source.read_frame()
            if not ret:
                if not self.config.is_live_camera:
                    break # End of video file
                continue # Keep trying for live streams
                
            # 1. Vehicle Detection
            detections = self.vehicle_detector.detect(frame)
            
            # 2. Tracking
            track_ids = self.vehicle_tracker.update(detections)
            
            # 3. Plate Detection & Reading
            plates = self.plate_detector.detect(frame)
            
            frame_h, frame_w = frame.shape[:2]

            for plate in plates:
                x1, y1, x2, y2, conf, cls = plate
                
                # ── Geometric filter: reject detections that are clearly the whole car ──
                plate_w = x2 - x1
                plate_h = y2 - y1

                # A real plate should not be larger than 35% of frame width or 20% of height
                if plate_w > frame_w * 0.35 or plate_h > frame_h * 0.20:
                    continue  # Skip — this is the whole car or a sign, not a plate

                # Colombian plates are wide: aspect ratio should be between 1.8 and 6.0
                aspect_ratio = plate_w / max(plate_h, 1)
                if not (1.8 <= aspect_ratio <= 6.0):
                    continue  # Skip — wrong shape for a plate

                # ────────────────────────────────────────────────────────────────────────

                car_x1, car_y1, car_x2, car_y2, car_id = get_car((x1, y1, x2, y2, conf, cls), track_ids)
                
                # Draw plate bounding box (red)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                
                # Crop Plate for OCR
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                
                # Read Text
                text, score = self.plate_reader.read_text(crop)
                
                if text and score > 0.4:
                    cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    self._process_recognized_plate(text, frame)
                
                if car_id != -1:
                    # Draw Vehicle Bounding Box (green)
                    cv2.rectangle(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"Car {int(car_id)}", (int(car_x1), int(car_y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Optional: Display output
            cv2.imshow("Gate Entry System - ANPR", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print("Shutting down...")
        self.video_source.release()
        cv2.destroyAllWindows()

    def _process_recognized_plate(self, plate_text: str, frame):
        print(f"Read Plate: {plate_text}")
        
        is_authorized = self._get_authorized_status(plate_text)

        if is_authorized:
            print(f"Plate {plate_text} AUTHORIZED.")
            # Action!
            self.gate_controller.open_gate()
        else:
            print(f"Plate {plate_text} DENIED.")
