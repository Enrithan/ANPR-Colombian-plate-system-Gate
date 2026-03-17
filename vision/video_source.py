import cv2
import time
from .interfaces import IVideoSource

class RTSPVideoSource(IVideoSource):
    def __init__(self, source_url: str):
        self.source_url = source_url
        self.cap = None
        self._connect()
        
    def _connect(self):
        print(f"Connecting to video stream: {self.source_url}")
        self.cap = cv2.VideoCapture(self.source_url)
        if not self.cap.isOpened():
            print(f"Warning: Failed to connect to stream {self.source_url}")
            
    def read_frame(self):
        if self.cap is None or not self.cap.isOpened():
            print("Connection lost. Reconnecting...")
            self._connect()
            time.sleep(1) # Prevent rapid spin on failure
            return False, None
            
        ret, frame = self.cap.read()
        if not ret:
            # Reconnect logic for live streams that drop out
            print("Frame dropped. Reconnecting...")
            self.release()
            self._connect()
            return False, None
            
        return ret, frame
        
    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

class MockVideoSource(IVideoSource):
    """Used for testing with an mp4 file without endless reconnects on EOF."""
    def __init__(self, source_url: str):
        self.source_url = source_url
        self.cap = cv2.VideoCapture(self.source_url)
        
    def read_frame(self):
        if not self.cap.isOpened():
            return False, None
        return self.cap.read()
        
    def release(self):
        if self.cap:
            self.cap.release()
