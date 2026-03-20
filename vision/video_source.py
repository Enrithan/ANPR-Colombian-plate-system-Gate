import cv2
import time
import threading
from .interfaces import IVideoSource

class RTSPVideoSource(IVideoSource):
    def __init__(self, source_url: str):
        self.source_url = source_url
        self.cap = None
        self.last_frame = None
        self.ret = False
        self.running = False
        self.lock = threading.Lock()
        self._connect()
        self._start_capture_thread()
        
    def _connect(self):
        print(f"[VideoSource] Connecting to: {self.source_url}")
        self.cap = cv2.VideoCapture(self.source_url)
        if not self.cap.isOpened():
            print(f"Warning: Failed to connect to stream {self.source_url}")

    def _start_capture_thread(self):
        self.running = True
        self.thread = threading.Thread(target=self._update, args=(), daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(1)
                self._connect()
                continue
            
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.last_frame = frame
            
            if not ret:
                time.sleep(0.1) # Small pause on failure

    def read_frame(self):
        with self.lock:
            return self.ret, self.last_frame
        
    def release(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

class MockVideoSource(IVideoSource):
    """Threaded version of Video File source that respects original FPS."""
    def __init__(self, source_url: str):
        self.source_url = source_url
        self.cap = cv2.VideoCapture(self.source_url)
        
        # Get original FPS to simulate real-time
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0: self.fps = 30.0
        self.frame_delay = 1.0 / self.fps
        
        self.last_frame = None
        self.ret = False
        self.running = True
        self.lock = threading.Lock()
        
        print(f"[VideoSource] Opened file '{source_url}' at {self.fps} FPS")
        
        # Start capture thread
        self.thread = threading.Thread(target=self._update, args=(), daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            start_time = time.time()
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                with self.lock:
                    self.ret = ret
                    self.last_frame = frame
                if not ret:
                    self.running = False
            
            # Precisely maintain original playback speed
            elapsed = time.time() - start_time
            sleep_time = max(0, self.frame_delay - elapsed)
            time.sleep(sleep_time)

    def read_frame(self):
        with self.lock:
            return self.ret, self.last_frame
        
    def release(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()
