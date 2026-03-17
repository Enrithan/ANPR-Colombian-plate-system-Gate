import json
import os

class AppConfig:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self._config = {}
        self.load()

    def load(self):
        if not os.path.exists(self.config_path):
            self._create_default_config()
            
        with open(self.config_path, "r") as f:
            self._config = json.load(f)
            
        self._validate()

    def _create_default_config(self):
        default_config = {
            "camera": {
                "source": "./sample.mp4",
                "is_live": False
            },
            "models": {
                "vehicle_detector": "yolov8n.pt",
                "plate_detector": "./models/best.pt"
            },
            "gate": {
                "type": "mock",
                "trigger_url": "http://192.168.1.100/relay/open",
                "cooldown_seconds": 10
            },
            "auth": {
                "database_path": "authorized_plates.db"
            }
        }
        with open(self.config_path, "w") as f:
            json.dump(default_config, f, indent=4)

    def _validate(self):
        # Poka-Yoke: Fail fast if config is missing critical keys
        required_keys = ["camera", "models", "gate", "auth"]
        for key in required_keys:
            if key not in self._config:
                raise ValueError(f"Config Error: Missing required section '{key}'")
                
        if "source" not in self._config["camera"]:
            raise ValueError("Config Error: Missing 'camera.source' (e.g., RTSP URL or mp4 path)")

    @property
    def camera_source(self) -> str:
        return self._config["camera"]["source"]
        
    @property
    def is_live_camera(self) -> bool:
        return self._config["camera"].get("is_live", False)

    @property
    def vehicle_model_path(self) -> str:
        return self._config["models"]["vehicle_detector"]

    @property
    def plate_model_path(self) -> str:
        return self._config["models"]["plate_detector"]

    @property
    def gate_type(self) -> str:
        return self._config["gate"]["type"]
        
    @property
    def gate_trigger_url(self) -> str:
        return self._config["gate"].get("trigger_url", "")
        
    @property
    def gate_cooldown(self) -> int:
        return self._config["gate"].get("cooldown_seconds", 10)

    @property
    def db_path(self) -> str:
        return self._config["auth"].get("database_path", "authorized_plates.db")
