import time
import requests
from .interfaces import IGateController

class HTTPRelayGateController(IGateController):
    def __init__(self, trigger_url: str, cooldown_seconds: int = 10):
        self.trigger_url = trigger_url
        self.cooldown = cooldown_seconds
        self.last_triggered = 0.0

    def open_gate(self) -> bool:
        now = time.time()
        if (now - self.last_triggered) < self.cooldown:
            print(f"Gate is on cooldown for {int(self.cooldown - (now - self.last_triggered))} more seconds.")
            return False
            
        print(f"Opening physical gate via HTTP to: {self.trigger_url}")
        
        try:
            # Example HTTP trigger, adjust based on actual relay api
            response = requests.get(self.trigger_url, timeout=3)
            if response.status_code == 200:
                print(">>> GATE OPENED SUCCESSFULLY <<<")
                self.last_triggered = now
                return True
            else:
                print(f"Failed to open gate. Relay returned: {response.status_code}")
                return False
        except requests.RequestException as e:
            print(f"Error communicating with gate relay: {e}")
            return False

class MockGateController(IGateController):
    def __init__(self, cooldown_seconds: int = 10):
        self.cooldown = cooldown_seconds
        self.last_triggered = 0.0
        
    def open_gate(self) -> bool:
        now = time.time()
        if (now - self.last_triggered) < self.cooldown:
            return False
            
        print("\n" + "="*40)
        print("MOCK GATE CONTROLLER: >>> GATE OPENING! <<<")
        print("="*40 + "\n")
        self.last_triggered = now
        return True
