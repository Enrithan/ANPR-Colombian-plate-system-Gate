from abc import ABC, abstractmethod

class IAuthorizationService(ABC):
    @abstractmethod
    def is_authorized(self, plate_text: str) -> bool:
        """Checks if a given license plate is authorized to enter."""
        pass
        
    @abstractmethod
    def add_authorized_plate(self, plate_text: str, owner_name: str = ""):
        """Adds a new plate to the authorized list."""
        pass

class IGateController(ABC):
    @abstractmethod
    def open_gate(self) -> bool:
        """Triggers the physical gate to open. Returns True on success."""
        pass
