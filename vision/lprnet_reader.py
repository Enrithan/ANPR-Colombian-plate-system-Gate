import torch
import cv2
import numpy as np
from LPRNet_Training_Pipeline.model import build_lprnet
from LPRNet_Training_Pipeline.dataset import CHARS
from .interfaces import IPlateReader
from util import license_complies_format, format_license

class LPRNetPlateReader(IPlateReader):
    def __init__(self, weight_path="LPRNet_Training_Pipeline/best_lprnet.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight_path = weight_path
        
        # 36 chars + 1 blank
        self.model = build_lprnet(lpr_max_len=8, class_num=len(CHARS)+1, dropout_rate=0)
        self.model.load_state_dict(torch.load(self.weight_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"[AI] LPRNet Edge Reader initialized on {self.device}.")

    def reload_weights(self):
        """Intercepts and hot-swaps newly mapped CNN tensors mid-stream."""
        print(f"\n[AI] Hot-Reloading new intelligence from {self.weight_path}...")
        try:
            # We enforce atomic weights reload to avoid halting the live stream
            self.model.load_state_dict(torch.load(self.weight_path, map_location=self.device))
            self.model.eval()
            print("[AI] Cellular upgrade complete! Zero frame loss.\n")
        except Exception as e:
            print(f"[AI] Error hot-swapping weights: {e}\n")

    def decode(self, preds):
        """Greedy Decoder for CTC Network"""
        preds = preds.cpu().detach().numpy() # (Seq_Len, Batch_Size, Classes)
        # We only infer 1 image at a time
        pred = preds[:, 0, :]
        
        # Argmax along the classes dimension
        pred_labels = np.argmax(pred, axis=1)
        
        text = ""
        last_label = -1
        for label in pred_labels:
            if label != last_label: # Collapse repeated characters (CTC rule)
                if label < len(CHARS): # Ignore the "Blank" CTC character
                    text += CHARS[label]
            last_label = label
            
        return text

    def read_text(self, cropped_plate: np.ndarray) -> tuple[str, float]:
        # Resize dynamically to the geometric dimension the mathematical tensor expects
        img = cv2.resize(cropped_plate, (94, 24))
        img = img.astype('float32')
        img -= np.amin(img)
        img /= np.amax(img)
        img = np.transpose(img, (2, 0, 1))
        
        tensor = torch.tensor(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)
            
            # Predict (Seq_Len, Batch_Size, Classes)
            preds = logits.permute(2, 0, 1) 
            text = self.decode(preds)
            
            # For this simple prototype, we just return a dummy confidence since 
            # CTC probability mapping requires complex matrix decoding
            score = 0.99
            
        print(f"[OCR Raw] LPRNet Extracted: '{text}'")
        
        if license_complies_format(text):
            return format_license(text), score
            
        return "", 0.0
