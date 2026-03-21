import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
import numpy as np

# Core Characters for Colombian Plates
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}

class LPRNetDataset(Dataset):
    def __init__(self, data_dir, img_size=(94, 24), transform=None):
        """
        Args:
            data_dir (string): Path to the curated_dataset directory.
            img_size (tuple): Width, Height for LPRNet expected input.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.transform = transform
        self.samples = []
        
        # Parse the dataset directory
        # curated_dataset/NAME/NAME_X.jpg
        for label_dir in os.listdir(data_dir):
            dir_path = os.path.join(data_dir, label_dir)
            if not os.path.isdir(dir_path):
                continue
                
            label = label_dir.upper().replace("-", "").replace(" ", "")
            
            # Skip invalid labels or empty folders
            if label == "UNKNOWN" or len(label) == 0:
                continue
                
            for img_name in os.listdir(dir_path):
                img_path = os.path.join(dir_path, img_name)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((img_path, label))
                    
        print(f"[Dataset] Found {len(self.samples)} valid plate crops.")

    def __len__(self):
        return len(self.samples)

    def encode_label(self, label):
        """Converts string label 'NEM621' into a list of integers [13, 4, 12, 32, 28, 27]"""
        return [CHARS_DICT[c] for c in label if c in CHARS_DICT]

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image via OpenCV
        img = cv2.imread(img_path)
        if img is None:
            # Fallback for corrupted images
            img = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
            label = ""
            
        # Resize to exactly 94x24 (LPRNet standard)
        img = cv2.resize(img, self.img_size)
        
        # Data Augmentation (Random shifts, blurs, crops to prevent overfitting)
        if self.transform is not None:
            img = self.transform(img)
            
        # Convert BGR (OpenCV) to RGB, normalize, and convert to Tensor
        img = img.astype('float32')
        # Normalize to [0,1]
        img -= np.amin(img)
        img /= np.amax(img)
        img = np.transpose(img, (2, 0, 1)) # HWC to CHW
        
        target = self.encode_label(label)
        target_len = len(target)
        
        return torch.tensor(img), torch.tensor(target, dtype=torch.long), torch.tensor(target_len, dtype=torch.long)
