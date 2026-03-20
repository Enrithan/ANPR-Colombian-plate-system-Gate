import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import LPRNetDataset, CHARS
from model import build_lprnet
import cv2
import numpy as np
import time

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Training] Using device: {device}")
    
    # 1. Dataset & Dataloader
    dataset_dir = r"D:\recoveryuserorchi\documentos proyectos 2025\ANPR colombian plate\curated_dataset"
    train_dataset = LPRNetDataset(dataset_dir, img_size=(94, 24))
    
    # Custom collate_fn to handle variable length targets for CTC Loss
    def collate_fn(batch):
        imgs, targets, target_lengths = zip(*batch)
        imgs = torch.stack(imgs, 0)
        # Flatten targets for CTCLoss
        targets = torch.cat(targets, 0)
        target_lengths = torch.stack(target_lengths, 0)
        return imgs, targets, target_lengths

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # 2. Model Initialization
    lpr_max_len = 8
    class_num = len(CHARS) + 1 # Include CTC Blank
    model = build_lprnet(lpr_max_len=lpr_max_len, class_num=class_num, dropout_rate=0.5)
    model.to(device)

    # 3. Loss & Optimizer
    # CTC Loss expects: log_probs (T, N, C), targets, input_lengths, target_lengths
    ctc_loss = nn.CTCLoss(blank=len(CHARS), reduction='mean')
    optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, eps=1e-08, momentum=0.9, weight_decay=2e-5)

    num_epochs = 100
    
    print("[Training] Starting Loop...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        start_time = time.time()
        for i, (imgs, targets, target_lengths) in enumerate(train_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            # Forward
            logits = model(imgs) # Output: (Batch, Classes, Seq_Length)
            
            # log_softmax expects (Seq_Length, Batch, Classes)
            log_probs = logits.permute(2, 0, 1) 
            log_probs = nn.functional.log_softmax(log_probs, dim=2)
            
            # Input lengths are fixed for every item in batch based on the CNN output sequence length
            input_lengths = torch.full(size=(imgs.size(0),), fill_value=log_probs.size(0), dtype=torch.long).to(device)

            # Compute CTC Loss
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
        
        # Save checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"auto_lprnet_epoch_{epoch+1}.pth")

    print("[Training] Complete! Saving best_lprnet.pth")
    torch.save(model.state_dict(), "best_lprnet.pth")

if __name__ == '__main__':
    train()
