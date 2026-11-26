import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from pathlib import Path
import random
import sys
from tqdm import tqdm

from unet_flat import UNetFlat
from pure_cnn_flat import PureCNNFlat

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mr_tae_fusion.data.dataset import PDSignalDataset
from mr_tae_fusion.config import get_config

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class BaselineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        
    def forward(self, pred, target):
        return 0.5 * self.mse(pred, target) + 0.5 * self.l1(pred, target)

def _get_composite_noise_type(*args, **kwargs):
    return 'composite'

def override_curriculum_to_flat(dataset):
    """Override dataset curriculum to be flat [-20, 15] dB and use composite noise."""
    dataset.curriculum.config.snr_range_phase1 = (-20, 15)
    dataset.curriculum.config.snr_range_phase2 = (-20, 15)
    dataset.curriculum.config.snr_range_phase3 = (-20, 15)
    dataset.curriculum.get_noise_type = _get_composite_noise_type

def train(model_name: str, epochs: int = 100, batch_size: int = 32):
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_name == "unet":
        model = UNetFlat().to(device)
    elif model_name == "cnn":
        model = PureCNNFlat().to(device)
    else:
        raise ValueError(f"Unknown baseline: {model_name}")
        
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = BaselineLoss()
    
    config = get_config()
    config.training.total_epochs = epochs
    
    print(f"Initializing synthetic dataset for {model_name}...")
    train_dataset = PDSignalDataset(config, num_samples=10000, mode='train')
    override_curriculum_to_flat(train_dataset)
    val_dataset = PDSignalDataset(config, num_samples=2000, mode='val')
    override_curriculum_to_flat(val_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    best_val_loss = float('inf')
    save_dir = Path("results") / f"baseline_{model_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        train_dataset.update_epoch(epoch)
        val_dataset.update_epoch(epoch)
        
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            noisy, clean = batch['noisy'].to(device), batch['clean'].to(device)
            
            optimizer.zero_grad()
            pred = model(noisy)
            loss = criterion(pred, clean)
            loss.backward()
            
            # Gradient clipping as standard
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                noisy, clean = batch['noisy'].to(device), batch['clean'].to(device)
                pred = model(noisy)
                loss = criterion(pred, clean)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir / "best.pt")
            print(f"Saved new best model to {save_dir / 'best.pt'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["unet", "cnn"], required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    train(args.model, args.epochs)
