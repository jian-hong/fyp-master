import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mr_tae_fusion.models.mr_tae_fusion import MR_TAE_Fusion

def generate_n2v_mask(signal, mask_prob=0.01):
    """
    Generate Noise2Void blind-spot masking.
    We replace random points with uniform noise or adjacent values.
    """
    mask = torch.rand_like(signal) < mask_prob
    masked_signal = signal.clone()
    
    # Randomly replace masked values with uniform noise to break identity mapping
    masked_signal[mask] = torch.randn_like(masked_signal[mask])
    return masked_signal, mask

def train_noise2void(model_path="checkpoints/best_model.pt", epochs=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading Base Model from {model_path} for N2V Fine-Tuning...")
    # Initialize empty generic model for demonstration
    model = MR_TAE_Fusion().to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    except Exception as e:
        print(f"Base model not found or mismatch, proceeding with randomly initialized model. ({e})")
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()
    model.train()
    
    # Mocking real unlabeled dataset
    print("Mocking unlabelled real dataset stream...")
    real_data = [torch.randn(16, 1, 2048).to(device)] # 1 batch of real
    
    for epoch in range(epochs):
        pbar = tqdm(real_data, desc=f"N2V Epoch {epoch+1}")
        for real_signal in pbar:
            noisy, mask = generate_n2v_mask(real_signal)
            
            optimizer.zero_grad()
            # Forward pass
            pred_signal, _ = model(noisy)
            
            # N2V explicitly only backpropagates the loss on the artificially blinded spots
            if mask.sum() > 0:
                loss = criterion(pred_signal[mask], real_signal[mask])
                loss.backward()
                optimizer.step()
                pbar.set_postfix({"N2V Loss": f"{loss.item():.4f}"})
            
    Path("results/n2v").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "results/n2v/n2v_finetuned.pt")
    print("N2V Fine-tuning complete! Saved to results/n2v/n2v_finetuned.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pt")
    args = parser.parse_args()
    train_noise2void(args.model_path, args.epochs)
