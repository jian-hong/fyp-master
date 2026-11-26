import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
from tqdm import tqdm
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mr_tae_fusion.models.wgan import Generator1D, Discriminator1D, compute_gradient_penalty
from mr_tae_fusion.data.dataset import PDSignalDataset
from mr_tae_fusion.config import get_config

def extract_pure_noise_segments(samples=5000, seq_len=2048):
    """
    Extract pure noise segments from dataset where target_seg == 0
    Returns: Tensor of shape (N, 1, seq_len)
    """
    config = get_config()
    dataset = PDSignalDataset(config, num_samples=samples, mode='train')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    noise_segments = []
    print("Extracting pure noise segments...")
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        # Noise is intrinsically (noisy - clean)
        noisy = data['noisy']
        clean = data['clean']
        pure_noise = noisy - clean
        
        # We can also check data['mask'] == 0 to explicitly grab only regions without PD, 
        # but since 'clean' only contains PD pulses, 'noisy - clean' is exactly the 
        # injected noise field across the entire 2048 window!
        # Given WGAN needs 2048 length to train, we'll use the full extracted noise.
        noise_segments.append(pure_noise)
        
    return torch.stack(noise_segments)

def train_wgan(epochs=1, batch_size=64, extract_samples=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Extract noise
    real_noise = extract_pure_noise_segments(samples=extract_samples)
    dataset = TensorDataset(real_noise)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Models
    generator = Generator1D().to(device)
    discriminator = Discriminator1D().to(device)
    
    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.0, 0.9))
    
    lambda_gp = 10
    n_critic = 5
    
    save_dir = Path("results/wgan")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"WGAN Epoch {epoch+1}/{epochs}")
        for batch_idx, (real_imgs,) in enumerate(pbar):
            real_imgs = real_imgs.to(device)
            bs = real_imgs.size(0)
            
            # --- Train Discriminator ---
            discriminator.zero_grad()
            z = torch.randn(bs, 100, device=device)
            fake_imgs = generator(z)
            
            real_validity = discriminator(real_imgs)
            fake_validity = discriminator(fake_imgs.detach())
            
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, device)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            
            d_loss.backward()
            d_optimizer.step()
            
            # --- Train Generator ---
            if batch_idx % n_critic == 0:
                generator.zero_grad()
                gen_imgs = generator(z)
                g_loss = -torch.mean(discriminator(gen_imgs))
                
                g_loss.backward()
                g_optimizer.step()
                
            pbar.set_postfix({"D Loss": f"{d_loss.item():.4f}", "G Loss": f"{g_loss.item():.4f}"})
            
    torch.save(generator.state_dict(), save_dir / "generator.pt")
    print("WGAN training finished!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    train_wgan(epochs=args.epochs, extract_samples=args.samples)
