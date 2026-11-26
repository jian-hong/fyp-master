import torch
import torch.nn as nn
import torch.autograd as autograd

class Generator1D(nn.Module):
    def __init__(self, latent_dim=100, seq_len=2048):
        super().__init__()
        self.seq_len = seq_len
        self.init_len = seq_len // 8 # 256 for 2048
        self.latent_dim = latent_dim
        
        self.fc = nn.Linear(latent_dim, 128 * self.init_len)
        self.conv = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, self.init_len)
        return self.conv(x)

class Discriminator1D(nn.Module):
    def __init__(self, seq_len=2048):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * (seq_len // 8), 1)
        )
        
    def forward(self, x):
        return self.conv(x)

def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    
    fake = torch.ones(real_samples.size(0), 1, device=device)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
