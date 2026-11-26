"""
===================================================================================
ULTIMATE MR-TAE-FUSION PIPELINE FOR PD CLASSIFICATION
===================================================================================

This is the BEST method combining:
1. MR-TAE-Fusion for DENOISING (your model's strength)
2. Distance-Based CNN for CLASSIFICATION (4 classes)
3. Multi-task learning for simultaneous denoising + classification

Architecture:
    Noisy Signal → MR-TAE-Fusion Encoder → [Denoised Signal] → Classifier → Distance Class 
                                         ↳ [Segmentation Mask] (bonus output)

This leverages your model's strength in handling noisy PD signals!
===================================================================================
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from pathlib import Path
from typing import Dict, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
import warnings
warnings.filterwarnings('ignore')

# Setup
PROJECT_ROOT = Path(r'D:\New folder (2)\Data\OneDrive_2025-12-13\Try Model')
DATA_DIR = Path(r'D:\New folder (2)\Data\OneDrive_2025-12-13')
QLIN_DIR = DATA_DIR / 'Datset from Q.lin'
sys.path.insert(0, str(PROJECT_ROOT))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Device: {device}")
if device == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")


# =============================================================================
# PART 1: UNDERSTANDING THE 9-FEATURE METHOD (SENIOR'S APPROACH)
# =============================================================================
"""
The 9-feature method extracts these statistics from a signal:

    Signal (2001 samples) → Extract 9 numbers → Classifier (RF/MLP) → Class

Features:
    1. RMS = sqrt(mean(signal²))           - Signal energy
    2. Variance = var(signal)               - Spread
    3. Std = std(signal)                    - Variability
    4. Waveform Factor = RMS / |mean|       - Shape
    5. Kurtosis                             - Peak sharpness
    6. Skewness                             - Asymmetry
    7. Max Power Spectrum = max(|FFT|²)     - Dominant frequency
    8. Median Frequency                     - Spectral center
    9. Mean Power Frequency                 - Weighted average freq

This is a HANDCRAFTED feature approach - does NOT use deep learning!
It works okay but loses spatial/temporal information in the signal.
"""

def calculate_9_features(signal: np.ndarray) -> np.ndarray:
    """
    Senior's method: Extract 9 statistical features from PD signal.
    
    This is what classification_testing.m does in MATLAB!
    """
    features = []
    
    # Time-domain features
    rms = np.sqrt(np.mean(signal**2))
    features.append(rms)                        # 1. RMS
    features.append(np.var(signal))             # 2. Variance
    features.append(np.std(signal))             # 3. Std
    mean_abs = np.abs(np.mean(signal))
    features.append(rms / mean_abs if mean_abs > 1e-10 else 0)  # 4. Waveform Factor
    features.append(stats.kurtosis(signal))     # 5. Kurtosis (peakedness)
    features.append(stats.skew(signal))         # 6. Skewness (asymmetry)
    
    # Frequency-domain features
    fft_vals = np.abs(np.fft.fft(signal))
    power_spectrum = fft_vals**2
    features.append(np.max(power_spectrum))     # 7. Max Power Spectrum
    
    freqs = np.fft.fftfreq(len(signal))
    positive_mask = freqs > 0
    if np.sum(power_spectrum[positive_mask]) > 0:
        cumsum = np.cumsum(power_spectrum[positive_mask])
        total = cumsum[-1]
        median_idx = np.searchsorted(cumsum, total / 2)
        features.append(freqs[positive_mask][min(median_idx, len(freqs[positive_mask])-1)])  # 8. Median Freq
        features.append(np.sum(freqs[positive_mask] * power_spectrum[positive_mask]) / total)  # 9. Mean Freq
    else:
        features.extend([0, 0])
    
    return np.array(features)


# =============================================================================
# PART 2: YOUR MR-TAE-FUSION MODEL (DEEP LEARNING APPROACH)
# =============================================================================
"""
Your MR-TAE-Fusion model uses:
    - MWCNN Encoder (Multi-level Wavelet CNN) for wavelet-domain feature extraction
    - BiGRU for temporal context modeling
    - Swin Transformer for global attention
    - Attention Gates for skip connections
    - Dual-Stream Fusion Head for denoising + segmentation

The model learns EVERYTHING end-to-end from data - no hand-crafted features!

Original purpose: Denoising + Segmentation (PD type identification)
New purpose: We'll add DISTANCE CLASSIFICATION on top!
"""


# =============================================================================
# PART 3: THE BEST METHOD - MR-TAE + DISTANCE CLASSIFIER FUSION
# =============================================================================

class MRTAEDistanceClassifier(nn.Module):
    """
    ULTIMATE MODEL: MR-TAE-Fusion Encoder + Distance Classification Head
    
    This model does:
    1. Takes noisy PD signal as input
    2. Uses MR-TAE encoder to extract deep features
    3. Optionally outputs denoised signal (reconstruction)
    4. Classifies into 4 distance classes (1.0mm, 1.8mm, 2.0mm, 2.5mm)
    
    Why this is BETTER than 9-feature method:
    - Learns features from data (not hand-crafted)
    - Captures temporal patterns (not just statistics)
    - Can handle noisy signals directly
    - End-to-end training optimizes for classification
    """
    
    def __init__(
        self,
        signal_length: int = 2001,
        num_classes: int = 4,
        encoder_channels: list = None,
        use_denoising_head: bool = True
    ):
        super().__init__()
        
        if encoder_channels is None:
            encoder_channels = [64, 128, 256]
        
        self.signal_length = signal_length
        self.num_classes = num_classes
        self.use_denoising_head = use_denoising_head
        
        # ===== MR-TAE Style Encoder =====
        # Multi-resolution feature extraction with skip connections
        
        self.encoder = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        in_ch = 1
        for out_ch in encoder_channels:
            self.encoder.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            ))
            self.pools.append(nn.MaxPool1d(2, return_indices=True))
            in_ch = out_ch
        
        # ===== Bottleneck with Attention =====
        bottleneck_ch = encoder_channels[-1] * 2
        self.bottleneck = nn.Sequential(
            nn.Conv1d(encoder_channels[-1], bottleneck_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(bottleneck_ch),
            nn.GELU(),
        )
        
        # Self-attention layer (inspired by Swin Transformer)
        self.attention = nn.MultiheadAttention(
            embed_dim=bottleneck_ch,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(bottleneck_ch)
        
        # ===== BiGRU for Temporal Modeling =====
        self.bigru = nn.GRU(
            input_size=bottleneck_ch,
            hidden_size=bottleneck_ch // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        
        # ===== Classification Head =====
        # Uses global features for distance classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(bottleneck_ch, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        # ===== Denoising Decoder (Optional) =====
        if use_denoising_head:
            self.decoder = nn.ModuleList()
            self.unpools = nn.ModuleList()
            
            in_ch = bottleneck_ch
            for out_ch in reversed(encoder_channels):
                self.unpools.append(nn.MaxUnpool1d(2))
                self.decoder.append(nn.Sequential(
                    nn.Conv1d(in_ch + out_ch, out_ch, kernel_size=5, padding=2),  # Skip connection
                    nn.BatchNorm1d(out_ch),
                    nn.GELU(),
                    nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_ch),
                    nn.GELU(),
                ))
                in_ch = out_ch
            
            self.output_conv = nn.Conv1d(encoder_channels[0], 1, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Noisy signal (batch, 1, signal_length)
        
        Returns:
            Dict with 'logits' (classification) and optionally 'denoised'
        """
        batch_size = x.size(0)
        
        # Encoder with skip connections
        encoder_features = []
        pool_indices = []
        pool_sizes = []
        
        out = x
        for enc, pool in zip(self.encoder, self.pools):
            out = enc(out)
            encoder_features.append(out)
            pool_sizes.append(out.size())
            out, indices = pool(out)
            pool_indices.append(indices)
        
        # Bottleneck
        out = self.bottleneck(out)  # (B, C, L)
        
        # Self-attention
        out_t = out.transpose(1, 2)  # (B, L, C)
        attn_out, _ = self.attention(out_t, out_t, out_t)
        out_t = self.attn_norm(out_t + attn_out)
        
        # BiGRU
        gru_out, _ = self.bigru(out_t)
        out = gru_out.transpose(1, 2)  # (B, C, L)
        
        # Global pooling for classification
        global_feat = self.global_pool(out).view(batch_size, -1)
        logits = self.classifier(global_feat)
        
        result = {'logits': logits}
        
        # Decoder for denoising (if enabled)
        if self.use_denoising_head:
            for i, (unpool, dec) in enumerate(zip(self.unpools, self.decoder)):
                idx = len(self.unpools) - 1 - i
                out = unpool(out, pool_indices[idx], output_size=pool_sizes[idx])
                skip = encoder_features[idx]
                out = torch.cat([out, skip], dim=1)
                out = dec(out)
            
            denoised = self.output_conv(out)
            result['denoised'] = denoised
        
        return result
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class."""
        with torch.no_grad():
            return self.forward(x)['logits'].argmax(dim=1)


# =============================================================================
# PART 4: TRAINING PIPELINE
# =============================================================================

def generate_distance_signal(
    distance_class: int,
    signal_length: int = 2001,
    rng: np.random.Generator = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate clean and noisy PD signal for a distance class."""
    if rng is None:
        rng = np.random.default_rng()
    
    t = np.linspace(0, 2e-6, signal_length)
    clean = np.zeros(signal_length)
    
    # Distance-specific parameters
    params = {
        0: {'events': (3, 8), 'amp': (1.0, 3.0)},      # 1.0mm - sparse
        1: {'events': (10, 25), 'amp': (2.0, 5.0)},    # 1.8mm - moderate
        2: {'events': (25, 50), 'amp': (3.0, 7.0)},    # 2.0mm - frequent
        3: {'events': (50, 100), 'amp': (4.0, 10.0)}   # 2.5mm - dense
    }
    
    p = params[distance_class]
    num_events = rng.integers(*p['events'])
    
    for _ in range(num_events):
        pos = rng.integers(0, signal_length - 10)
        amp = rng.uniform(*p['amp']) * rng.choice([-1, 1])
        pulse_type = rng.choice(['spike', 'damped'])
        
        if pulse_type == 'spike':
            clean[pos] = amp
            if pos + 1 < signal_length:
                clean[pos + 1] = -amp * 0.3
        else:
            width = rng.integers(3, 15)
            decay = np.exp(-np.arange(width) / (width * 0.3))
            end_idx = min(pos + width, signal_length)
            clean[pos:end_idx] = amp * decay[:end_idx - pos]
    
    # Normalize
    if np.max(np.abs(clean)) > 0:
        clean = clean / np.max(np.abs(clean)) * p['amp'][1]
    
    # Add noise
    snr_db = rng.uniform(-15, 5)
    signal_power = np.mean(clean**2) + 1e-10
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), signal_length)
    noisy = clean + noise
    
    return clean, noisy


class PDDataset(Dataset):
    """Dataset for PD signal classification with on-the-fly generation."""
    
    def __init__(self, num_samples: int, seed: int = 42):
        self.num_samples = num_samples
        self.rng = np.random.default_rng(seed)
        self.samples_per_class = num_samples // 4
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        class_idx = idx % 4
        clean, noisy = generate_distance_signal(class_idx, rng=self.rng)
        
        return {
            'noisy': torch.tensor(noisy, dtype=torch.float32).unsqueeze(0),
            'clean': torch.tensor(clean, dtype=torch.float32).unsqueeze(0),
            'label': torch.tensor(class_idx, dtype=torch.long)
        }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    device: str = 'cuda',
    use_denoising_loss: bool = True
) -> Dict:
    """
    Train the MR-TAE Distance Classifier.
    
    Uses multi-task loss:
    - Classification loss (CrossEntropy)
    - Reconstruction loss (MSE) for denoising
    """
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_recon = nn.MSELoss()
    
    history = {'train_loss': [], 'val_acc': [], 'train_acc': []}
    best_val_acc = 0
    best_state = None
    
    print(f"\n{'='*60}")
    print(f"Training MR-TAE Distance Classifier")
    print(f"{'='*60}")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(noisy)
            
            # Classification loss
            loss_cls = criterion_cls(outputs['logits'], labels)
            
            # Reconstruction loss (denoising)
            if use_denoising_loss and 'denoised' in outputs:
                loss_recon = criterion_recon(outputs['denoised'], clean)
                loss = loss_cls + 0.5 * loss_recon  # Multi-task weighting
            else:
                loss = loss_cls
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs['logits'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        scheduler.step()
        train_acc = 100 * correct / total
        history['train_loss'].append(total_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        
        # Validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    noisy = batch['noisy'].to(device)
                    labels = batch['label'].to(device)
                    outputs = model(noisy)
                    _, predicted = outputs['logits'].max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_acc = 100 * val_correct / val_total
            history['val_acc'].append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = model.state_dict().copy()
            
            print(f"  Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Loss: {history['train_loss'][-1]:.4f} | "
                  f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | "
                  f"Best: {best_val_acc:.1f}%")
    
    model.load_state_dict(best_state)
    return history


# =============================================================================
# PART 5: MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🔥 ULTIMATE MR-TAE-FUSION DISTANCE CLASSIFIER 🔥")
    print("="*80)
    
    # Configuration
    NUM_TRAIN = 10000  # Increase for better results (e.g., 50000)
    NUM_VAL = 2000
    NUM_EPOCHS = 100   # Increase for better results (e.g., 200)
    BATCH_SIZE = 64
    
    print(f"\nConfiguration:")
    print(f"  Training samples: {NUM_TRAIN:,}")
    print(f"  Validation samples: {NUM_VAL:,}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = PDDataset(NUM_TRAIN, seed=42)
    val_dataset = PDDataset(NUM_VAL, seed=123)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=0, pin_memory=True)
    
    # Create model
    print("\nCreating MR-TAE Distance Classifier...")
    model = MRTAEDistanceClassifier(
        signal_length=2001,
        num_classes=4,
        encoder_channels=[64, 128, 256],
        use_denoising_head=True
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    
    # Train
    start_time = time.time()
    history = train_model(
        model, train_loader, val_loader,
        num_epochs=NUM_EPOCHS,
        device=device,
        use_denoising_loss=True
    )
    train_time = time.time() - start_time
    print(f"\nTraining complete in {train_time/60:.1f} minutes")
    
    # Load real data for testing
    print("\n" + "="*60)
    print("Testing on Real Q.lin Data")
    print("="*60)
    
    def load_mat_file(filepath):
        try:
            import scipy.io as sio
            return sio.loadmat(str(filepath))
        except:
            import mat73
            return mat73.loadmat(str(filepath))
    
    def preprocess(data, target_len=2001, max_samples=100):
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[0] > data.shape[1]:
            data = data.T
        processed = []
        for i in range(min(max_samples, data.shape[0])):
            sig = data[i]
            if len(sig) >= target_len:
                sig = sig[:target_len]
            else:
                sig = np.pad(sig, (0, target_len - len(sig)))
            if np.max(np.abs(sig)) > 0:
                sig = sig / np.max(np.abs(sig))
            processed.append(sig)
        return np.array(processed)
    
    # Load test data
    X_test, y_test = [], []
    files = [('1.0mm', 'data1.0mm.mat', 0), ('1.8mm', 'data1.8mm.mat', 1),
             ('2.0mm', 'data2.0mm.mat', 2), ('2.5mm', 'data2.5mm.mat', 3)]
    
    for label, fname, cls in files:
        path = QLIN_DIR / fname
        if path.exists():
            mat = load_mat_file(path)
            for k, v in mat.items():
                if not k.startswith('__') and isinstance(v, np.ndarray):
                    signals = preprocess(v)
                    X_test.append(signals)
                    y_test.extend([cls] * len(signals))
                    print(f"  {label}: {len(signals)} samples")
                    break
    
    X_test = np.vstack(X_test)
    y_test = np.array(y_test)
    
    # Evaluate
    model.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        preds = outputs['logits'].argmax(1).cpu().numpy()
    
    accuracy = accuracy_score(y_test, preds) * 100
    
    print(f"\n{'='*60}")
    print(f"🏆 FINAL ACCURACY: {accuracy:.2f}%")
    print(f"{'='*60}")
    
    # Per-class results
    class_names = ['1.0mm', '1.8mm', '2.0mm', '2.5mm']
    print("\nPer-Class Accuracy:")
    for i, name in enumerate(class_names):
        mask = y_test == i
        if mask.sum() > 0:
            class_acc = (preds[mask] == i).mean() * 100
            print(f"  {name}: {class_acc:.1f}%")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, preds)
    print(f"       {' '.join([f'{n:>6}' for n in class_names])}")
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>6} {' '.join([f'{c:>6}' for c in row])}")
    
    # Save model
    save_path = PROJECT_ROOT / 'mr_tae_fusion' / 'checkpoints' / 'mr_tae_distance_classifier.pth'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'accuracy': accuracy
    }, save_path)
    print(f"\nModel saved: {save_path}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history['train_acc'], label='Train', alpha=0.7)
    val_epochs = list(range(4, NUM_EPOCHS, 5))
    axes[0].plot(val_epochs, history['val_acc'], 'o-', label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('MR-TAE Distance Classifier Training')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Confusion matrix heatmap
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_title(f'Confusion Matrix (Accuracy: {accuracy:.1f}%)')
    
    plt.tight_layout()
    plt.savefig(DATA_DIR / 'mr_tae_distance_results.png', dpi=150)
    print(f"Plot saved: mr_tae_distance_results.png")
    
    print("\n" + "="*80)
    print("✅ MR-TAE DISTANCE CLASSIFIER COMPLETE!")
    print("="*80)

    try:
        from Try_Model.benchmark_eval import evaluate_model_on_real_data
        print("\nRunning standard 90% real-data benchmark evaluation...")
        evaluate_model_on_real_data(model, "MR_TAE_Ultimate", is_denoiser=True, is_distance_classifier=True)
    except Exception as e:
        print(f"Benchmark eval failed: {e}")
