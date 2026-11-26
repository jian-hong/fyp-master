"""
Extended Training - 200k Samples, 500 Epochs

Three-Method Comparison for PD Classification
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import time
warnings.filterwarnings('ignore')

# Setup paths
PROJECT_ROOT = Path(r'D:\New folder (2)\Data\OneDrive_2025-12-13\Try Model')
DATA_DIR = Path(r'D:\New folder (2)\Data\OneDrive_2025-12-13')
QLIN_DIR = DATA_DIR / 'Datset from Q.lin'
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
NUM_SAMPLES_PER_CLASS = 50000  # 50k per class = 200k total
NUM_EPOCHS = 500
BATCH_SIZE = 256  # Larger batch for faster training

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\n" + "="*80)
print(f"EXTENDED TRAINING: {NUM_SAMPLES_PER_CLASS * 4:,} SAMPLES, {NUM_EPOCHS} EPOCHS")
print("="*80)

start_time = time.time()

# ============================================================================
# DATA LOADING
# ============================================================================
print("\n[1] LOADING DATA...")

def load_mat_file(filepath):
    try:
        import scipy.io as sio
        return sio.loadmat(str(filepath))
    except NotImplementedError:
        import mat73
        return mat73.loadmat(str(filepath))

# Load Q.lin real data for testing
real_data = {}
files = [
    ('1.0mm', 'data1.0mm.mat', 0),
    ('1.8mm', 'data1.8mm.mat', 1),
    ('2.0mm', 'data2.0mm.mat', 2),
    ('2.5mm', 'data2.5mm.mat', 3)
]

for label, filename, class_idx in files:
    filepath = QLIN_DIR / filename
    if not filepath.exists():
        continue
    mat = load_mat_file(filepath)
    for key, value in mat.items():
        if not key.startswith('__') and isinstance(value, np.ndarray):
            real_data[label] = {'data': value, 'class': class_idx}
            print(f"  {label}: shape={value.shape}")
            break

# Preprocess real data
def preprocess_signals(data, target_length=2001, max_samples=100):
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] > data.shape[1]:
        data = data.T
    
    processed = []
    for i in range(min(max_samples, data.shape[0])):
        sig = data[i]
        if len(sig) >= target_length:
            sig = sig[:target_length]
        else:
            sig = np.pad(sig, (0, target_length - len(sig)))
        if np.max(np.abs(sig)) > 0:
            sig = sig / np.max(np.abs(sig))
        processed.append(sig)
    return np.array(processed)

# Prepare test data (use more samples for better evaluation)
X_test_signals = []
y_test = []
for label, info in real_data.items():
    signals = preprocess_signals(info['data'], max_samples=100)
    X_test_signals.append(signals)
    y_test.extend([info['class']] * len(signals))

X_test_signals = np.vstack(X_test_signals)
y_test = np.array(y_test)
print(f"  Total test samples: {len(y_test)}")

# Generate large training dataset
print(f"\n  Generating {NUM_SAMPLES_PER_CLASS * 4:,} training samples...")
from mr_tae_fusion.data.distance_signals import generate_batch, DistanceSignalConfig

config = DistanceSignalConfig()

# Generate in batches to avoid memory issues
print("  Generating training data in batches...")
train_clean = []
train_noisy = []
train_labels = []

batch_gen_size = 10000  # Generate 10k at a time
for i in range(NUM_SAMPLES_PER_CLASS // (batch_gen_size // 4)):
    batch = generate_batch(
        batch_size=batch_gen_size // 4, 
        config=config, 
        snr_range=(-15, 10),
        seed=42 + i
    )
    train_clean.append(batch['clean'])
    train_noisy.append(batch['noisy'])
    train_labels.append(batch['labels'])
    if (i + 1) % 5 == 0:
        print(f"    Progress: {(i + 1) * batch_gen_size:,} samples generated")

train_data = {
    'clean': np.vstack(train_clean),
    'noisy': np.vstack(train_noisy),
    'labels': np.concatenate(train_labels)
}
del train_clean, train_noisy, train_labels  # Free memory

print(f"  Training samples: {len(train_data['labels']):,}")
print(f"  Class distribution: {np.bincount(train_data['labels'])}")

# Validation data
val_data = generate_batch(batch_size=1000, config=config, snr_range=(-10, 5), seed=999)
print(f"  Validation samples: {len(val_data['labels']):,}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_9_features(signal):
    """Calculate 9 statistical features."""
    features = []
    features.append(np.sqrt(np.mean(signal**2)))  # RMS
    features.append(np.var(signal))
    features.append(np.std(signal))
    mean_abs = np.abs(np.mean(signal))
    features.append(features[0] / mean_abs if mean_abs > 1e-10 else 0)
    features.append(stats.kurtosis(signal))
    features.append(stats.skew(signal))
    
    fft_vals = np.abs(np.fft.fft(signal))
    power_spectrum = fft_vals**2
    features.append(np.max(power_spectrum))
    
    freqs = np.fft.fftfreq(len(signal))
    positive_mask = freqs > 0
    if np.sum(power_spectrum[positive_mask]) > 0:
        cumsum = np.cumsum(power_spectrum[positive_mask])
        total = cumsum[-1]
        median_idx = np.searchsorted(cumsum, total / 2)
        features.append(freqs[positive_mask][min(median_idx, len(freqs[positive_mask])-1)])
        features.append(np.sum(freqs[positive_mask] * power_spectrum[positive_mask]) / total)
    else:
        features.append(0)
        features.append(0)
    
    return np.array(features)

# ============================================================================
# METHOD 1: DISTANCE-BASED CLASSIFICATION (4 CLASSES)
# ============================================================================
print("\n" + "="*80)
print("[METHOD 1] DISTANCE-BASED CLASSIFICATION (4 CLASSES)")
print(f"Training with {len(train_data['labels']):,} samples, {NUM_EPOCHS} epochs")
print("="*80)

from mr_tae_fusion.models.distance_classifier import DistanceClassifier

model1 = DistanceClassifier(
    in_channels=1, signal_length=2001, num_classes=4,
    hidden_channels=[64, 128, 256]  # Larger model for more data
).to(device)

total_params = sum(p.numel() for p in model1.parameters())
print(f"Model parameters: {total_params:,}")

# DataLoader
def create_loader(data, batch_size=BATCH_SIZE, shuffle=True):
    X = torch.tensor(data['clean'], dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(data['labels'], dtype=torch.long)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle, 
                     num_workers=0, pin_memory=True)

train_loader1 = create_loader(train_data)
val_loader1 = create_loader(val_data, shuffle=False)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model1.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

best_val_acc = 0
best_state = None
history1 = {'train_acc': [], 'val_acc': [], 'train_loss': []}

print("\nTraining started...")
epoch_start = time.time()

for epoch in range(NUM_EPOCHS):
    model1.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X, y in train_loader1:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model1(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
    
    scheduler.step()
    train_acc = 100 * correct / total
    history1['train_acc'].append(train_acc)
    history1['train_loss'].append(total_loss / len(train_loader1))
    
    # Validation
    if (epoch + 1) % 10 == 0 or epoch == NUM_EPOCHS - 1:
        model1.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X, y in val_loader1:
                X, y = X.to(device), y.to(device)
                outputs = model1(X)
                _, predicted = outputs.max(1)
                val_total += y.size(0)
                val_correct += predicted.eq(y).sum().item()
        
        val_acc = 100 * val_correct / val_total
        history1['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model1.state_dict().copy()
        
        elapsed = time.time() - epoch_start
        eta = elapsed / (epoch + 1) * (NUM_EPOCHS - epoch - 1)
        print(f"  Epoch {epoch+1:3d}/{NUM_EPOCHS} | Train: {train_acc:.1f}% | "
              f"Val: {val_acc:.1f}% | Best: {best_val_acc:.1f}% | ETA: {eta/60:.1f}min")

model1.load_state_dict(best_state)
print(f"\nMethod 1 Training complete. Best Val Acc: {best_val_acc:.2f}%")

# Evaluate on real data
model1.eval()
X_test_tensor = torch.tensor(X_test_signals, dtype=torch.float32).unsqueeze(1).to(device)
with torch.no_grad():
    preds1 = model1(X_test_tensor).argmax(1).cpu().numpy()

acc1 = accuracy_score(y_test, preds1) * 100
print(f"\n*** Method 1 Real-World Accuracy: {acc1:.2f}% ***")

# ============================================================================
# METHOD 2: PHYSICS-TYPE MAPPING (PD DETECTION)
# ============================================================================
print("\n" + "="*80)
print("[METHOD 2] PHYSICS-TYPE MAPPING (PD DETECTION)")
print("="*80)

class PDDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(32)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 32, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))

model2 = PDDetector().to(device)

# Training data with background
n_pd = min(100000, len(train_data['labels']))  # Limit for speed
rng = np.random.default_rng(42)
X_pd = train_data['clean'][:n_pd]
y_pd = np.ones(n_pd, dtype=np.int64)
X_bg = rng.normal(0, 0.1, (n_pd, 2001))
y_bg = np.zeros(n_pd, dtype=np.int64)

X_train2 = np.vstack([X_pd, X_bg])
y_train2 = np.concatenate([y_pd, y_bg])
perm = rng.permutation(len(y_train2))
X_train2, y_train2 = X_train2[perm], y_train2[perm]

loader2 = DataLoader(
    TensorDataset(
        torch.tensor(X_train2, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_train2, dtype=torch.long)
    ),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
)

optimizer2 = optim.AdamW(model2.parameters(), lr=1e-3)
scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, NUM_EPOCHS)

print(f"Training PD detector with {len(y_train2):,} samples...")
for epoch in range(NUM_EPOCHS):
    model2.train()
    for X, y in loader2:
        X, y = X.to(device), y.to(device)
        optimizer2.zero_grad()
        loss = criterion(model2(X), y)
        loss.backward()
        optimizer2.step()
    scheduler2.step()
    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS}")

model2.eval()
with torch.no_grad():
    preds2_binary = model2(X_test_tensor).argmax(1).cpu().numpy()

pd_detection_rate = (preds2_binary == 1).mean() * 100
print(f"\n*** Method 2 PD Detection Rate: {pd_detection_rate:.2f}% ***")

# ============================================================================
# METHOD 3: 9-FEATURE CLASSIFICATION
# ============================================================================
print("\n" + "="*80)
print("[METHOD 3] 9-FEATURE CLASSIFICATION")
print("="*80)

# Use subset for feature extraction (9-feature is fast)
n_feat = min(50000, len(train_data['labels']))
print(f"Extracting features from {n_feat:,} samples...")

X_train_features = np.array([calculate_9_features(sig) for sig in train_data['clean'][:n_feat]])
y_train_labels = train_data['labels'][:n_feat]

X_test_features = np.array([calculate_9_features(sig) for sig in X_test_signals])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)
X_test_scaled = scaler.transform(X_test_features)

print("Training MLP classifier...")
clf3 = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=False
)
clf3.fit(X_train_scaled, y_train_labels)

print("Training Random Forest...")
rf3 = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf3.fit(X_train_scaled, y_train_labels)

preds3_mlp = clf3.predict(X_test_scaled)
preds3_rf = rf3.predict(X_test_scaled)

acc3_mlp = accuracy_score(y_test, preds3_mlp) * 100
acc3_rf = accuracy_score(y_test, preds3_rf) * 100

print(f"\n*** Method 3 (MLP) Accuracy: {acc3_mlp:.2f}% ***")
print(f"*** Method 3 (RF) Accuracy: {acc3_rf:.2f}% ***")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
total_time = time.time() - start_time

print("\n" + "="*80)
print("FINAL COMPARISON SUMMARY")
print("="*80)

results = {
    'Method 1 (Distance CNN)': acc1,
    'Method 2 (PD Detection)': pd_detection_rate,
    'Method 3 (9-Feature MLP)': acc3_mlp,
    'Method 3 (9-Feature RF)': acc3_rf
}

print(f"\n{'Method':<30} {'Accuracy':>10}")
print("-" * 45)
for method, acc in results.items():
    print(f"{method:<30} {acc:>9.2f}%")

best_method = max(results, key=results.get)
print(f"\n🏆 Best Method: {best_method} ({results[best_method]:.2f}%)")
print(f"\nTotal training time: {total_time/60:.1f} minutes")

# Per-class for Method 1
print("\n" + "-"*45)
print("Method 1 Per-Class Accuracy:")
class_names = ['1.0mm', '1.8mm', '2.0mm', '2.5mm']
for i, name in enumerate(class_names):
    mask = y_test == i
    if mask.sum() > 0:
        class_acc = (preds1[mask] == i).mean() * 100
        print(f"  {name}: {class_acc:.1f}%")

# Save
print("\nSaving results...")
save_dir = PROJECT_ROOT / 'mr_tae_fusion' / 'checkpoints'
save_dir.mkdir(parents=True, exist_ok=True)

torch.save({
    'model_state_dict': model1.state_dict(),
    'history': history1,
    'results': results,
    'config': {
        'samples': NUM_SAMPLES_PER_CLASS * 4,
        'epochs': NUM_EPOCHS
    }
}, save_dir / 'distance_classifier_200k.pth')

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Training curves
axes[0].plot(history1['train_acc'], label='Train Acc', alpha=0.8)
epochs_val = list(range(9, NUM_EPOCHS, 10)) + [NUM_EPOCHS - 1]
axes[0].plot(epochs_val, history1['val_acc'], 'o-', label='Val Acc')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title(f'Method 1 Training ({NUM_SAMPLES_PER_CLASS*4:,} samples)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Comparison bar chart
methods = list(results.keys())
accs = list(results.values())
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
bars = axes[1].bar(range(len(methods)), accs, color=colors)
axes[1].set_xticks(range(len(methods)))
axes[1].set_xticklabels(['CNN\n(Distance)', 'PD\nDetection', '9-Feat\nMLP', '9-Feat\nRF'])
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Method Comparison on Real Data')
axes[1].set_ylim(0, 105)
for bar, acc in zip(bars, accs):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 f'{acc:.1f}%', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(DATA_DIR / 'comparison_200k_500epochs.png', dpi=150)
print(f"Saved: comparison_200k_500epochs.png")

print("\n" + "="*80)
print("EXTENDED TRAINING COMPLETE")
print("="*80)

try:
    from benchmark_eval import evaluate_model_on_real_data
    print("\nRunning standard 90% real-data benchmark evaluation...")
    evaluate_model_on_real_data(model1, "Train_Extended_Model1", is_denoiser=False, is_distance_classifier=True)
except Exception as e:
    print(f"Benchmark eval failed: {e}")
