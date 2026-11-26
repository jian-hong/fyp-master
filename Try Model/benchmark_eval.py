import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(r'D:\New folder (2)\Data\OneDrive_2025-12-13\Try Model')
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_tracker import BenchmarkTracker

def evaluate_model_on_real_data(model, model_name: str, is_denoiser: bool = False, is_distance_classifier: bool = True, device: str = 'cuda'):
    """
    Evaluates a trained model on the 90% holdout real data split, 
    and logs the result using BenchmarkTracker.
    """
    tracker = BenchmarkTracker(log_dir=str(PROJECT_ROOT / 'outputs' / 'benchmark'))
    
    # Load 90% real test data (clean)
    test_data_path = PROJECT_ROOT / 'Datset from Q.lin' / 'processed' / 'real_test_clean.npz'
    if not test_data_path.exists():
        print(f"Error: Test data not found at {test_data_path}")
        return
        
    data = np.load(test_data_path)
    X_test = data['clean']
    y_test = data['labels']
    
    X_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
    
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'predict'):
            # Handling models with predict method
            outputs = model.predict(X_tensor)
            if isinstance(outputs, torch.Tensor):
                preds = outputs.cpu().numpy()
            else:
                preds = outputs
        else:
            # Standard nn.Module forward
            outputs = model(X_tensor)
            
            # Dictionary output (like mr_tae_ultimate.py)
            if isinstance(outputs, dict):
                if 'logits' in outputs and is_distance_classifier:
                    preds = outputs['logits'].argmax(1).cpu().numpy()
                elif 'denoised' in outputs and is_denoiser:
                    denoised = outputs['denoised'].squeeze(1).cpu().numpy()
                    tracker.log_denoising_result(model_name, X_test, denoised)
                    
                if 'logits' not in outputs and not is_denoiser:
                    print(f"Warning: Model {model_name} dict output does not match expected keys.")
                    return
            
            # Tuple output (like mr_tae_fusion models returning denoised, seg_logits)
            elif isinstance(outputs, tuple):
                denoised, seg_logits = outputs
                if is_denoiser:
                    denoised_np = denoised.squeeze(1).cpu().numpy()
                    tracker.log_denoising_result(model_name, X_test, denoised_np)
                # Ignore seg_logits for this benchmark unless we want to track it
                
            # Direct tensor output (like DistanceClassifier)
            elif isinstance(outputs, torch.Tensor):
                if is_distance_classifier:
                    preds = outputs.argmax(1).cpu().numpy()
                elif is_denoiser:
                    denoised = outputs.squeeze(1).cpu().numpy()
                    tracker.log_denoising_result(model_name, X_test, denoised)
                    
    if is_distance_classifier and 'preds' in locals():
        tracker.log_classification_result(model_name, y_test, preds)
        
    # Generate the comparison plot automatically
    tracker.plot_comparison()
