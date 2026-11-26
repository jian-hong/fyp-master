import torch
import numpy as np

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

def explain_model_with_shap(model, background_data, test_data, device):
    """
    Extract SHAP values for the MR-TAE-Fusion model to understand 
    feature importance and time-frequency reliance on partial discharge identification.
    
    Args:
        model: Trained PyTorch MR-TAE-Fusion BaseModel
        background_data: (N, 1, 2048) background noise tensors
        test_data: (K, 1, 2048) test tensors to explain
        device: Torch device corresponding to the model
    """
    if not _HAS_SHAP:
        raise ImportError("Please install shap library first via: pip install shap")
        
    model.eval()
    
    # SHAP expects models to output scalars for standard attribution 
    # For seq-to-seq denoising, we wrap the model to return the relative 
    # 'energy' or 'presence' score over the output.
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            # For explainability, we only care about the signal reconstruction output
            denoised, seg = self.model(x)
            # Sum the normalized power as a single scalar output indicating PD presence confidence
            return torch.sum(denoised**2, dim=(1,2)).unsqueeze(1)
            
    wrapper = ModelWrapper(model).to(device)
    
    background = torch.tensor(background_data, dtype=torch.float32).to(device)
    test = torch.tensor(test_data, dtype=torch.float32).to(device)
    
    print("Computing Gradient SHAP attributions...")
    explainer = shap.GradientExplainer(wrapper, background)
    shap_values = explainer.shap_values(test)
    
    # shap_values represents the attribution arrays over the 2048 window
    return shap_values

if __name__ == "__main__":
    print("Explainability module ready for SHAP analysis.")
