import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix

class BenchmarkTracker:
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.log_dir / 'benchmark_results.json'
        self.csv_path = self.log_dir / 'results_summary.csv'
        self.results = {}
        
        if self.json_path.exists():
            with open(self.json_path, 'r') as f:
                self.results = json.load(f)
                
    def log_classification_result(self, method_name: str, y_true: np.ndarray, y_pred: np.ndarray, 
                                  train_time: float = None, inference_time: float = None):
        """Logs classification accuracy and generates confusion matrix metrics."""
        acc = accuracy_score(y_true, y_pred) * 100.0
        cm = confusion_matrix(y_true, y_pred).tolist()
        
        # Calculate per-class accuracy
        unique_classes = np.unique(y_true)
        per_class_acc = {}
        cm_np = np.array(cm)
        for i, cls in enumerate(unique_classes):
            if np.sum(cm_np[i, :]) > 0:
                per_class_acc[f"Class {cls}"] = (cm_np[i, i] / np.sum(cm_np[i, :])) * 100.0
        
        if method_name not in self.results:
            self.results[method_name] = {}
            
        self.results[method_name].update({
            'accuracy': acc,
            'per_class_accuracy': per_class_acc,
            'confusion_matrix': cm,
            'train_time_sec': train_time,
            'inference_time_sec': inference_time
        })
        self._save()
        print(f"Logged {method_name}: Accuracy = {acc:.2f}%")
        
    def log_denoising_result(self, method_name: str, clean_true: np.ndarray, denoised_pred: np.ndarray):
        """Calculates and logs MSE, RMSE, and NCC for denoising performance."""
        # Ensure flat arrays for MSE/RMSE calculation over the whole dataset
        mse = np.mean((clean_true - denoised_pred) ** 2)
        rmse = np.sqrt(mse)
        
        # Calculate Normalized Cross Correlation (NCC) for each signal and average
        ncc_scores = []
        for i in range(len(clean_true)):
            sig1 = clean_true[i].flatten()
            sig2 = denoised_pred[i].flatten()
            norm1 = np.linalg.norm(sig1)
            norm2 = np.linalg.norm(sig2)
            if norm1 > 0 and norm2 > 0:
                ncc = np.dot(sig1, sig2) / (norm1 * norm2)
                ncc_scores.append(ncc)
            else:
                ncc_scores.append(0.0)
                
        avg_ncc = np.mean(ncc_scores)
        
        if method_name not in self.results:
            self.results[method_name] = {}
            
        self.results[method_name].update({
            'mse': mse,
            'rmse': rmse,
            'ncc': avg_ncc
        })
        self._save()
        print(f"Logged {method_name} Denoising: MSE={mse:.5f}, RMSE={rmse:.5f}, NCC={avg_ncc:.4f}")
        
    def _save(self):
        with open(self.json_path, 'w') as f:
            json.dump(self.results, f, indent=4)
            
        # Also write CSV summary
        with open(self.csv_path, 'w', newline='') as f:
            if not self.results:
                return
            writer = csv.writer(f)
            headers = ['Method', 'Accuracy', 'MSE', 'RMSE', 'NCC', 'Train Time (s)']
            writer.writerow(headers)
            for method, data in self.results.items():
                writer.writerow([
                    method,
                    f"{data.get('accuracy', 0):.2f}%",
                    f"{data.get('mse', 'N/A')}",
                    f"{data.get('rmse', 'N/A')}",
                    f"{data.get('ncc', 'N/A')}",
                    f"{data.get('train_time_sec', 'N/A')}"
                ])
                
    def plot_comparison(self):
        """Generates a bar chart comparing accuracy across all tracked methods."""
        methods = list(self.results.keys())
        accs = [self.results[m].get('accuracy', 0) for m in methods]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, accs, color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f1c40f', '#34495e', '#1abc9c'])
        plt.ylabel('Accuracy (%)')
        plt.title('Method Comparison Accuracy (90% Real Test Data)')
        plt.ylim(0, 105)
        plt.xticks(rotation=25, ha='right')
        
        for bar, acc in zip(bars, accs):
            if acc > 0:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, 
                         f'{acc:.1f}%', ha='center', fontsize=10, fontweight='bold')
                 
        plt.tight_layout()
        plt.savefig(self.log_dir / 'method_comparison_benchmark.png', dpi=150)
        print(f"Plot saved to {self.log_dir / 'method_comparison_benchmark.png'}")
