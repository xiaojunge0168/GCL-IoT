"""
Evaluation metrics for intrusion detection
"""

import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from typing import Tuple, Dict, Optional


def compute_metrics(y_true: np.ndarray, 
                    y_pred: np.ndarray,
                    average: str = 'binary') -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging method for multi-class
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # Micro and macro versions
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    metrics['true_positives'] = tp
    
    # Additional metrics
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return metrics


def compute_node_metrics(model, 
                         data,
                         device: str = 'cuda') -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Compute metrics for node classification
    
    Args:
        model: Trained GCL-IoT model
        data: PyG Data object
        device: Device to run on
        
    Returns:
        metrics: Dictionary of metrics
        y_true: Ground truth labels
        y_pred: Predicted labels
    """
    model.eval()
    
    with torch.no_grad():
        # Get predictions
        _, _, logits = model(data.x.to(device), data.edge_index.to(device))
        y_pred = logits.argmax(dim=1).cpu().numpy()
        y_true = data.y.cpu().numpy()
        
        # Test mask
        test_mask = data.test_mask.cpu().numpy()
        
        # Filter to test set
        y_true_test = y_true[test_mask]
        y_pred_test = y_pred[test_mask]
        
        # Compute metrics
        metrics = compute_metrics(y_true_test, y_pred_test)
        
        # Add ROC-AUC if applicable
        if len(np.unique(y_true_test)) == 2:
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_test = probs[test_mask, 1]
            metrics['roc_auc'] = roc_auc_score(y_true_test, probs_test)
        
    return metrics, y_true_test, y_pred_test


def compute_batch_metrics(predictions: torch.Tensor, 
                          targets: torch.Tensor,
                          average: str = 'binary') -> Dict[str, float]:
    """
    Compute metrics for a batch
    
    Args:
        predictions: Model predictions [batch_size, num_classes]
        targets: Ground truth labels [batch_size]
        average: Averaging method
        
    Returns:
        metrics: Dictionary of metrics
    """
    pred_labels = predictions.argmax(dim=1).cpu().numpy()
    true_labels = targets.cpu().numpy()
    
    return compute_metrics(true_labels, pred_labels, average)


class MetricsTracker:
    """Track metrics across multiple runs"""
    
    def __init__(self):
        self.metrics_history = []
        
    def update(self, metrics: Dict[str, float]):
        """Add metrics from a new run"""
        self.metrics_history.append(metrics)
    
    def get_mean_std(self) -> Dict[str, Tuple[float, float]]:
        """Compute mean and standard deviation across runs"""
        if not self.metrics_history:
            return {}
        
        results = {}
        for key in self.metrics_history[0].keys():
            values = [m[key] for m in self.metrics_history if key in m]
            if values:
                results[key] = (np.mean(values), np.std(values))
        
        return results
    
    def get_summary_table(self) -> str:
        """Generate summary table string"""
        stats = self.get_mean_std()
        
        lines = ["| Metric | Mean ± Std |"]
        lines.append("|--------|------------|")
        
        for metric, (mean, std) in stats.items():
            lines.append(f"| {metric} | {mean:.4f} ± {std:.4f} |")
        
        return "\n".join(lines)