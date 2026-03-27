#!/usr/bin/env python3
"""
Evaluation script for GCL-IoT

Computes all metrics reported in the paper:
- Precision, Recall, Micro-F1
- Per-class metrics
- Statistical significance tests
"""

import os
import sys
import argparse
import numpy as np
import torch
from scipy import stats
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.gcl_iot import GCLIoT
from data.dataset import load_iot_data
from utils.helpers import set_seed, load_config
from utils.metrics import compute_metrics, compute_node_metrics, MetricsTracker
from utils.visualization import plot_tsne, plot_spectral_energy


def evaluate_model(model, data, device):
    """Evaluate model and return comprehensive metrics"""
    model.eval()
    
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y
    test_mask = data.test_mask
    
    with torch.no_grad():
        _, _, logits = model(x, edge_index)
        y_pred = logits.argmax(dim=1).cpu()
        y_probs = torch.softmax(logits, dim=1).cpu()
    
    # Filter to test set
    y_true_test = y[test_mask].numpy()
    y_pred_test = y_pred[test_mask].numpy()
    y_probs_test = y_probs[test_mask].numpy()
    
    # Compute metrics
    metrics = compute_metrics(y_true_test, y_pred_test, average='binary')
    
    # Additional metrics
    metrics['classification_report'] = classification_report(
        y_true_test, y_pred_test, target_names=['Benign', 'Malicious'], output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true_test, y_pred_test)
    metrics['confusion_matrix'] = cm
    
    return metrics, y_true_test, y_pred_test, y_probs_test


def run_statistical_tests(results, baseline_results, method_name):
    """
    Run statistical significance tests between GCL-IoT and baselines
    """
    print("\n" + "="*60)
    print("Statistical Significance Tests (Friedman Test)")
    print("="*60)
    
    # Prepare results for Friedman test
    num_datasets = len(results)
    num_methods = 1 + len(baseline_results)
    
    # Create ranking matrix
    rankings = []
    
    for dataset_idx in range(num_datasets):
        scores = []
        # GCL-IoT score
        scores.append(results[dataset_idx]['f1_micro'])
        # Baseline scores
        for baseline in baseline_results.values():
            scores.append(baseline[dataset_idx]['f1_micro'])
        
        # Rank (1 is best)
        ranked = stats.rankdata([-s for s in scores])
        rankings.append(ranked)
    
    rankings = np.array(rankings)
    avg_ranks = rankings.mean(axis=0)
    
    print(f"\nAverage ranks across datasets:")
    for i, name in enumerate([method_name] + list(baseline_results.keys())):
        print(f"  {name}: {avg_ranks[i]:.4f}")
    
    # Compute Friedman statistic
    k = num_methods
    N = num_datasets
    R_j = avg_ranks
    chi2 = (12 * N) / (k * (k + 1)) * (np.sum(R_j ** 2) - (k * (k + 1) ** 2) / 4)
    
    print(f"\nFriedman χ² = {chi2:.4f}")
    
    # Compute F_F statistic
    F_F = ((N - 1) * chi2) / (N * (k - 1) - chi2)
    print(f"F_F = {F_F:.4f}")
    
    # Critical value at α=0.05
    from scipy.stats import f
    df1 = k - 1
    df2 = (k - 1) * (N - 1)
    critical_f = f.ppf(0.95, df1, df2)
    print(f"Critical F({df1}, {df2}) at α=0.05 = {critical_f:.4f}")
    
    if F_F > critical_f:
        print("\n✓ Reject null hypothesis: Significant differences exist between methods")
    else:
        print("\n✗ Fail to reject null hypothesis: No significant differences detected")
    
    return {'chi2': chi2, 'F_F': F_F, 'critical_f': critical_f}


def generate_table1(model_results, baseline_results, datasets):
    """
    Generate Table 1: Performance comparison
    """
    print("\n" + "="*60)
    print("Table 1: Performance Comparison")
    print("="*60)
    
    # Create DataFrame
    rows = []
    
    # Add GCL-IoT
    for dataset_name in datasets:
        for run_idx, metrics in enumerate(model_results[dataset_name]):
            row = {
                'Method': 'GCL-IoT',
                'Dataset': dataset_name,
                'Run': run_idx + 1,
                'Precision': metrics['precision'],
                'Micro-F1': metrics['f1_micro']
            }
            rows.append(row)
    
    # Add baselines (simplified)
    for method_name, method_results in baseline_results.items():
        for dataset_name in datasets:
            for run_idx, metrics in enumerate(method_results[dataset_name]):
                row = {
                    'Method': method_name,
                    'Dataset': dataset_name,
                    'Run': run_idx + 1,
                    'Precision': metrics['precision'],
                    'Micro-F1': metrics['f1_micro']
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Compute mean and std per method and dataset
    summary = df.groupby(['Method', 'Dataset']).agg({
        'Precision': ['mean', 'std'],
        'Micro-F1': ['mean', 'std']
    }).round(4)
    
    print(summary.to_string())
    
    return summary


def visualize_embeddings(model, data, device, save_dir):
    """
    Generate t-SNE visualizations of learned embeddings
    """
    print("\n" + "="*60)
    print("Generating t-SNE Visualizations")
    print("="*60)
    
    model.eval()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    
    with torch.no_grad():
        z_homo, z_heter, _ = model(x, edge_index)
        z_concat = torch.cat([z_homo, z_heter], dim=1)
        
        # Move to CPU
        z_homo = z_homo.cpu().numpy()
        z_heter = z_heter.cpu().numpy()
        z_concat = z_concat.cpu().numpy()
        labels = data.y.cpu().numpy()
    
    # Plot homophily embeddings
    plot_tsne(z_homo, labels, title="Homophily-Focused Embeddings", 
              save_path=os.path.join(save_dir, "tsne_homo.png"))
    
    # Plot heterophily embeddings
    plot_tsne(z_heter, labels, title="Heterophily-Focused Embeddings", 
              save_path=os.path.join(save_dir, "tsne_heter.png"))
    
    # Plot concatenated embeddings
    plot_tsne(z_concat, labels, title="Concatenated Embeddings (Final)", 
              save_path=os.path.join(save_dir, "tsne_concat.png"))
    
    print(f"Visualizations saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate GCL-IoT')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--config', type=str, default='configs/gcl_iot.yaml', help='Config file path')
    parser.add_argument('--dataset', type=str, default='ton_iot', help='Dataset name')
    parser.add_argument('--data_path', type=str, default='./data/raw/', help='Raw data path')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results/', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data_path = os.path.join(args.data_path, f"{args.dataset}.csv")
    data = load_iot_data(args.dataset, data_path,
                         feature_dim=config['dataset']['feature_dim'],
                         temporal_split=config['dataset']['temporal_split'])
    
    # Initialize model
    model = GCLIoT(
        in_dim=data.x.size(1),
        hidden_dim=config['model']['hidden_dim'],
        out_dim=config['model']['out_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        temperature=config['model']['temperature'],
        alpha=config['model']['alpha'],
        top_k=config['model']['top_k'],
        device=device
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Evaluate
    metrics, y_true, y_pred, y_probs = evaluate_model(model, data, device)
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Micro-F1: {metrics['f1_micro']:.4f}")
    print(f"Macro-F1: {metrics['f1_macro']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    print("\nClassification Report:")
    for class_name, class_metrics in metrics['classification_report'].items():
        if isinstance(class_metrics, dict):
            print(f"  {class_name}: Precision={class_metrics['precision']:.4f}, "
                  f"Recall={class_metrics['recall']:.4f}, F1={class_metrics['f1-score']:.4f}")
    
    # Save results
    results = {
        'metrics': metrics,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_probs': y_probs
    }
    torch.save(results, os.path.join(args.output_dir, 'evaluation_results.pt'))
    
    # Generate visualizations
    visualize_embeddings(model, data, device, args.output_dir)
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()