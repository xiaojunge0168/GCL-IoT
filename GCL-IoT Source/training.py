#!/usr/bin/env python3
"""
Main training script for GCL-IoT

Implements three-phase training:
1. Phase 1: Reinforced Neighbor Selection Pre-training (50 epochs)
2. Phase 2: Contrastive Pre-training (100 epochs)
3. Phase 3: Supervised Fine-tuning (50 epochs)
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import wandb  # Optional: for experiment tracking

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.gcl_iot import GCLIoT
from data.dataset import IoTGraphDataset, load_iot_data, get_dataset_statistics
from utils.helpers import set_seed, load_config, save_checkpoint, EarlyStopping, AverageMeter
from utils.metrics import compute_node_metrics, MetricsTracker


def train_phase1(model, data, config, device):
    """
    Phase 1: Reinforced Neighbor Selection Pre-training
    Train RL policy to identify optimal k neighbors
    """
    print("\n" + "="*50)
    print("Phase 1: Reinforced Neighbor Selection Pre-training")
    print("="*50)
    
    # For Phase 1, we only need to run the neighbor selector training
    # This is implemented within the ReinforcedNeighborSelector class
    
    model.train()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    
    # The neighbor selector is trained when get_top_k_neighbors is called
    top_k_neighbors = model.get_top_k_neighbors(x, edge_index)
    
    print(f"Phase 1 completed. Top-k neighbors shape: {top_k_neighbors.shape}")
    
    return top_k_neighbors


def train_phase2(model, data, config, device):
    """
    Phase 2: Contrastive Pre-training
    Train dual-frequency encoders using contrastive objective
    """
    print("\n" + "="*50)
    print("Phase 2: Contrastive Pre-training")
    print("="*50)
    
    epochs = config['training']['epochs'] // 2  # 100 epochs
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    contrastive_weight = config['contrastive']['loss_weight']
    
    # Separate optimizer for encoders
    encoder_params = list(model.low_encoder.parameters()) + list(model.high_encoder.parameters())
    optimizer = optim.AdamW(encoder_params, lr=lr, weight_decay=weight_decay)
    
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    top_k_neighbors = model.get_top_k_neighbors(x, edge_index)
    
    early_stopping = EarlyStopping(patience=config['training']['early_stopping_patience'])
    
    for epoch in range(epochs):
        model.train()
        
        # Forward pass
        z_homo, z_heter, _ = model(x, edge_index)
        
        # Compute contrastive loss
        contrastive_loss = model.compute_contrastive_loss(z_homo, z_heter, top_k_neighbors)
        
        loss = contrastive_weight * contrastive_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
        optimizer.step()
        
        # Validation
        if (epoch + 1) % config['experiment']['log_interval'] == 0:
            model.eval()
            with torch.no_grad():
                z_homo_val, z_heter_val, _ = model(x, edge_index)
                val_loss = model.compute_contrastive_loss(z_homo_val, z_heter_val, top_k_neighbors)
                
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")
            
            # Early stopping
            if early_stopping(-val_loss.item()):
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print("Phase 2 completed.")


def train_phase3(model, data, config, device):
    """
    Phase 3: Supervised Fine-tuning
    Train classifier with cross-entropy loss
    """
    print("\n" + "="*50)
    print("Phase 3: Supervised Fine-tuning")
    print("="*50)
    
    epochs = config['training']['epochs'] // 4  # 50 epochs
    lr_classifier = config['training']['learning_rate']
    lr_encoders = lr_classifier * 0.2  # Lower learning rate for encoders
    
    # Separate optimizers for encoders and classifier
    encoder_params = list(model.low_encoder.parameters()) + list(model.high_encoder.parameters())
    classifier_params = list(model.classifier.parameters())
    
    optimizer_encoders = optim.AdamW(encoder_params, lr=lr_encoders, weight_decay=config['training']['weight_decay'])
    optimizer_classifier = optim.AdamW(classifier_params, lr=lr_classifier, weight_decay=config['training']['weight_decay'])
    
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    
    best_val_f1 = 0.0
    best_epoch = 0
    early_stopping = EarlyStopping(patience=config['training']['early_stopping_patience'])
    
    for epoch in range(epochs):
        model.train()
        
        # Forward pass
        _, _, logits = model(x, edge_index)
        
        # Compute classification loss
        loss = model.compute_classification_loss(logits, y, train_mask)
        
        # Update encoders and classifier
        optimizer_encoders.zero_grad()
        optimizer_classifier.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
        optimizer_encoders.step()
        optimizer_classifier.step()
        
        # Validation
        if (epoch + 1) % config['experiment']['log_interval'] == 0:
            model.eval()
            with torch.no_grad():
                _, _, logits_val = model(x, edge_index)
                val_loss = model.compute_classification_loss(logits_val, y, val_mask)
                
                # Compute validation metrics
                y_pred = logits_val.argmax(dim=1)
                val_acc = (y_pred[val_mask] == y[val_mask]).float().mean()
                
                # Calculate F1
                from sklearn.metrics import f1_score
                y_true_val = y[val_mask].cpu().numpy()
                y_pred_val = y_pred[val_mask].cpu().numpy()
                val_f1 = f1_score(y_true_val, y_pred_val, average='binary')
            
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | "
                  f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(config['experiment']['output_dir'], 'best_model.pt'))
                print(f"  -> Best model saved (F1: {best_val_f1:.4f})")
            
            # Early stopping
            if early_stopping(-val_f1):
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"Phase 3 completed. Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}")


def train_single_run(data, config, run_seed, device):
    """
    Train GCL-IoT for a single run with a specific seed
    """
    print(f"\n{'='*60}")
    print(f"Run with seed: {run_seed}")
    print('='*60)
    
    # Set seed for reproducibility
    set_seed(run_seed, deterministic=(run_seed <= 456))
    
    # Get data statistics
    stats = get_dataset_statistics(data)
    print(f"Dataset statistics: {stats}")
    
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
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Phase 1: Reinforced neighbor selection
    top_k_neighbors = train_phase1(model, data, config, device)
    
    # Phase 2: Contrastive pre-training
    train_phase2(model, data, config, device)
    
    # Phase 3: Supervised fine-tuning
    train_phase3(model, data, config, device)
    
    # Load best model
    best_model_path = os.path.join(config['experiment']['output_dir'], 'best_model.pt')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    
    # Evaluate
    metrics, y_true, y_pred = compute_node_metrics(model, data, device)
    
    return metrics, y_true, y_pred


def main():
    parser = argparse.ArgumentParser(description='Train GCL-IoT')
    parser.add_argument('--config', type=str, default='configs/gcl_iot.yaml', help='Config file path')
    parser.add_argument('--dataset', type=str, default='ton_iot', choices=['ton_iot', 'im_iot'], help='Dataset name')
    parser.add_argument('--data_path', type=str, default='./data/raw/', help='Raw data path')
    parser.add_argument('--output_dir', type=str, default='./outputs/', help='Output directory')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of independent runs')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config['dataset']['name'] = args.dataset
    config['experiment']['output_dir'] = args.output_dir
    config['experiment']['num_runs'] = args.num_runs
    config['experiment']['seed'] = args.seed
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data_path = os.path.join(args.data_path, f"{args.dataset}.csv")
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please place the dataset file in the correct location.")
        sys.exit(1)
    
    data = load_iot_data(args.dataset, data_path, 
                         feature_dim=config['dataset']['feature_dim'],
                         temporal_split=config['dataset']['temporal_split'])
    
    print(f"Data loaded: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges")
    
    # Track results across runs
    all_metrics = []
    
    # Run multiple independent runs
    seeds = [args.seed + i for i in range(args.num_runs)]
    
    for run_idx, seed in enumerate(seeds):
        print(f"\n{'#'*60}")
        print(f"Starting run {run_idx + 1}/{args.num_runs} (seed: {seed})")
        print('#'*60)
        
        metrics, y_true, y_pred = train_single_run(data, config, seed, device)
        
        print(f"\nRun {run_idx + 1} results:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
        
        all_metrics.append(metrics)
        
        # Save run results
        run_path = os.path.join(args.output_dir, f"run_{run_idx + 1}.pt")
        torch.save({'metrics': metrics, 'y_true': y_true, 'y_pred': y_pred}, run_path)
    
    # Compute aggregate statistics
    print("\n" + "="*60)
    print("AGGREGATE RESULTS (over all runs)")
    print("="*60)
    
    aggregate = {}
    for key in all_metrics[0].keys():
        if isinstance(all_metrics[0][key], (int, float)):
            values = [m[key] for m in all_metrics]
            mean = np.mean(values)
            std = np.std(values)
            aggregate[key] = (mean, std)
            print(f"{key}: {mean:.4f} ± {std:.4f}")
    
    # Save aggregate results
    results_path = os.path.join(args.output_dir, 'aggregate_results.pt')
    torch.save(aggregate, results_path)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()