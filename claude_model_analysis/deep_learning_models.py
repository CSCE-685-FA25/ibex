#!/usr/bin/env python3
"""
Deep learning models for RISC-V test coverage prediction.

This script implements and compares various deep learning architectures
including standard neural networks, attention mechanisms, and ensemble methods.
Fixed version with better compatibility.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


class TabularDataset(Dataset):
    """Custom dataset for tabular data."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BasicNN(nn.Module):
    """Basic fully connected neural network."""
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout=0.3):
        super(BasicNN, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 2))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):
    """Residual block for deep networks."""
    def __init__(self, size, dropout=0.3):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(size, size)
        self.bn1 = nn.BatchNorm1d(size)
        self.fc2 = nn.Linear(size, size)
        self.bn2 = nn.BatchNorm1d(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += residual
        return F.relu(out)


class ResNet(nn.Module):
    """Residual Network for tabular data."""
    def __init__(self, input_size, hidden_size=128, num_blocks=3, dropout=0.3):
        super(ResNet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout) for _ in range(num_blocks)
        ])

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.residual_blocks:
            x = block(x)
        return self.output_layer(x)


class SimpleAttentionNN(nn.Module):
    """Simplified neural network with attention mechanism."""
    def __init__(self, input_size, hidden_size=128, dropout=0.3):
        super(SimpleAttentionNN, self).__init__()

        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Attention weights
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.Sigmoid()
        )

        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        attention = self.attention_weights(features)
        weighted_features = features * attention
        return self.output_layer(weighted_features)


class GatedNN(nn.Module):
    """Neural network with gating mechanism."""
    def __init__(self, input_size, hidden_sizes=[128, 64], dropout=0.3):
        super(GatedNN, self).__init__()

        self.gates = nn.ModuleList()
        self.transforms = nn.ModuleList()

        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.gates.append(nn.Sequential(
                nn.Linear(prev_size, hidden_size),
                nn.Sigmoid()
            ))
            self.transforms.append(nn.Sequential(
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            prev_size = hidden_size

        self.output_layer = nn.Linear(prev_size, 2)

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            g = gate(x)
            t = transform(x)
            x = g * t
        return self.output_layer(x)


class SimpleEnsembleNN(nn.Module):
    """Ensemble of simple neural networks."""
    def __init__(self, input_size):
        super(SimpleEnsembleNN, self).__init__()

        # Three different architectures
        self.model1 = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(50, 2)
        )

        self.model2 = nn.Sequential(
            nn.Linear(input_size, 80),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(40, 2)
        )

        self.model3 = nn.Sequential(
            nn.Linear(input_size, 120),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(120, 60),
            nn.ReLU(),
            nn.Linear(60, 2)
        )

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        out3 = self.model3(x)
        # Average predictions
        return (out1 + out2 + out3) / 3


def load_data(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load and prepare data."""
    data = []
    with path.open("r") as f:
        for line in f:
            data.append(json.loads(line))

    # Extract features (excluding coverage deltas by default)
    skip_prefixes = ["delta_", "total_delta", "covergroup_delta"]
    skip_names = {"label", "testdotseed", "coverage_path"}

    feature_keys = []
    for row in data:
        for key, value in row.items():
            if key in skip_names:
                continue
            if any(key.startswith(prefix) for prefix in skip_prefixes):
                continue
            if isinstance(value, (int, float, bool)) and key not in feature_keys:
                feature_keys.append(key)

    feature_keys = sorted(feature_keys)

    # Build arrays
    X = np.zeros((len(data), len(feature_keys)), dtype=np.float32)
    y = np.zeros(len(data), dtype=np.int32)

    for idx, row in enumerate(data):
        y[idx] = int(row.get("label", 0))
        for col, key in enumerate(feature_keys):
            value = row.get(key, 0.0)
            if isinstance(value, bool):
                X[idx, col] = float(value)
            else:
                X[idx, col] = float(value)

    return X, y, feature_keys


def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001):
    """Train a neural network model."""

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0]).to(device))  # Weight for imbalanced data
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Try to create scheduler without verbose parameter (for compatibility)
    try:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    except TypeError:
        # Fallback for older PyTorch versions
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_f1 = 0
    best_model_state = None
    early_stop_patience = 15
    early_stop_counter = 0

    train_losses = []
    val_f1_scores = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_preds = []
        val_true = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)

                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())

        # Calculate metrics
        val_f1 = f1_score(val_true, val_preds)
        val_acc = accuracy_score(val_true, val_preds)

        train_losses.append(train_loss / len(train_loader))
        val_f1_scores.append(val_f1)

        # Learning rate scheduling
        scheduler.step(val_f1)

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}")

        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_f1_scores


def evaluate_model(model, test_loader, device) -> Dict:
    """Evaluate a trained model."""
    model.eval()
    predictions = []
    probabilities = []
    true_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            predictions.extend(predicted.cpu().numpy())
            probabilities.extend(probs[:, 1].cpu().numpy())
            true_labels.extend(batch_y.cpu().numpy())

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions, zero_division=0),
        'recall': recall_score(true_labels, predictions),
        'f1': f1_score(true_labels, predictions),
        'roc_auc': roc_auc_score(true_labels, probabilities) if len(np.unique(true_labels)) > 1 else 0
    }

    # Confusion matrix
    if len(np.unique(true_labels)) > 1:
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
        metrics.update({
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        })

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Deep learning models for coverage prediction")
    parser.add_argument("--features", type=Path, default=Path("coverage_features.jsonl"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=Path("dl_results"))

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    X, y, feature_keys = load_data(args.features)
    print(f"Loaded {len(X)} samples with {len(feature_keys)} features")

    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    # Further split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Create data loaders
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    test_dataset = TabularDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    input_size = X_train.shape[1]

    # Define models (simplified set that should work)
    models = {
        "Basic NN": BasicNN(input_size, [128, 64, 32], dropout=0.3),
        "ResNet": ResNet(input_size, hidden_size=128, num_blocks=3, dropout=0.3),
        "Simple Attention": SimpleAttentionNN(input_size, hidden_size=128, dropout=0.3),
        "Gated NN": GatedNN(input_size, [128, 64], dropout=0.3),
        "Ensemble": SimpleEnsembleNN(input_size)
    }

    results = []

    print("\n" + "="*80)
    print("TRAINING DEEP LEARNING MODELS")
    print("="*80)

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        print("-" * 40)

        try:
            model = model.to(device)

            # Count parameters
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Number of parameters: {n_params:,}")

            # Train model
            start_time = time.time()
            trained_model, train_losses, val_f1s = train_model(
                model, train_loader, val_loader, device,
                epochs=args.epochs, lr=args.learning_rate
            )
            train_time = time.time() - start_time

            # Evaluate on test set
            metrics = evaluate_model(trained_model, test_loader, device)
            metrics['model'] = model_name
            metrics['parameters'] = n_params
            metrics['train_time'] = train_time

            results.append(metrics)

            print(f"Test Performance:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

            # Save model
            torch.save({
                'model_state_dict': trained_model.state_dict(),
                'scaler': scaler,
                'feature_keys': feature_keys,
                'metrics': metrics
            }, args.output_dir / f"{model_name.replace(' ', '_').lower()}.pth")

        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue

    # Create results DataFrame
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('f1', ascending=False)

        # Print summary
        print("\n" + "="*80)
        print("RESULTS SUMMARY (Deep Learning Models)")
        print("="*80)
        print(results_df[['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']].to_string(index=False))

        # Save results
        results_df.to_csv(args.output_dir / 'dl_results.csv', index=False)
        print(f"\nResults saved to {args.output_dir / 'dl_results.csv'}")

        # Best model
        best_model = results_df.iloc[0]
        print(f"\nBest performing model: {best_model['model']}")
        print(f"  F1 Score: {best_model['f1']:.4f}")
        print(f"  ROC-AUC: {best_model['roc_auc']:.4f}")
    else:
        print("\nNo models were successfully trained.")


if __name__ == "__main__":
    main()
