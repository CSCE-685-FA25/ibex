#!/usr/bin/env python3
"""
Quick model comparison for RISC-V test coverage prediction.
Focuses on the most promising models for imbalanced classification.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import time
from typing import Dict, List, Tuple

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, average_precision_score
)
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

import joblib
import warnings
warnings.filterwarnings('ignore')


def load_data(path: Path) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with path.open("r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def prepare_features(data: List[Dict], include_coverage: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract features and labels."""
    
    # Determine features to use
    skip_prefixes = [] if include_coverage else ["delta_", "total_delta", "covergroup_delta"]
    skip_names = {"label", "testdotseed", "coverage_path"}
    
    # Find all numeric features
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
    X = np.zeros((len(data), len(feature_keys)), dtype=np.float64)
    y = np.zeros(len(data), dtype=np.int32)
    
    for idx, row in enumerate(data):
        y[idx] = int(row.get("label", 0))
        for col, key in enumerate(feature_keys):
            value = row.get(key, 0.0)
            X[idx, col] = float(value) if isinstance(value, (int, float, bool)) else 0.0
    
    return X, y, feature_keys


def get_quick_models() -> Dict:
    """Get a focused set of models for comparison."""
    
    models = {
        # Best ensemble methods for imbalanced data
        "Random Forest (balanced)": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        
        "Extra Trees (balanced)": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        
        # With custom class weights for extreme imbalance
        "Random Forest (weighted)": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight={0: 1, 1: 5},  # Higher weight for minority class
            random_state=42,
            n_jobs=-1
        ),
        
        # Linear models as baseline
        "Logistic Regression (L2)": LogisticRegression(
            C=1.0,
            penalty='l2',
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),
        
        "Logistic Regression (L1)": LogisticRegression(
            C=1.0,
            penalty='l1',
            solver='liblinear',
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),
        
        # Simple baseline
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42
        ),
    }
    
    return models


def evaluate_model(model, X_test, y_test) -> Dict:
    """Comprehensive evaluation of a model."""
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Probabilities (if available)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred
    
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'avg_precision': average_precision_score(y_test, y_proba)
    }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Additional metrics for imbalanced data
    metrics.update({
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,
        'balanced_accuracy': (metrics['recall'] + metrics.get('specificity', 0)) / 2
    })
    
    return metrics


def feature_importance_summary(model, feature_keys: List[str], top_k: int = 15) -> List[Tuple[str, float]]:
    """Extract feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_k]
        return [(feature_keys[i], importances[i]) for i in indices]
    return []


def main():
    parser = argparse.ArgumentParser(description="Quick model comparison for coverage prediction")
    parser.add_argument("--features", type=Path, default=Path("coverage_features.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("quick_results"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--include-coverage", action="store_true")
    parser.add_argument("--scale", action="store_true")
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    data = load_data(args.features)
    print(f"Loaded {len(data)} samples")
    
    # Prepare features
    X, y, feature_keys = prepare_features(data, args.include_coverage)
    print(f"Using {len(feature_keys)} features")
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} ({count/len(y):.1%})")
    
    # Calculate class weights
    classes = np.unique(y)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y)
    class_weights_dict = dict(zip(classes, weights))
    print(f"\nComputed class weights: {class_weights_dict}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(y_train)} samples")
    print(f"Test set: {len(y_test)} samples")
    
    # Scale if requested
    scaler = None
    if args.scale:
        print("\nApplying feature scaling...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Get models
    models = get_quick_models()
    
    # Training and evaluation
    print("\n" + "="*80)
    print("MODEL TRAINING AND EVALUATION")
    print("="*80)
    
    results = []
    best_f1 = 0
    best_model = None
    best_model_name = None
    
    for model_name, model in models.items():
        print(f"\n{model_name}")
        print("-" * 40)
        
        # Train
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        metrics['model'] = model_name
        metrics['train_time'] = train_time
        
        # Cross-validation for F1 score
        if args.cv_folds > 1:
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42),
                scoring='f1',
                n_jobs=-1
            )
            metrics['cv_f1_mean'] = cv_scores.mean()
            metrics['cv_f1_std'] = cv_scores.std()
        
        results.append(metrics)
        
        # Print key metrics
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        
        if args.cv_folds > 1:
            print(f"  CV F1:     {metrics['cv_f1_mean']:.4f} (±{metrics['cv_f1_std']:.4f})")
        
        print(f"  Training time: {train_time:.2f}s")
        
        # Track best model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model = model
            best_model_name = model_name
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1', ascending=False)
    
    # Summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY (sorted by F1 Score)")
    print("="*80)
    
    summary_cols = ['model', 'f1', 'precision', 'recall', 'roc_auc', 'balanced_accuracy']
    if args.cv_folds > 1:
        summary_cols.append('cv_f1_mean')
    
    print(results_df[summary_cols].to_string(index=False))
    
    # Confusion matrix for best model
    print("\n" + "="*80)
    print(f"BEST MODEL: {best_model_name}")
    print("="*80)
    
    y_pred = best_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Coverage', 'Adds Coverage']))
    
    # Feature importance for best model
    if hasattr(best_model, 'feature_importances_'):
        print("\nTop 15 Most Important Features:")
        top_features = feature_importance_summary(best_model, feature_keys, top_k=15)
        for rank, (feature, importance) in enumerate(top_features, 1):
            print(f"  {rank:2d}. {feature:30s} {importance:.4f}")
    
    # Save results
    results_df.to_csv(args.output_dir / 'comparison_results.csv', index=False)
    print(f"\nResults saved to {args.output_dir / 'comparison_results.csv'}")
    
    # Save best model
    model_data = {
        'model': best_model,
        'feature_keys': feature_keys,
        'scaler': scaler,
        'metrics': results_df.iloc[0].to_dict()
    }
    joblib.dump(model_data, args.output_dir / 'best_model.joblib')
    print(f"Best model saved to {args.output_dir / 'best_model.joblib'}")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    print(f"1. Best overall model: {best_model_name}")
    print(f"   - F1 Score: {best_f1:.4f}")
    print(f"   - Best for balanced precision/recall trade-off")
    
    # Find models with highest precision and recall
    high_precision_model = results_df.nlargest(1, 'precision')['model'].iloc[0]
    high_recall_model = results_df.nlargest(1, 'recall')['model'].iloc[0]
    
    if high_precision_model != best_model_name:
        print(f"\n2. For minimizing false positives: {high_precision_model}")
        print(f"   - Precision: {results_df[results_df['model']==high_precision_model]['precision'].iloc[0]:.4f}")
    
    if high_recall_model != best_model_name:
        print(f"\n3. For catching most coverage-adding tests: {high_recall_model}")
        print(f"   - Recall: {results_df[results_df['model']==high_recall_model]['recall'].iloc[0]:.4f}")
    
    # Practical recommendations
    print("\n4. Practical deployment suggestions:")
    print("   - Use probability thresholds to tune precision/recall trade-off")
    print("   - Consider ensemble of top 3 models for robustness")
    print("   - Monitor performance on new test patterns")
    
    if not args.include_coverage:
        print("\n5. Try with --include-coverage to see if coverage deltas improve predictions")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
