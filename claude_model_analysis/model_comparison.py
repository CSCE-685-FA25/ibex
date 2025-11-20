#!/usr/bin/env python3
"""
Compare multiple ML models for predicting high-value coverage tests.

This script trains and evaluates multiple classifiers on the RISC-V test coverage
prediction task, providing comprehensive performance metrics and comparisons.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    make_scorer
)
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    cross_val_score,
    GridSearchCV
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek

import joblib
import warnings
warnings.filterwarnings('ignore')


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file into a list of dictionaries."""
    data = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def prepare_features(
    rows: List[Dict],
    include_coverage: bool = False,
    feature_selection: Optional[str] = None,
    n_features: int = 30
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract features and labels from the data."""
    
    # Determine which features to use
    skip_prefixes = [] if include_coverage else ["delta_", "total_delta", "covergroup_delta"]
    skip_names = {"label", "testdotseed", "coverage_path"}
    
    feature_keys = []
    for row in rows:
        for key, value in row.items():
            if key in skip_names:
                continue
            if any(key.startswith(prefix) for prefix in skip_prefixes):
                continue
            if isinstance(value, (int, float, bool)) and key not in feature_keys:
                feature_keys.append(key)
    
    feature_keys = sorted(feature_keys)
    
    # Build feature matrix and labels
    X = np.zeros((len(rows), len(feature_keys)), dtype=np.float64)
    y = np.zeros(len(rows), dtype=np.int32)
    
    for idx, row in enumerate(rows):
        y[idx] = int(row.get("label", 0))
        for col, key in enumerate(feature_keys):
            value = row.get(key, 0.0)
            if isinstance(value, bool):
                X[idx, col] = float(value)
            elif isinstance(value, (int, float)):
                X[idx, col] = float(value)
            else:
                X[idx, col] = 0.0
    
    # Feature selection if requested
    if feature_selection and n_features < len(feature_keys):
        if feature_selection == "kbest":
            selector = SelectKBest(f_classif, k=n_features)
        elif feature_selection == "mutual_info":
            selector = SelectKBest(mutual_info_classif, k=n_features)
        else:
            selector = None
        
        if selector:
            X = selector.fit_transform(X, y)
            selected_mask = selector.get_support()
            feature_keys = [feat for feat, selected in zip(feature_keys, selected_mask) if selected]
    
    return X, y, feature_keys


def handle_imbalance(X_train, y_train, method: str = "none"):
    """Handle class imbalance using various techniques."""
    if method == "smote":
        sampler = SMOTE(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    elif method == "adasyn":
        sampler = ADASYN(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    elif method == "undersample":
        sampler = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    elif method == "smoteenn":
        sampler = SMOTEENN(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    elif method == "smotetomek":
        sampler = SMOTETomek(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    else:
        X_resampled, y_resampled = X_train, y_train
    
    return X_resampled, y_resampled


def get_models() -> Dict:
    """Define all models to compare."""
    models = {
        # Ensemble Methods (typically best for tabular data)
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=42
        ),
        
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            scale_pos_weight=5,  # Helps with imbalanced data
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        ),
        
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=-1,
            num_leaves=31,
            class_weight="balanced",
            random_state=42,
            verbosity=-1
        ),
        
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100,
            learning_rate=1.0,
            random_state=42
        ),
        
        # Linear Models
        "Logistic Regression": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ),
        
        # Support Vector Machine
        "SVM (RBF)": SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight="balanced",
            probability=True,
            random_state=42
        ),
        
        # Neural Network
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        ),
        
        # Simple baseline models
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42
        ),
        
        "Gaussian Naive Bayes": GaussianNB(),
        
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5,
            weights='distance'
        )
    }
    
    return models


def evaluate_model(model, X_test, y_test, model_name: str) -> Dict:
    """Evaluate a single model and return metrics."""
    start_time = time.time()
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get probability predictions if available
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred
    
    train_time = time.time() - start_time
    
    # Calculate metrics
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "avg_precision": average_precision_score(y_test, y_proba),
        "inference_time": train_time
    }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics["true_positives"] = tp
    metrics["false_positives"] = fp
    metrics["true_negatives"] = tn
    metrics["false_negatives"] = fn
    
    # Additional useful metrics for imbalanced data
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    metrics["fnr"] = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    
    return metrics, y_proba


def cross_validate_model(model, X, y, cv_folds: int = 5) -> Dict:
    """Perform cross-validation for a model."""
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score),
        'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
    }
    
    cv_results = {}
    for metric_name, scorer in scoring.items():
        try:
            scores = cross_val_score(
                model, X, y, 
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                scoring=scorer,
                n_jobs=-1
            )
            cv_results[f"cv_{metric_name}_mean"] = scores.mean()
            cv_results[f"cv_{metric_name}_std"] = scores.std()
        except:
            cv_results[f"cv_{metric_name}_mean"] = 0
            cv_results[f"cv_{metric_name}_std"] = 0
    
    return cv_results


def plot_results(results_df: pd.DataFrame, output_dir: Path):
    """Create visualization of model comparison results."""
    
    # Set style
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Accuracy comparison
    ax = axes[0, 0]
    results_df.sort_values('accuracy').plot(x='model', y='accuracy', kind='barh', ax=ax)
    ax.set_title('Model Accuracy Comparison')
    ax.set_xlabel('Accuracy')
    
    # 2. F1 Score comparison
    ax = axes[0, 1]
    results_df.sort_values('f1').plot(x='model', y='f1', kind='barh', ax=ax)
    ax.set_title('F1 Score Comparison')
    ax.set_xlabel('F1 Score')
    
    # 3. ROC-AUC comparison
    ax = axes[0, 2]
    results_df.sort_values('roc_auc').plot(x='model', y='roc_auc', kind='barh', ax=ax)
    ax.set_title('ROC-AUC Comparison')
    ax.set_xlabel('ROC-AUC')
    
    # 4. Precision vs Recall
    ax = axes[1, 0]
    ax.scatter(results_df['recall'], results_df['precision'])
    for idx, row in results_df.iterrows():
        ax.annotate(row['model'], (row['recall'], row['precision']), 
                   fontsize=8, rotation=45)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision vs Recall Trade-off')
    ax.grid(True)
    
    # 5. False Positive vs False Negative rates
    ax = axes[1, 1]
    ax.scatter(results_df['fpr'], results_df['fnr'])
    for idx, row in results_df.iterrows():
        ax.annotate(row['model'], (row['fpr'], row['fnr']), 
                   fontsize=8, rotation=45)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('False Negative Rate')
    ax.set_title('FPR vs FNR Trade-off')
    ax.grid(True)
    
    # 6. Metrics heatmap
    ax = axes[1, 2]
    metrics_subset = results_df[['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']].set_index('model')
    sns.heatmap(metrics_subset.T, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
    ax.set_title('Performance Metrics Heatmap')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


def plot_roc_curves(models_dict: Dict, X_test, y_test, output_dir: Path):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    
    for model_name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(output_dir / 'roc_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


def feature_importance_analysis(models_dict: Dict, feature_keys: List[str], output_dir: Path):
    """Analyze and visualize feature importance for tree-based models."""
    importance_data = {}
    
    for model_name, model in models_dict.items():
        if hasattr(model, 'feature_importances_'):
            importance_data[model_name] = model.feature_importances_
    
    if importance_data:
        # Create DataFrame
        importance_df = pd.DataFrame(importance_data, index=feature_keys)
        
        # Plot top 20 features for each model
        fig, axes = plt.subplots(1, min(3, len(importance_data)), 
                                 figsize=(6*min(3, len(importance_data)), 8))
        
        if len(importance_data) == 1:
            axes = [axes]
        
        for idx, (model_name, importances) in enumerate(list(importance_data.items())[:3]):
            top_features = importance_df[model_name].nlargest(20)
            top_features.plot(kind='barh', ax=axes[idx])
            axes[idx].set_title(f'Top 20 Features - {model_name}')
            axes[idx].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Save importance data
        importance_df.to_csv(output_dir / 'feature_importance.csv')


def hyperparameter_tuning(X_train, y_train, X_test, y_test):
    """Perform hyperparameter tuning for the best models."""
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING FOR TOP MODELS")
    print("="*80)
    
    # Define parameter grids for top models
    param_grids = {
        "XGBoost": {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'scale_pos_weight': [1, 5, 10]
        },
        "LightGBM": {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.3],
            'num_leaves': [15, 31, 63],
            'min_child_samples': [5, 10, 20],
            'subsample': [0.7, 0.8, 0.9]
        },
        "Random Forest": {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }
    }
    
    tuned_models = {}
    
    for model_name, param_grid in param_grids.items():
        print(f"\nTuning {model_name}...")
        
        if model_name == "XGBoost":
            base_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        elif model_name == "LightGBM":
            base_model = LGBMClassifier(random_state=42, verbosity=-1, class_weight="balanced")
        else:  # Random Forest
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")
        
        # Evaluate on test set
        y_pred = grid_search.best_estimator_.predict(X_test)
        test_f1 = f1_score(y_test, y_pred)
        print(f"Test F1 Score: {test_f1:.4f}")
        
        tuned_models[model_name] = grid_search.best_estimator_
    
    return tuned_models


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Compare multiple ML models for coverage prediction")
    parser.add_argument("--features", type=Path, default=Path("coverage_features.jsonl"),
                       help="Path to feature file")
    parser.add_argument("--output-dir", type=Path, default=Path("model_comparison_results"),
                       help="Directory for output files")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Test set proportion")
    parser.add_argument("--cv-folds", type=int, default=5,
                       help="Number of cross-validation folds")
    parser.add_argument("--include-coverage", action="store_true",
                       help="Include coverage delta features")
    parser.add_argument("--balance-method", choices=["none", "smote", "adasyn", "undersample", 
                                                     "smoteenn", "smotetomek"],
                       default="none", help="Method to handle class imbalance")
    parser.add_argument("--scale", action="store_true",
                       help="Apply feature scaling")
    parser.add_argument("--feature-selection", choices=["none", "kbest", "mutual_info"],
                       default="none", help="Feature selection method")
    parser.add_argument("--n-features", type=int, default=30,
                       help="Number of features to select")
    parser.add_argument("--tune-hyperparameters", action="store_true",
                       help="Perform hyperparameter tuning for top models")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    data = load_jsonl(args.features)
    print(f"Loaded {len(data)} samples")
    
    # Prepare features
    print("\nPreparing features...")
    X, y, feature_keys = prepare_features(
        data,
        include_coverage=args.include_coverage,
        feature_selection=args.feature_selection if args.feature_selection != "none" else None,
        n_features=args.n_features
    )
    print(f"Using {len(feature_keys)} features")
    print(f"Feature names: {', '.join(feature_keys[:10])}...")
    
    # Check class balance
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nClass distribution:")
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} samples ({count/len(y):.2%})")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    # Handle class imbalance
    if args.balance_method != "none":
        print(f"\nApplying {args.balance_method} for class balancing...")
        X_train, y_train = handle_imbalance(X_train, y_train, args.balance_method)
        print(f"Training set after balancing: {len(y_train)} samples")
    
    # Apply scaling if requested
    scaler = None
    if args.scale:
        print("\nApplying feature scaling...")
        scaler = RobustScaler()  # RobustScaler is better for outliers
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Get models
    models = get_models()
    
    # Train and evaluate models
    print("\n" + "="*80)
    print("TRAINING AND EVALUATING MODELS")
    print("="*80)
    
    results = []
    trained_models = {}
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        print("-" * 40)
        
        try:
            # Train model
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Evaluate
            metrics, y_proba = evaluate_model(model, X_test, y_test, model_name)
            metrics['train_time'] = train_time
            
            # Cross-validation (optional, takes time)
            if args.cv_folds > 1:
                print(f"  Performing {args.cv_folds}-fold cross-validation...")
                cv_results = cross_validate_model(model, X_train, y_train, args.cv_folds)
                metrics.update(cv_results)
            
            results.append(metrics)
            trained_models[model_name] = model
            
            # Print summary
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"  Training time: {train_time:.2f}s")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1', ascending=False)
    
    # Print summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY (sorted by F1 Score)")
    print("="*80)
    print(results_df[['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']].to_string(index=False))
    
    # Save results
    results_df.to_csv(args.output_dir / 'model_comparison_results.csv', index=False)
    print(f"\nResults saved to {args.output_dir / 'model_comparison_results.csv'}")
    
    # Hyperparameter tuning for top models
    if args.tune_hyperparameters:
        tuned_models = hyperparameter_tuning(X_train, y_train, X_test, y_test)
        
        # Save tuned models
        for model_name, model in tuned_models.items():
            model_data = {
                'model': model,
                'feature_keys': feature_keys,
                'scaler': scaler
            }
            joblib.dump(model_data, args.output_dir / f'tuned_{model_name.lower().replace(" ", "_")}.joblib')
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_results(results_df, args.output_dir)
    plot_roc_curves(trained_models, X_test, y_test, args.output_dir)
    feature_importance_analysis(trained_models, feature_keys, args.output_dir)
    
    # Save best model
    best_model_name = results_df.iloc[0]['model']
    best_model = trained_models[best_model_name]
    model_data = {
        'model': best_model,
        'feature_keys': feature_keys,
        'scaler': scaler,
        'metrics': results_df.iloc[0].to_dict()
    }
    joblib.dump(model_data, args.output_dir / 'best_model.joblib')
    print(f"\nBest model ({best_model_name}) saved to {args.output_dir / 'best_model.joblib'}")
    
    # Print recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print(f"1. Best performing model: {best_model_name}")
    print(f"   - F1 Score: {results_df.iloc[0]['f1']:.4f}")
    print(f"   - ROC-AUC: {results_df.iloc[0]['roc_auc']:.4f}")
    
    # Identify models good at different trade-offs
    high_precision = results_df.nlargest(3, 'precision')['model'].iloc[0]
    high_recall = results_df.nlargest(3, 'recall')['model'].iloc[0]
    
    print(f"\n2. For minimizing false positives (high precision): {high_precision}")
    print(f"3. For minimizing false negatives (high recall): {high_recall}")
    
    if args.balance_method == "none":
        print("\n4. Consider trying with --balance-method smote for better handling of class imbalance")
    
    if not args.tune_hyperparameters:
        print("\n5. Run with --tune-hyperparameters for optimized model performance")
    
    print("\n" + "="*80)
    print("Model comparison complete!")


if __name__ == "__main__":
    main()
