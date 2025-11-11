#!/usr/bin/env python3
"""Example ML pipeline for test prioritization using extracted features.

This demonstrates how to:
1. Load enhanced features from JSONL
2. Prepare data for training
3. Train a simple test prioritization model
4. Evaluate and use the model
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse


def load_data(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load enhanced features from JSONL."""
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def prepare_features(records: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[int], List[str]]:
    """Convert records to feature matrix and labels.

    Returns:
        features: List of feature vectors (numeric only)
        labels: List of binary labels (0/1 for coverage contribution)
        test_names: List of test names for tracking
    """
    features = []
    labels = []
    test_names = []

    for record in records:
        # Extract label
        label = record.get("label", 0)

        # Build feature vector (numeric features only)
        feat_vec = []

        # Derived features (most important for prioritization)
        derived = record.get("derived_features", {})
        feat_vec.extend([
            derived.get("branch_ratio", 0.0),
            derived.get("load_ratio", 0.0),
            derived.get("store_ratio", 0.0),
            derived.get("memory_ratio", 0.0),
            derived.get("jump_ratio", 0.0),
            derived.get("system_ratio", 0.0),
            derived.get("arithmetic_ratio", 0.0),
            derived.get("control_flow_complexity", 0.0),
            derived.get("cpi", 2.0) if derived.get("cpi") else 2.0,  # Default CPI
            derived.get("exception_rate", 0.0),
        ])

        # Execution features
        exec_feat = record.get("execution_features", {})
        feat_vec.extend([
            float(exec_feat.get("trace_instruction_count", 0)),
            float(exec_feat.get("trace_branch_count", 0)),
            float(exec_feat.get("trace_load_count", 0)),
            float(exec_feat.get("trace_store_count", 0)),
            float(exec_feat.get("trace_unique_pcs", 0)),
            float(exec_feat.get("exception_count", 0)),
            float(exec_feat.get("log_line_count", 0)),
        ])

        # Test metadata (encode test type as numeric)
        meta = record.get("test_metadata", {})
        test_type_numeric = 1.0 if meta.get("test_type") == "RISCVDV" else 0.0
        feat_vec.extend([
            test_type_numeric,
            float(meta.get("seed", 0)),
            float(meta.get("gen_opts_count", 0)),
            float(meta.get("sim_opts_count", 0)),
        ])

        # Coverage metrics before (important context)
        metrics_before = record.get("metrics_before", {})
        feat_vec.extend([
            metrics_before.get("block", 0.0),
            metrics_before.get("branch", 0.0),
        ])

        features.append(feat_vec)
        labels.append(label)
        test_names.append(record.get("testdotseed", "unknown"))

    return features, labels, test_names


def train_simple_model(X_train, y_train, X_test, y_test):
    """Train a simple random forest classifier."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
        import numpy as np
    except ImportError:
        print("Error: scikit-learn is required for ML training.")
        print("Install with: pip install scikit-learn")
        return None

    print("\nTraining Random Forest classifier...")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(X_train[0]) if X_train else 0}")

    # Train model
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced'  # Handle imbalanced labels
    )
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # Evaluate
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Contribution", "Contributes"]))

    if len(set(y_test)) > 1:
        auc = roc_auc_score(y_test, y_prob)
        print(f"\nROC AUC Score: {auc:.4f}")

    # Feature importance
    feature_names = [
        "branch_ratio", "load_ratio", "store_ratio", "memory_ratio",
        "jump_ratio", "system_ratio", "arithmetic_ratio", "control_flow_complexity",
        "cpi", "exception_rate",
        "trace_instruction_count", "trace_branch_count", "trace_load_count",
        "trace_store_count", "trace_unique_pcs", "exception_count", "log_line_count",
        "test_type", "seed", "gen_opts_count", "sim_opts_count",
        "block_cov_before", "branch_cov_before"
    ]

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nTop 10 Most Important Features:")
    for i in range(min(10, len(indices))):
        idx = indices[i]
        feat_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        print(f"  {i+1}. {feat_name}: {importances[idx]:.4f}")

    return clf, y_prob


def demonstrate_prioritization(clf, X_test, y_test, test_names, top_k=20):
    """Demonstrate test prioritization using the trained model."""
    try:
        import numpy as np
    except ImportError:
        print("Error: numpy is required.")
        return

    # Get predicted probabilities
    y_prob = clf.predict_proba(X_test)[:, 1]

    # Sort tests by predicted probability of contribution (descending)
    indices = np.argsort(y_prob)[::-1]

    print("\n" + "=" * 80)
    print(f"TEST PRIORITIZATION - Top {top_k} Tests")
    print("=" * 80)
    print("\nTests ranked by predicted coverage contribution:")
    print(f"{'Rank':<6} {'Test Name':<45} {'Prob':<8} {'Actual':<8}")
    print("-" * 80)

    actual_contributions_in_top_k = 0
    for i, idx in enumerate(indices[:top_k], 1):
        test_name = test_names[idx]
        prob = y_prob[idx]
        actual = "✓" if y_test[idx] == 1 else "✗"

        if y_test[idx] == 1:
            actual_contributions_in_top_k += 1

        print(f"{i:<6} {test_name:<45} {prob:.4f}   {actual:<8}")

    # Calculate effectiveness
    total_contributions = sum(y_test)
    if total_contributions > 0:
        recall_at_k = actual_contributions_in_top_k / total_contributions
        precision_at_k = actual_contributions_in_top_k / top_k

        print("\n" + "-" * 80)
        print(f"Effectiveness of Top-{top_k} Selection:")
        print(f"  Tests with actual contribution: {actual_contributions_in_top_k}/{top_k}")
        print(f"  Precision@{top_k}: {precision_at_k:.2%}")
        print(f"  Recall@{top_k}: {recall_at_k:.2%}")
        print(f"  (Found {actual_contributions_in_top_k} of {total_contributions} contributing tests)")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train test prioritization model from enhanced features."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input JSONL file with enhanced features.",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data to use for training (default: 0.8).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top tests to show in prioritization (default: 20).",
    )

    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Error: Input file {args.input} does not exist.")
        return 1

    # Check dependencies
    try:
        import numpy as np
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("Error: This example requires scikit-learn and numpy.")
        print("Install with: pip install scikit-learn numpy")
        return 1

    # Load data
    print(f"Loading data from {args.input}...")
    records = load_data(args.input)
    print(f"Loaded {len(records)} records.")

    # Prepare features
    print("\nPreparing features...")
    X, y, test_names = prepare_features(records)

    if not X:
        print("Error: No features extracted.")
        return 1

    # Split train/test
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X, y, test_names, train_size=args.train_split, random_state=42, stratify=y
    )

    # Train model
    clf, y_prob = train_simple_model(X_train, y_train, X_test, y_test)

    if clf is None:
        return 1

    # Demonstrate prioritization
    demonstrate_prioritization(clf, X_test, y_test, names_test, top_k=args.top_k)

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. Tune hyperparameters (grid search, cross-validation)
2. Try different models (XGBoost, LightGBM, Neural Networks)
3. Add feature engineering (polynomial features, interactions)
4. Collect more training data across multiple regressions
5. Add temporal features (historical test performance)
6. Deploy model to prioritize tests in CI/CD pipeline

Example deployment:
  - Save model: pickle.dump(clf, open('model.pkl', 'wb'))
  - In CI/CD: Load model, predict on new tests, run top-K first
  - Monitor: Track actual vs predicted contribution over time
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
