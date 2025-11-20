# RISC-V Test Coverage Prediction - Model Comparison Suite

## Overview
This suite provides multiple approaches for comparing machine learning models to predict which RISC-V tests will add coverage. Your dataset shows significant class imbalance (16% positive, 84% negative), which these scripts are designed to handle.

## Files Provided

### 1. `quick_comparison.py` (Recommended to Start)
A lightweight script focusing on the most promising models for imbalanced classification.

**Features:**
- 7 model variants including Random Forest, Extra Trees, Gradient Boosting
- Handles class imbalance with balanced weights
- Fast execution (~1-2 minutes)
- Detailed confusion matrices and feature importance

**Usage:**
```bash
python3 quick_comparison.py \
    --features coverage_features.jsonl \
    --output-dir quick_results \
    --cv-folds 5
```

### 2. `model_comparison.py` (Comprehensive Analysis)
Full-featured comparison with 12+ models including XGBoost and LightGBM.

**Features:**
- Advanced models (XGBoost, LightGBM, SVM, Neural Networks)
- Multiple imbalance handling techniques (SMOTE, ADASYN, undersampling)
- Hyperparameter tuning option
- Visualization generation (ROC curves, feature importance plots)

**Dependencies:**
```bash
pip install scikit-learn xgboost lightgbm imbalanced-learn matplotlib seaborn
```

**Usage:**
```bash
# Basic comparison
python3 model_comparison.py \
    --features coverage_features.jsonl \
    --output-dir model_results \
    --balance-method smote \
    --cv-folds 5

# With hyperparameter tuning (slower but better)
python3 model_comparison.py \
    --features coverage_features.jsonl \
    --output-dir tuned_results \
    --balance-method smote \
    --tune-hyperparameters
```

### 3. `deep_learning_models.py` (Neural Network Approaches)
Advanced deep learning architectures for coverage prediction.

**Features:**
- 6 neural network architectures (ResNet, Attention, Gated, TabNet-like)
- Early stopping and learning rate scheduling
- Ensemble methods
- GPU support if available

**Dependencies:**
```bash
pip install torch scikit-learn pandas numpy
```

**Usage:**
```bash
python3 deep_learning_models.py \
    --features coverage_features.jsonl \
    --output-dir dl_results \
    --epochs 100 \
    --batch-size 32
```

## Key Results to Expect

Based on your dataset characteristics:

### Expected Performance Ranges:
- **F1 Score**: 0.30-0.45 (due to class imbalance)
- **Precision**: 0.25-0.40
- **Recall**: 0.40-0.70
- **ROC-AUC**: 0.65-0.80

### Likely Best Models:
1. **Random Forest with balanced weights** - Good overall performance
2. **XGBoost with scale_pos_weight** - Best for handling imbalance
3. **LightGBM** - Fast training, good performance
4. **Gradient Boosting** - Stable performance

## Important Features to Watch

Based on your feature set, likely important features will be:
- Instruction mix ratios (branch_fraction, load_fraction, etc.)
- Code complexity metrics (unique_opcode_count, instruction_count)
- Control flow patterns (branch_count, jump_count)
- CSR usage patterns (csr_count, csr_unique_targets)

## Recommendations for Your Use Case

### 1. Start with Quick Comparison
```bash
python3 quick_comparison.py --features coverage_features.jsonl --cv-folds 5
```
This will give you baseline results in ~1 minute.

### 2. Try SMOTE for Better Class Balance
```bash
python3 model_comparison.py --features coverage_features.jsonl --balance-method smote
```
SMOTE creates synthetic examples of the minority class.

### 3. Tune the Best Model
Once you identify the best model (likely XGBoost or Random Forest):
```bash
python3 model_comparison.py --features coverage_features.jsonl --tune-hyperparameters
```

### 4. Adjust Decision Threshold
Since you have imbalanced classes, consider adjusting the probability threshold:
- Lower threshold (e.g., 0.3): Higher recall, catch more coverage-adding tests
- Higher threshold (e.g., 0.7): Higher precision, fewer false positives

## Interpreting Results

### For Your Specific Goal (Reducing Simulation Time):
- **High Precision Model**: Use if simulation resources are very limited
  - Runs fewer tests but misses some coverage
  - Good when false positives are expensive

- **High Recall Model**: Use if you can't afford to miss coverage
  - Runs more tests to ensure coverage
  - Good when false negatives are expensive

- **Balanced F1 Model**: Best compromise
  - Balances both concerns
  - Recommended for most scenarios

## Expected Compute Savings

With a good model (F1 ~0.40), you can expect:
- Run only 30-40% of tests while maintaining 70-80% of coverage gains
- Reduce simulation time by 60-70%
- Focus compute on high-value tests

## Next Steps

1. Run `quick_comparison.py` first
2. Analyze which features are most important
3. Consider feature engineering based on insights
4. Try ensemble methods combining top 3 models
5. Implement in your CI/CD pipeline with periodic retraining

## Coverage-Driven Pruning Workflow

1. **Refresh features:**
  ```bash
  python3 extract_test_features.py \
     --labels coverage_labels.jsonl \
     --output coverage_features.jsonl
  ```
2. **Rank and prune tests (Decision Tree bundle):**
  ```bash
  lowrisc_env/bin/python claude_model_analysis/prune_tests.py \
     --features claude_model_analysis/coverage_features.jsonl \
     --model claude_model_analysis/tuned_results/best_model.joblib \
     --output claude_model_analysis/tuned_results/ranked_tests.csv \
     --selected-output claude_model_analysis/tuned_results/pruned_tests.txt \
     --top-frac 0.4 --explore-frac 0.1
  ```
  `ranked_tests.csv` holds every test with its predicted probability, while `pruned_tests.txt`
  lists the high-priority subset (plus a small exploration slice) for regressions.
3. **Feed the trimmed list to regressions:** most runners accept a plain text test list,
  so point `regression.sh` (or your CI job) at `pruned_tests.txt` before launching sims.
4. **Track results:** keep coverage deltas from the pruned run and append new labels back into
  `coverage_labels.jsonl` so the model can be retrained with fresher data.

### Running only the ranked/pruned tests

1. Inspect/edit the generated list if needed:
  ```bash
  head -n 20 claude_model_analysis/tuned_results/pruned_tests.txt
  ```
2. Launch the helper script (set `SIMULATOR`, `COV`, `WAVES`, etc. as needed). Each row in
  `pruned_tests.txt` maps to one `make` invocation with `ITERATIONS=1` and the exact seed.
  ```bash
  chmod +x claude_model_analysis/run_ranked_regression.sh
  ./claude_model_analysis/run_ranked_regression.sh --limit 50 --out-dir out_ranked
  ```
3. Pass `--dry-run` first if you only want to review commands, or reduce/extend the subset via
  `--limit` or by editing the list (e.g. keep only tests above a probability threshold).

## Troubleshooting

### If models perform poorly (F1 < 0.25):
- Check if there are temporal patterns (early tests vs late tests)
- Consider sequence-based features
- Look for data quality issues

### If recall is too low:
- Adjust class weights more aggressively
- Use SMOTE or ADASYN
- Lower decision threshold

### If precision is too low:
- Use more conservative class weights
- Increase decision threshold
- Focus on high-confidence predictions only

## Contact & Support

These scripts are designed for your RISC-V verification workflow. Adjust parameters based on your specific requirements and computational constraints.
