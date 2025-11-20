#!/usr/bin/env python3
"""Rank tests with the tuned model and emit a pruned regression list."""

import argparse
import math
from pathlib import Path
from typing import Iterable, List

import joblib
import numpy as np
import pandas as pd

NUMERIC_FILL = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank RISC-V tests via the tuned model")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("coverage_features.jsonl"),
        help="Path to JSONL features produced by extract_test_features.py",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("claude_model_analysis/tuned_results/best_model.joblib"),
        help="Path to joblib bundle saved by model_comparison.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ranked_tests.csv"),
        help="CSV file that will contain ranking information",
    )
    parser.add_argument(
        "--selected-output",
        type=Path,
        default=None,
        help="Optional text file listing the tests that should run",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Probability threshold for selecting a test (0-1)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Always keep the top K highest-scoring tests",
    )
    parser.add_argument(
        "--top-frac",
        type=float,
        default=0.0,
        help="Always keep the top fraction (0-1) of tests",
    )
    parser.add_argument(
        "--explore-frac",
        type=float,
        default=0.1,
        help="Random fraction of the remaining tests to keep for exploration",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed used when sampling the exploration bucket",
    )
    return parser.parse_args()


def load_bundle(model_path: Path) -> tuple:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_keys: List[str] = bundle["feature_keys"]
    scaler = bundle.get("scaler")
    return model, feature_keys, scaler


def load_features(feature_path: Path, feature_keys: Iterable[str]) -> pd.DataFrame:
    df = pd.read_json(feature_path, lines=True)
    for key in feature_keys:
        if key not in df.columns:
            df[key] = NUMERIC_FILL
    numeric = df[list(feature_keys)].apply(pd.to_numeric, errors="coerce").fillna(NUMERIC_FILL)
    df_features = numeric.astype(float)
    df_meta = df.drop(columns=[col for col in df.columns if col in feature_keys], errors="ignore")
    return pd.concat([df_meta, df_features], axis=1)


def pick_identifiers(df: pd.DataFrame) -> pd.Series:
    for candidate in ("testdotseed", "test", "name", "coverage_path"):
        if candidate in df.columns:
            series = df[candidate].fillna("").astype(str)
            series = series.replace("nan", "")
            if (series == "").any():
                series = series.where(series != "", series.index.map(lambda idx: f"sample_{idx}"))
            return series
    return pd.Series(
        (f"sample_{idx}" for idx in df.index), index=df.index, dtype="string"
    )


def score_tests(df: pd.DataFrame, feature_keys: List[str], model, scaler) -> pd.DataFrame:
    feature_matrix = df[feature_keys].to_numpy(dtype=np.float64)
    if scaler is not None:
        feature_matrix = scaler.transform(feature_matrix)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(feature_matrix)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(feature_matrix)
        span = raw.max() - raw.min()
        scores = (raw - raw.min()) / span if span else np.zeros_like(raw)
    else:
        preds = model.predict(feature_matrix)
        scores = preds.astype(float)
    ids = pick_identifiers(df)
    ranked = pd.DataFrame(
        {
            "test_id": ids,
            "probability": scores,
            "label": df["label"] if "label" in df.columns else np.nan,
            "coverage_path": df["coverage_path"] if "coverage_path" in df.columns else "",
        }
    )
    ranked = ranked.sort_values("probability", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", ranked.index + 1)
    return ranked


def select_tests(ranked: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    keep = np.zeros(len(ranked), dtype=bool)
    if args.top_k > 0:
        keep[: min(args.top_k, len(ranked))] = True
    if args.top_frac > 0:
        count = math.ceil(args.top_frac * len(ranked))
        keep[: min(max(count, 0), len(ranked))] = True
    if args.threshold is not None:
        keep |= ranked["probability"].to_numpy() >= args.threshold
    if args.selected_output is not None and args.explore_frac > 0:
        remaining = np.where(~keep)[0]
        if remaining.size:
            sample_size = min(math.ceil(args.explore_frac * len(ranked)), remaining.size)
            rng = np.random.default_rng(args.random_seed)
            chosen = rng.choice(remaining, size=sample_size, replace=False)
            keep[chosen] = True
    return keep


def write_outputs(ranked: pd.DataFrame, keep_mask: np.ndarray, args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.output, index=False)
    if args.selected_output:
        args.selected_output.parent.mkdir(parents=True, exist_ok=True)
        selected = ranked.loc[keep_mask]
        with args.selected_output.open("w", encoding="utf-8") as handle:
            for _, row in selected.iterrows():
                handle.write(f"{row['test_id']},{row['probability']:.6f}\n")
        print(f"Selected {len(selected)} / {len(ranked)} tests -> {args.selected_output}")
    print(f"Ranking written to {args.output}")


def main() -> None:
    args = parse_args()
    model, feature_keys, scaler = load_bundle(args.model)
    df = load_features(args.features, feature_keys)
    ranked = score_tests(df, feature_keys, model, scaler)
    keep_mask = select_tests(ranked, args)
    write_outputs(ranked, keep_mask, args)


if __name__ == "__main__":
    main()
