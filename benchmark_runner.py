"""
benchmark_runner.py

Task 3.2 - Reproducible multi-model cross-validation benchmark.

Evaluates three classification algorithms across two or more
scikit-learn datasets, reports fold-level accuracy, mean accuracy,
standard deviation, variance, and writes a Markdown comparison table.

Default benchmark:
    Datasets:
        - wine
        - breast_cancer

    Algorithms:
        - logistic_regression
        - decision_tree
        - svc

    Cross-validation:
        - StratifiedKFold
        - 5 folds
        - shuffle=True
        - random_state=42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml_tools import DATASETS


# ---------------------------------------------------------
# Supported Algorithms
# ---------------------------------------------------------

SUPPORTED_ALGORITHMS = {
    "logistic_regression",
    "decision_tree",
    "svc",
}


# ---------------------------------------------------------
# Normalization Helpers
# ---------------------------------------------------------

def normalize_algorithm_name(
    algorithm: str,
) -> str:
    """
    Normalizes common aliases to the canonical benchmark names.
    """

    normalized = (
        str(algorithm)
        .lower()
        .strip()
        .replace("-", "_")
        .replace(" ", "_")
    )

    aliases = {
        "logistic": "logistic_regression",
        "logreg": "logistic_regression",
        "lr": "logistic_regression",
        "decisiontree": "decision_tree",
        "tree": "decision_tree",
        "dt": "decision_tree",
        "svm": "svc",
        "kernel_svm": "svc",
        "support_vector_classifier": "svc",
    }

    return aliases.get(
        normalized,
        normalized,
    )


# ---------------------------------------------------------
# Estimator Factory
# ---------------------------------------------------------

def build_estimator(
    algorithm: str,
) -> BaseEstimator:
    """
    Builds one reproducible estimator.

    Scaling is placed INSIDE sklearn Pipelines for Logistic
    Regression and SVC so that StandardScaler is fitted only
    on each training fold during cross-validation.
    """

    model_name = normalize_algorithm_name(
        algorithm
    )

    if model_name == "logistic_regression":

        return Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        random_state=42,
                    ),
                ),
            ]
        )

    if model_name == "decision_tree":

        return DecisionTreeClassifier(
            random_state=42,
        )

    if model_name == "svc":

        return Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                    ),
                ),
            ]
        )

    raise ValueError(
        f"Unsupported algorithm '{algorithm}'. "
        f"Supported algorithms: "
        f"{sorted(SUPPORTED_ALGORITHMS)}"
    )


# ---------------------------------------------------------
# Dataset Validation
# ---------------------------------------------------------

def validate_dataset_name(
    dataset_name: str,
) -> str:
    """
    Validates and normalizes a dataset name.
    """

    normalized = (
        str(dataset_name)
        .lower()
        .strip()
    )

    if normalized not in DATASETS:

        raise ValueError(
            f"Dataset '{normalized}' not found. "
            f"Available datasets: "
            f"{sorted(DATASETS.keys())}"
        )

    return normalized


# ---------------------------------------------------------
# Single CV Experiment
# ---------------------------------------------------------

def evaluate_model_cv(
    dataset_name: str,
    algorithm: str,
    cv: int = 5,
) -> dict[str, Any]:
    """
    Evaluates one algorithm on one dataset using stratified
    cross-validation.

    Returns fold-level scores plus:
        - CV mean accuracy
        - CV standard deviation
        - CV sample variance
    """

    dataset_name = validate_dataset_name(
        dataset_name
    )

    algorithm = normalize_algorithm_name(
        algorithm
    )

    if algorithm not in SUPPORTED_ALGORITHMS:

        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. "
            f"Supported algorithms: "
            f"{sorted(SUPPORTED_ALGORITHMS)}"
        )

    if not isinstance(
        cv,
        int,
    ):

        raise TypeError(
            "cv must be an integer."
        )

    if cv < 2:

        raise ValueError(
            "cv must be at least 2."
        )

    dataset = DATASETS[
        dataset_name
    ]()

    X = np.asarray(
        dataset.data
    )

    y = np.asarray(
        dataset.target
    )

    unique_classes, class_counts = (
        np.unique(
            y,
            return_counts=True,
        )
    )

    minimum_class_count = int(
        class_counts.min()
    )

    if cv > minimum_class_count:

        raise ValueError(
            f"cv={cv} is too large for dataset "
            f"'{dataset_name}'. The smallest class "
            f"contains only {minimum_class_count} samples."
        )

    estimator = build_estimator(
        algorithm
    )

    splitter = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=42,
    )

    scores = cross_val_score(
        estimator,
        X,
        y,
        cv=splitter,
        scoring="accuracy",
        n_jobs=-1,
        error_score="raise",
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    mean_accuracy = float(
        np.mean(
            scores
        )
    )

    std_accuracy = float(
        np.std(
            scores,
            ddof=1,
        )
    )

    variance_accuracy = float(
        np.var(
            scores,
            ddof=1,
        )
    )

    return {
        "dataset": dataset_name,
        "algorithm": algorithm,
        "n_samples": int(
            X.shape[0]
        ),
        "n_features": int(
            X.shape[1]
        ),
        "n_classes": int(
            len(
                unique_classes
            )
        ),
        "cv_folds": cv,
        "scoring": "accuracy",
        "fold_scores": [
            round(
                float(score),
                4,
            )
            for score in scores
        ],
        "cv_mean_accuracy": round(
            mean_accuracy,
            4,
        ),
        "cv_std_accuracy": round(
            std_accuracy,
            4,
        ),
        "cv_variance_accuracy": round(
            variance_accuracy,
            6,
        ),
    }


# ---------------------------------------------------------
# Full Benchmark
# ---------------------------------------------------------

def run_benchmark(
    datasets: list[str] | tuple[str, ...] | None = None,
    algorithms: list[str] | tuple[str, ...] | None = None,
    cv: int = 5,
) -> dict[str, Any]:
    """
    Runs every requested algorithm on every requested dataset.

    Default:
        2 datasets x 3 algorithms = 6 experiments.
    """

    if datasets is None:

        datasets = (
            "wine",
            "breast_cancer",
        )

    if algorithms is None:

        algorithms = (
            "logistic_regression",
            "decision_tree",
            "svc",
        )

    normalized_datasets = [
        validate_dataset_name(
            dataset_name
        )
        for dataset_name in datasets
    ]

    normalized_algorithms = [
        normalize_algorithm_name(
            algorithm
        )
        for algorithm in algorithms
    ]

    unsupported = [
        algorithm
        for algorithm in normalized_algorithms
        if algorithm
        not in SUPPORTED_ALGORITHMS
    ]

    if unsupported:

        raise ValueError(
            "Unsupported algorithms: "
            f"{unsupported}. Supported algorithms: "
            f"{sorted(SUPPORTED_ALGORITHMS)}"
        )

    experiments = []

    for dataset_name in normalized_datasets:

        for algorithm in normalized_algorithms:

            result = evaluate_model_cv(
                dataset_name=dataset_name,
                algorithm=algorithm,
                cv=cv,
            )

            experiments.append(
                result
            )

    return {
        "benchmark": (
            "three_algorithm_two_dataset_cv"
        ),
        "datasets": normalized_datasets,
        "algorithms": normalized_algorithms,
        "cv_strategy": (
            "StratifiedKFold"
        ),
        "cv_folds": cv,
        "shuffle": True,
        "random_state": 42,
        "scoring": "accuracy",
        "experiment_count": len(
            experiments
        ),
        "experiments": experiments,
    }


# ---------------------------------------------------------
# Markdown Table
# ---------------------------------------------------------

def benchmark_to_markdown(
    benchmark_result: dict[str, Any],
) -> str:
    """
    Converts benchmark results into a Markdown comparison table.
    """

    experiments = benchmark_result.get(
        "experiments",
        [],
    )

    lines = [
        "# Model Comparison Benchmark",
        "",
        (
            f"Cross-validation: "
            f"{benchmark_result.get('cv_strategy', 'unknown')} "
            f"with {benchmark_result.get('cv_folds', '?')} folds"
        ),
        "",
        (
            f"Scoring metric: "
            f"{benchmark_result.get('scoring', 'accuracy')}"
        ),
        "",
        (
            "| Dataset | Algorithm | CV Mean Accuracy | "
            "CV Std | CV Variance |"
        ),
        "|---|---|---:|---:|---:|",
    ]

    for experiment in experiments:

        lines.append(
            (
                f"| {experiment['dataset']} "
                f"| {experiment['algorithm']} "
                f"| {experiment['cv_mean_accuracy']:.4f} "
                f"| {experiment['cv_std_accuracy']:.4f} "
                f"| {experiment['cv_variance_accuracy']:.6f} |"
            )
        )

    return "\n".join(
        lines
    )


# ---------------------------------------------------------
# Save Markdown Report
# ---------------------------------------------------------

def save_benchmark_markdown(
    benchmark_result: dict[str, Any],
    output_path: str | Path = (
        "report/benchmark_results.md"
    ),
) -> Path:
    """
    Writes the Markdown benchmark table to disk.
    """

    path = Path(
        output_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    markdown = benchmark_to_markdown(
        benchmark_result
    )

    path.write_text(
        markdown,
        encoding="utf-8",
    )

    return path


# ---------------------------------------------------------
# JSON Wrapper for Later Agent Integration
# ---------------------------------------------------------

def benchmark_models_tool(
    datasets: list[str] | None = None,
    algorithms: list[str] | None = None,
    cv: int = 5,
    save_markdown: bool = True,
    output_path: str = "report/benchmark_results.md",
) -> str:
    """
    Agent-facing benchmark tool.

    Runs the requested cross-validation benchmark and returns
    structured JSON. By default it also writes the Markdown
    comparison table to report/benchmark_results.md.

    The tool performs the experiments only. Model comparison and
    recommendation remain the responsibility of the autonomous
    agent/controller finalization stage.
    """

    try:

        result = run_benchmark(
            datasets=datasets,
            algorithms=algorithms,
            cv=cv,
        )

        markdown = benchmark_to_markdown(
            result
        )

        saved_path = None

        if save_markdown:

            saved_path = save_benchmark_markdown(
                result,
                output_path,
            )

        agent_result = dict(
            result
        )

        agent_result[
            "markdown_table"
        ] = markdown

        agent_result[
            "markdown_output_path"
        ] = (
            str(
                saved_path
            )
            if saved_path is not None
            else None
        )

        return json.dumps(
            agent_result
        )

    except Exception as exc:

        return json.dumps(
            {
                "error": (
                    f"{type(exc).__name__}: "
                    f"{str(exc)}"
                )
            }
        )


# ---------------------------------------------------------
# Command-Line Interface
# ---------------------------------------------------------

def main():
    """
    Runs the default 3-algorithm x 2-dataset benchmark.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run a reproducible multi-model "
            "cross-validation benchmark."
        )
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "wine",
            "breast_cancer",
        ],
        help=(
            "Dataset names. Default: "
            "wine breast_cancer"
        ),
    )

    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=[
            "logistic_regression",
            "decision_tree",
            "svc",
        ],
        help=(
            "Algorithms. Default: "
            "logistic_regression decision_tree svc"
        ),
    )

    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help=(
            "Number of StratifiedKFold folds. "
            "Default: 5"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "report/benchmark_results.md"
        ),
        help=(
            "Markdown output path. Default: "
            "report/benchmark_results.md"
        ),
    )

    args = parser.parse_args()

    result = run_benchmark(
        datasets=args.datasets,
        algorithms=args.algorithms,
        cv=args.cv,
    )

    markdown = benchmark_to_markdown(
        result
    )

    output_path = save_benchmark_markdown(
        result,
        args.output,
    )

    print(
        "=" * 70
    )

    print(
        "TASK 3.2 MODEL COMPARISON BENCHMARK"
    )

    print(
        "=" * 70
    )

    print()

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    print()

    print(
        markdown
    )

    print()

    print(
        f"Markdown report saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
