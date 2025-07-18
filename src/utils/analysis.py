"""
Analysis utilities for inverse scaling evaluation.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def load_summary(summary_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a summary file.

    Args:
        summary_path: Path to the summary file

    Returns:
        The summary dict
    """
    with open(summary_path, "r") as f:
        return json.load(f)


def check_inverse_scaling(df: pd.DataFrame, model: str, task: str) -> Dict[str, Any]:
    """
    Check if a model exhibits inverse scaling on a task.

    Args:
        df: DataFrame with results
        model: Model ID
        task: Task ID

    Returns:
        Dict with inverse scaling metrics
    """
    # Filter to the specified model and task
    model_task_df = df[(df["model"] == model) & (df["task_id"] == task)].copy()

    if len(model_task_df) == 0:
        return {
            "model": model,
            "task": task,
            "inverse_scaling": False,
            "error": "No data available",
        }

    # Group by reasoning budget and compute accuracy
    budget_accuracy = (
        model_task_df.groupby("reasoning_budget")["correct"].mean().reset_index()
    )

    if len(budget_accuracy) < 2:
        return {
            "model": model,
            "task": task,
            "inverse_scaling": False,
            "error": "Insufficient data points",
        }

    # Compute Spearman correlation
    correlation, p_value = stats.spearmanr(
        budget_accuracy["reasoning_budget"], budget_accuracy["correct"]
    )

    # Determine if there's inverse scaling
    is_inverse_scaling = bool(correlation < 0 and p_value < 0.05)
    is_significant = bool(p_value < 0.05)

    return {
        "model": model,
        "task": task,
        "inverse_scaling": is_inverse_scaling,
        "correlation": float(correlation) if pd.notna(correlation) else None,
        "p_value": float(p_value) if pd.notna(p_value) else None,
        "significant": is_significant,
        "budget_accuracy": budget_accuracy.to_dict(orient="records"),
    }


def analyze_all_model_task_pairs(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze all model-task pairs in the DataFrame for inverse scaling.

    Args:
        df: DataFrame with results

    Returns:
        Dict with inverse scaling metrics for all model-task pairs
    """
    models = df["model"].unique()
    tasks = df["task_id"].unique()

    results = {}
    for model in models:
        model_results = {}
        for task in tasks:
            model_results[task] = check_inverse_scaling(df, model, task)
        results[model] = model_results

    # Count inverse scaling cases
    inverse_scaling_count = 0
    total_pairs = 0
    for model in results:
        for task in results[model]:
            if "error" not in results[model][task]:
                total_pairs += 1
                if results[model][task]["inverse_scaling"]:
                    inverse_scaling_count += 1

    return {
        "model_task_results": results,
        "inverse_scaling_count": inverse_scaling_count,
        "total_pairs": total_pairs,
        "inverse_scaling_percentage": (
            (inverse_scaling_count / total_pairs * 100) if total_pairs > 0 else 0
        ),
    }


def find_strongest_inverse_scaling(
    df: pd.DataFrame, min_budgets: int = 3
) -> List[Dict[str, Any]]:
    """
    Find model-task pairs with the strongest inverse scaling.

    Args:
        df: DataFrame with results
        min_budgets: Minimum number of reasoning budgets required

    Returns:
        List of model-task pairs sorted by inverse scaling strength
    """
    models = df["model"].unique()
    tasks = df["task_id"].unique()

    pairs = []
    for model in models:
        for task in tasks:
            model_task_df = df[(df["model"] == model) & (df["task_id"] == task)]
            budget_counts = model_task_df["reasoning_budget"].nunique()

            if budget_counts >= min_budgets:
                budget_accuracy = (
                    model_task_df.groupby("reasoning_budget")["correct"]
                    .mean()
                    .reset_index()
                )
                correlation, p_value = stats.spearmanr(
                    budget_accuracy["reasoning_budget"], budget_accuracy["correct"]
                )

                pairs.append(
                    {
                        "model": model,
                        "task": task,
                        "correlation": correlation,
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                        "inverse_scaling": correlation < 0 and p_value < 0.05,
                        "budget_counts": budget_counts,
                    }
                )

    # Sort by correlation (negative correlations first)
    pairs.sort(key=lambda x: x["correlation"])

    return pairs


def compute_effect_size(
    df: pd.DataFrame, model: str, task: str
) -> Optional[Dict[str, Any]]:
    """
    Compute the effect size of reasoning budget on model performance.

    Args:
        df: DataFrame with results
        model: Model ID
        task: Task ID

    Returns:
        Dict with effect size metrics, or None if not enough data
    """
    # Filter to the specified model and task
    model_task_df = df[(df["model"] == model) & (df["task_id"] == task)].copy()

    if len(model_task_df) == 0:
        return None

    # Get min and max reasoning budgets
    budgets = sorted(model_task_df["reasoning_budget"].unique())

    if len(budgets) < 2:
        return None

    min_budget = min(budgets)
    max_budget = max(budgets)

    # Get accuracies for min and max budgets
    min_budget_df = model_task_df[model_task_df["reasoning_budget"] == min_budget]
    max_budget_df = model_task_df[model_task_df["reasoning_budget"] == max_budget]

    min_accuracy = min_budget_df["correct"].mean()
    max_accuracy = max_budget_df["correct"].mean()

    # Compute effect size
    abs_effect = max_accuracy - min_accuracy
    rel_effect = abs_effect / min_accuracy if min_accuracy > 0 else 0

    # Explicitly convert numpy bool_ to Python bool for JSON serialization
    is_inverse_scaling = bool(abs_effect < 0)

    return {
        "model": model,
        "task": task,
        "min_budget": int(min_budget),  # Also ensure budgets are standard ints
        "max_budget": int(max_budget),  # Also ensure budgets are standard ints
        "min_accuracy": min_accuracy,
        "max_accuracy": max_accuracy,
        "absolute_effect": abs_effect,
        "relative_effect": rel_effect,
        "inverse_scaling": is_inverse_scaling,  # Use the converted Python bool
    }


def analyze_inverse_scaling(
    df: pd.DataFrame,
    models: List[str],
    tasks: List[str],
    reasoning_budgets: List[int],
    output_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Comprehensive analysis of inverse scaling across models and tasks.

    Args:
        df: DataFrame with results
        models: List of model IDs to analyze
        tasks: List of task IDs to analyze
        reasoning_budgets: List of reasoning budgets used
        output_dir: Optional output directory for saving results

    Returns:
        List of effect size data for all model-task pairs
    """
    # First, analyze all model-task pairs
    analysis_results = analyze_all_model_task_pairs(df)

    # Compute effect sizes for all model-task pairs
    effect_sizes = []
    for model in models:
        for task in tasks:
            effect = compute_effect_size(df, model, task)
            if effect:
                effect_sizes.append(effect)

    # Sort by absolute effect (strongest effects first)
    effect_sizes.sort(key=lambda x: abs(x["absolute_effect"]), reverse=True)

    # Save results if output directory is provided
    if output_dir is not None:
        # Save overall analysis
        with open(output_dir / "inverse_scaling_analysis.json", "w") as f:
            json.dump(analysis_results, f, indent=2)

        # Save effect sizes
        with open(output_dir / "effect_sizes.json", "w") as f:
            json.dump(effect_sizes, f, indent=2)

    return effect_sizes
