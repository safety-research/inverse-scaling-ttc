"""
Plotting utilities for inverse scaling evaluation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import ast # ADDED IMPORT

import matplotlib
# Set backend before importing pyplot to avoid GUI issues
matplotlib.use('Agg')
# Configure matplotlib to avoid layout engine conflicts
try:
    # Disable the new layout engine to avoid colorbar compatibility issues
    matplotlib.rcParams['figure.constrained_layout.use'] = False
except (KeyError, ValueError):
    # Fallback for older matplotlib versions
    pass

# Set other compatibility options
matplotlib.rcParams['figure.max_open_warning'] = 0  # Disable warnings about too many figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from matplotlib.cm import ScalarMappable
import math
from safetytooling.apis.inference.openai.utils import count_tokens
import textwrap

# Import TaskLoader specifically for the helper function
from src.task_loader import TaskLoader

logger = logging.getLogger(__name__)


NON_REASONING_MODELS = ["claude-3-5-sonnet-20240620", "claude-3-5-sonnet-20241022", "gpt-4.1-mini-2025-04-14"]
OPENAI_REASONING_MODELS = ["o3-mini-2025-01-31", "o3-2025-04-16", "o4-mini-2025-04-16"]

# New model constant
# CLAUDE_4_MODEL_NAMES = ["claude-sonnet-4-20250514", "claude-opus-4-20250514"]

# Mappings for prettier names
MODEL_TO_PRETTY_NAME = {
    "deepseek-reasoner": "DeepSeek R1",
    "claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",
    "claude-3-7-sonnet-20250219_natural_overthinking": "Claude 3.7 Sonnet (Natural)",
    "claude-3-7-sonnet-20250219_not_use_all_budget": "Claude 3.7 Sonnet (Cautioned)",
    "claude-sonnet-4-20250514": "Claude 4 Sonnet",
    "claude-opus-4-20250514": "Claude 4 Opus",
    "Qwen3-8B": "Qwen3 8B",
    "Qwen3-14B": "Qwen3 14B",
    "Qwen3-32B": "Qwen3 32B",
    "O4-mini": "o4-mini",
    "o4-mini-2025-04-16": "o4-mini",
    "o3-mini-2025-01-31": "o3-mini",
    "o3-2025-04-16": "o3"
}

def get_task_pretty_name(task_id: str, task_loader: Optional[TaskLoader] = None) -> str:
    """Get pretty name for a task from TaskLoader config, with fallback to task_id."""
    if task_loader is None:
        return task_id
    
    try:
        # Try to get the task metadata from TaskLoader
        task_metadata = task_loader.get_task_metadata(task_id)
        if task_metadata and 'name' in task_metadata:
            return task_metadata['name']
    except Exception:
        # If task not found or any error, fallback to task_id
        pass
    
    # Fallback: return the task_id itself
    return task_id


def _safe_count_tokens(sample: Any) -> int:
    """Safely count tokens, handling None, NaN, and potential errors."""

    # Handle OpenAI models that use 'cost' directly
    if sample["model"] in OPENAI_REASONING_MODELS:
        try:
            usage_data = sample.get("usage")
            if usage_data is None:
                raise KeyError("'usage' data is missing")

            # Ensure usage_data itself is a dictionary if it's a string
            if isinstance(usage_data, str):
                original_usage_data_str = usage_data
                try:
                    usage_data = json.loads(usage_data)
                except json.JSONDecodeError as jde:
                    try:
                        usage_data = ast.literal_eval(original_usage_data_str)
                        if not isinstance(usage_data, dict):
                            raise TypeError("ast.literal_eval did not return a dict for usage_data")
                    except Exception as ast_e:
                        raise # Re-raise to fall into the main except block which handles 'cost' fallback
            
            completion_token_details_data = usage_data.get("completion_tokens_details")

            if completion_token_details_data is None:
                raise KeyError("'completion_tokens_details' not found in usage_data") # Will go to outer except

            # Check if completion_token_details_data is a string and parse it
            if isinstance(completion_token_details_data, str):
                original_details_str = completion_token_details_data
                try:
                    completion_token_details_data = json.loads(completion_token_details_data)
                except json.JSONDecodeError as jde:
                    try:
                        completion_token_details_data = ast.literal_eval(original_details_str)
                        if not isinstance(completion_token_details_data, dict):
                            raise TypeError("ast.literal_eval did not return a dict for completion_token_details_data")
                    except Exception as ast_e:
                        raise # Re-raise to fall into the main except block
            
            # Now completion_token_details_data should be a dictionary
            tokens = completion_token_details_data.get("reasoning_tokens")
            if tokens is None:
                raise KeyError("'reasoning_tokens' not found in completion_token_details_data")

            return tokens
        except Exception as e:
            cost_val = sample["cost"]
            return cost_val

    # Handle the new Claude Sonnet 4 model specifically
    # if sample["model"] in CLAUDE_4_MODEL_NAMES:
    #     if "response" in sample and isinstance(sample["response"], str):
    #         response_text = sample["response"]
    #         start_tag = "<thinking>"
    #         end_tag = "</thinking>"
    #         start_index = response_text.find(start_tag)
    #         end_index = response_text.find(end_tag)

    #         if start_index != -1 and end_index != -1 and start_index < end_index:
    #             thinking_content = response_text[start_index + len(start_tag):end_index]
    #             try:
    #                 # FIXME: Use the Claude tokenizer - using o1 as placeholder
    #                 return count_tokens(thinking_content, "o1")
    #             except Exception as e:
    #                 logger.debug(f"Could not count tokens for thinking_content for {sample['model']}: {e}")
    #                 return 0
    #         else:
    #             logger.debug(f"No <thinking> tags found or tags malformed in response for {sample['model']}.")
    #             return 0
    #     else:
    #         logger.debug(f"'response' field missing or not a string for {sample['model']}.")
    #         return 0

    # Existing logic for when 'reasoning_content' is missing or None
    if "reasoning_content" not in sample or pd.isna(sample["reasoning_content"]) or sample["reasoning_content"] is None:
        if sample["model"] in NON_REASONING_MODELS:
            if "response" in sample and sample["response"] is not None and not pd.isna(sample["response"]):
                try:
                    # FIXME: Use the Claude tokenizer - using o1 as placeholder
                    if isinstance(sample["response"], str):
                        num_tokens = count_tokens(sample["response"], "o1")
                        return num_tokens
                    else:
                        logger.debug(f"Response is not a string for {sample['model']} in NON_REASONING_MODELS.")
                        return 0
                except Exception as e:
                    logger.debug(f"Could not count tokens for response for {sample['model']} (NON_REASONING_MODEL): {e}")
                    return 0 # Consistent error return
            else:
                logger.debug(f"Response field missing, None, or NaN for {sample['model']} in NON_REASONING_MODELS.")
                return 0
        else:
            # Models not in OPENAI_REASONING_MODELS, not in CLAUDE_4_MODEL_NAMES,
            # not in NON_REASONING_MODELS, and 'reasoning_content' is missing.
            return 0

    # Existing logic for when 'reasoning_content' is present
    if not isinstance(sample["reasoning_content"], str):
        # Handle non-string types that might not be caught by pd.isna (e.g., numbers)
        logger.debug(f"reasoning_content is not a string for {sample['model']}.")
        return 0
    try:
        # FIXME: Use the Claude tokenizer - using o1 as placeholder
        return count_tokens(sample["reasoning_content"], "o1")
    except Exception as e:
        logger.debug(f"Could not count tokens for reasoning_content for {sample['model']}: {e}")
        return 0


def _add_scaling_data_to_results(
    results_df: pd.DataFrame,
    task_loader: TaskLoader,
    scaling_columns: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Adds specified scaling columns from the original task data to the results DataFrame.

    Args:
        results_df: DataFrame containing evaluation results. Must have 'task_id' and 'instance_id'.
        task_loader: An initialized TaskLoader instance.
        scaling_columns: A dictionary mapping task_id to the name of the scaling column
                         in the original task data file. If None or empty, returns the
                         original DataFrame.

    Returns:
        A DataFrame with the scaling columns merged in, or the original DataFrame if
        scaling_columns is not provided or no columns are found.
    """
    if not scaling_columns or not isinstance(scaling_columns, dict):
        logger.info(
            "No valid scaling_columns provided. Skipping merge with input data."
        )
        return results_df

    if "instance_id" not in results_df.columns:
        logger.error(
            "Results DataFrame must contain 'instance_id' column to merge scaling data."
        )
        raise ValueError("Missing 'instance_id' column in results DataFrame.")
    if "task_id" not in results_df.columns:
        logger.error(
            "Results DataFrame must contain 'task_id' column to merge scaling data."
        )
        raise ValueError("Missing 'task_id' column in results DataFrame.")

    all_merged_dfs = []
    processed_tasks = set()

    # Group by task_id to process each task's scaling column
    for task_id, group_df in results_df.groupby("task_id"):
        if task_id in scaling_columns:
            scaling_col_name = scaling_columns[task_id]
            logger.info(
                f"Processing task '{task_id}': Attempting to merge scaling column '{scaling_col_name}'."
            )

            try:
                # Load the original task data
                input_data = task_loader.load_task(task_id)
                if not input_data:
                    logger.warning(f"No input data loaded for task '{task_id}'. Skipping merge for this task.")
                    all_merged_dfs.append(group_df) # Keep original group if input fails
                    continue

                # Create a DataFrame from the input data
                # We need the instance_id (constructed) and the scaling column
                input_rows = []
                has_scaling_col = False
                for i, instance in enumerate(input_data):
                    instance_id = f"{task_id}_{i}" # Reconstruct instance_id
                    if scaling_col_name in instance:
                        has_scaling_col = True
                        input_rows.append(
                            {
                                "instance_id": instance_id,
                                scaling_col_name: instance[scaling_col_name],
                            }
                        )
                    else:
                         # Log only once per task if column is missing
                         if not has_scaling_col and i == 0:
                             logger.warning(f"Scaling column '{scaling_col_name}' not found in input data for task '{task_id}'.")
                         # Append row with NaN if scaling column is missing for this instance
                         input_rows.append({"instance_id": instance_id, scaling_col_name: np.nan})


                if not has_scaling_col:
                     logger.warning(f"Scaling column '{scaling_col_name}' not found in ANY input instance for task '{task_id}'. Merged column will be NaN.")
                     # Still create the df so the merge adds the column with NaNs
                     input_scaling_df = pd.DataFrame(input_rows)
                else:
                    input_scaling_df = pd.DataFrame(input_rows)
                    logger.info(f"Successfully created input DataFrame for task '{task_id}' with column '{scaling_col_name}'.")


                # Merge the scaling data into the results group
                merged_group = pd.merge(
                    group_df,
                    input_scaling_df,
                    on="instance_id",
                    how="left", # Keep all results rows, add NaN if no match in input (shouldn't happen if IDs are consistent)
                    suffixes=("", "_input"), # Avoid column name conflicts if scaling_col_name already exists somehow
                )

                # Handle potential column name conflicts after merge
                if f"{scaling_col_name}_input" in merged_group.columns and scaling_col_name in group_df.columns:
                     logger.warning(f"Column '{scaling_col_name}' already exists in results_df for task '{task_id}'. Input data added as '{scaling_col_name}_input'.")
                elif f"{scaling_col_name}_input" in merged_group.columns:
                     # Rename the merged column if no conflict
                     merged_group.rename(columns={f"{scaling_col_name}_input": scaling_col_name}, inplace=True)


                all_merged_dfs.append(merged_group)
                processed_tasks.add(task_id)

            except FileNotFoundError:
                logger.error(f"Input data file not found for task '{task_id}'. Skipping merge for this task.")
                all_merged_dfs.append(group_df) # Keep original group
            except Exception as e:
                logger.error(
                    f"Error processing or merging scaling data for task '{task_id}': {e}",
                    exc_info=True,
                )
                all_merged_dfs.append(group_df) # Keep original group on error

        else:
            # Task not in scaling_columns map, keep original group
            all_merged_dfs.append(group_df)

    # Add unprocessed tasks back (shouldn't happen with groupby but safe)
    # unprocessed_results = results_df[~results_df['task_id'].isin(processed_tasks)]
    # all_merged_dfs.append(unprocessed_results) # Not needed with groupby iteration

    if not all_merged_dfs:
        logger.warning("Merging resulted in an empty DataFrame.")
        return pd.DataFrame(columns=results_df.columns) # Return empty df with original columns

    # Combine all processed groups back into a single DataFrame
    final_df = pd.concat(all_merged_dfs, ignore_index=True)

    # Log summary of added columns
    added_cols = set(final_df.columns) - set(results_df.columns)
    if added_cols:
        logger.info(f"Successfully added scaling columns: {list(added_cols)}")
    else:
        logger.info("No new scaling columns were added (or they already existed).")

    return final_df


def set_plotting_style():
    """Set seaborn style for plotting."""
    # Use seaborn directly
    sns.set_theme(style="whitegrid")

    # Set rcParams for text and font
    plt.rcParams["text.usetex"] = False
    plt.rcParams["font.size"] = "12.5"
    plt.rcParams["figure.dpi"] = 190
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams['font.family'] = 'cmr10'
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["axes.formatter.use_mathtext"] = False

    # Set rcParams for the border color and ticks
    plt.rcParams["axes.edgecolor"] = "black"  # Set border color
    plt.rcParams["axes.linewidth"] = 1.5  # Set border width
    plt.rcParams["xtick.color"] = "black"  # Set xtick color
    plt.rcParams["ytick.color"] = "black"  # Set ytick color

    # Set background color
    plt.rcParams["axes.facecolor"] = "#EFEFEAFF"

    # Set grid color and style
    plt.rcParams["grid.color"] = "white"
    plt.rcParams["grid.alpha"] = 0.7
    plt.rcParams["grid.linewidth"] = 1.5
    plt.rcParams["grid.linestyle"] = "--"

    # Make ticks show
    plt.rcParams["xtick.bottom"] = True  # Ensure xticks are shown at the bottom
    plt.rcParams["ytick.left"] = True  # Ensure yticks are shown on the left

    # Set context
    sns.set_context(context="talk", font_scale=0.9)


def plot_effect_sizes(
    effect_sizes: List[Dict[str, Any]],
    output_dir: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot effect sizes for model-task pairs.

    Args:
        effect_sizes: List of effect size dicts
        output_dir: Directory to save plot (default: don't save)

    Returns:
        Figure object
    """
    set_plotting_style()

    # Convert to DataFrame
    effect_df = pd.DataFrame(effect_sizes)

    # Sort by effect size (ascending)
    effect_df = effect_df.sort_values("absolute_effect")

    # Plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(effect_df) * 0.3)))

    # Create labels
    labels = [f"{row['model']} - {row['task']}" for _, row in effect_df.iterrows()]

    # Plot horizontal bars
    bars = ax.barh(
        range(len(effect_df)),
        effect_df["absolute_effect"],
        color=[
            "red" if effect < 0 else "green" for effect in effect_df["absolute_effect"]
        ],
        alpha=0.7,
    )

    # Add labels
    ax.set_yticks(range(len(effect_df)))
    ax.set_yticklabels(labels)

    # Add a vertical line at x=0
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)

    # Add labels for axes
    ax.set_xlabel("Effect Size (Accuracy Difference)")
    ax.set_title("Effect of Reasoning Budget on Accuracy")

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.3, axis="x")

    # Add values at the end of each bar
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(
            width + (0.01 if width >= 0 else -0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            ha="left" if width >= 0 else "right",
            va="center",
        )

    # Tight layout
    fig.tight_layout()

    # Save if output_dir is specified
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)
        fig.savefig(output_dir / "effect_sizes.png", dpi=300, bbox_inches="tight")

    return fig


def plot_token_correlations(
    df: pd.DataFrame,
    models: List[str] = None,
    tasks: List[str] = None,
    output_dir: Optional[Path] = None,
    min_samples: int = 2,  # Minimum number of aggregated budget groups required to calculate correlation
    sort_by: str = "correlation",  # "correlation", "task", "significance"
    include_zero_tokens: bool = True,  # Whether to include zero tokens in correlation calculation
    plot_type: str = "accuracy",  # "accuracy", "cost", "latency", "mse"
    x_axis_type: str = "tokens",  # "tokens" or "budget"
    log_transform_x_for_corr: bool = False, # Whether to log-transform x-axis data for correlation
    combined_plot: bool = False,  # Whether to also create a combined plot with all models
    max_task_label_length: int = 25,  # Maximum length for task labels before truncation
    min_samples_per_budget_group: int = 5, # Min raw samples per budget group for it to be included
) -> Dict[str, plt.Figure]:
    """
    Plot Pearson correlations. Data is first aggregated by 'reasoning_budget'.
    The x-axis for correlation can be the mean 'reasoning_content_tokens' per budget group
    (if x_axis_type='tokens') or the 'reasoning_budget' value itself (if x_axis_type='budget').
    The y-axis is the mean of the specified metric (e.g., accuracy) per budget group.

    Args:
        df: DataFrame with results.
        models: List of model IDs to include (default: all).
        tasks: List of task IDs to include (default: all).
        output_dir: Directory to save plots (default: don't save).
        min_samples: Minimum number of aggregated budget groups required to attempt a correlation
                     for a model-task pair. Default is 10.
        sort_by: How to sort the results ("correlation", "task", "significance").
        include_zero_tokens: Whether to include samples with zero reasoning_content_tokens
                             when forming budget groups. This filter is applied to raw data.
        plot_type: Type of metric to correlate with the x-axis variable
                   ("accuracy", "cost", "latency", "mse", "relative_error").
        x_axis_type: Variable for the x-axis of the correlation:
                     "tokens" (mean reasoning_content_tokens per budget group) or
                     "budget" (reasoning_budget value). Defaults to "tokens".
        log_transform_x_for_corr: If True, the x-axis data (tokens or budget)
                                  will be log-transformed before calculating Pearson correlation.
                                  Requires x-values to be positive. Defaults to False.
        combined_plot: Whether to also create a combined plot with all models.
        max_task_label_length: Maximum length for task labels before truncation.
        min_samples_per_budget_group: Minimum number of raw data instances within a budget group
                                      for that group to be included in aggregation and correlation.
                                      Default is 5.

    Returns:
        Dictionary mapping plot names to Figure objects.
    """
    set_plotting_style()

    if models is None:
        models = df["model"].unique()
    if tasks is None:
        tasks = df["task_id"].unique()

    # Select the appropriate column based on plot_type
    if plot_type == "accuracy":
        y_column = "correct"
        y_label = "Accuracy"
    elif plot_type == "cost":
        y_column = "cost"
        y_label = "Cost"
    elif plot_type == "latency":
        y_column = "latency"
        y_label = "Latency"
    elif plot_type == "mse":
        y_column = "squared_error" # Use the new column
        y_label = "Mean Squared Error"
    elif plot_type == "relative_error":
        y_column = "relative_error"
        y_label = "Relative Error"
    else:
        raise ValueError(f"Invalid plot_type: {plot_type}")

    # Determine x-axis column and label
    x_correlation_column = ""
    x_correlation_label = ""
    if x_axis_type == "tokens":
        x_correlation_column = "reasoning_content_tokens"
        x_correlation_label = "Reasoning Tokens"
    elif x_axis_type == "budget":
        x_correlation_column = "reasoning_budget"
        x_correlation_label = "Reasoning Budget (Tokens)"
        df[x_correlation_column] = pd.to_numeric(df[x_correlation_column], errors='coerce')
    else:
        raise ValueError(f"Invalid x_axis_type: {x_axis_type}. Must be 'tokens' or 'budget'.")

    # Ensure reasoning_content_tokens is present if needed for filtering or as the x-column
    if "reasoning_content_tokens" not in df.columns:
        df["reasoning_content_tokens"] = df.apply(_safe_count_tokens, axis=1)

    # Ensure reasoning_budget is numeric for grouping and potential use as x-column
    df["reasoning_budget"] = pd.to_numeric(df["reasoning_budget"], errors='coerce')

    # Filter out rows with missing essential values BEFORE aggregation
    # x_correlation_column is 'reasoning_content_tokens' or 'reasoning_budget'
    # y_column is the metric like 'correct'
    # Also drop rows if reasoning_budget became NaN after coercion, as it's needed for grouping.
    df = df.dropna(subset=[x_correlation_column, y_column, "reasoning_budget"])

    all_correlations = []

    for model in models:
        for task in tasks:
            model_task_df_initial = df[(df["model"] == model) & (df["task_id"] == task)].copy()

            # Apply include_zero_tokens filter to the raw data for this model-task pair
            if not include_zero_tokens:
                model_task_df_initial = model_task_df_initial[model_task_df_initial["reasoning_content_tokens"] > 0]

            if model_task_df_initial.empty:
                logger.debug(f"No data for {model}/{task} after initial filtering and zero token check.")
                continue

            # Group by 'reasoning_budget'
            grouped_by_budget = model_task_df_initial.groupby("reasoning_budget")

            aggregated_x_values = []
            aggregated_y_values = []

            for budget_value, budget_group_df in grouped_by_budget:
                if len(budget_group_df) < min_samples_per_budget_group:
                    logger.debug(f"Skipping budget group {budget_value} for {model}/{task} due to insufficient samples: {len(budget_group_df)} < {min_samples_per_budget_group}")
                    continue

                # Calculate Y value (mean of the metric for this budget group)
                current_y_metric_data = budget_group_df[y_column].astype(float).dropna()
                if len(current_y_metric_data) < min_samples_per_budget_group: # Re-check after dropna for the y-metric itself
                    logger.debug(f"Skipping budget group {budget_value} for {model}/{task} due to insufficient y-metric samples after dropna: {len(current_y_metric_data)} < {min_samples_per_budget_group}")
                    continue

                mean_y_for_group = current_y_metric_data.mean()
                if pd.isna(mean_y_for_group):
                    logger.debug(f"Skipping budget group {budget_value} for {model}/{task} due to NaN mean y-metric.")
                    continue

                # Calculate X value based on x_axis_type
                current_x_val_for_group = np.nan
                if x_axis_type == "tokens":
                    # x_correlation_column is 'reasoning_content_tokens'
                    tokens_for_group = budget_group_df[x_correlation_column].astype(float).dropna()
                    if len(tokens_for_group) < min_samples_per_budget_group: # Check again for x-column if it's tokens
                        logger.debug(f"Skipping budget group {budget_value} for {model}/{task} due to insufficient token samples after dropna: {len(tokens_for_group)} < {min_samples_per_budget_group}")
                        continue
                    if not tokens_for_group.empty:
                        current_x_val_for_group = tokens_for_group.mean()
                    else:
                        logger.debug(f"Skipping budget group {budget_value} for {model}/{task} due to empty token data after filtering.")
                        continue # No valid token data for this group to calculate mean
                elif x_axis_type == "budget":
                    # x_correlation_column is 'reasoning_budget', budget_value is already the group key
                    current_x_val_for_group = float(budget_value)

                if pd.isna(current_x_val_for_group):
                    logger.debug(f"Skipping budget group {budget_value} for {model}/{task} due to NaN aggregated x-value.")
                    continue

                aggregated_x_values.append(current_x_val_for_group)
                aggregated_y_values.append(mean_y_for_group)

            n_for_correlation = len(aggregated_x_values)

            if n_for_correlation < min_samples: # `min_samples` now refers to number of aggregated groups
                logger.warning(f"Not enough aggregated budget groups ({n_for_correlation}) for {model}/{task} to calculate correlation (min_samples: {min_samples}). Skipping.")
                continue

            # Convert aggregated lists to numpy arrays for pearsonr
            x_array = np.array(aggregated_x_values, dtype=float)
            y_array = np.array(aggregated_y_values, dtype=float)

            # Potentially log-transform x_array for correlation calculation
            effective_x_correlation_label = x_correlation_label # Use a temporary label for this iteration
            if log_transform_x_for_corr:
                if np.any(x_array < 0):
                    logger.warning(
                        f"Cannot apply log-like transformation to x-axis data for {model}/{task} because it contains negative values. "
                        "Correlation will be calculated on original x-values."
                    )
                else:  # All values are >= 0, safe for np.log1p
                    x_array = np.log1p(x_array)  # Computes log(1+x)
                    effective_x_correlation_label = f"Log(1+{x_correlation_label})"
                    logger.info(f"Applied log1p (log(1+x)) transformation to x-axis data for correlation for {model}/{task}.")

            # Check for variance AFTER aggregation and potential transformation; NaNs in aggregated values were already skipped.
            if np.std(x_array) < 1e-9 or np.std(y_array) < 1e-9: # Check for near-zero variance
                logger.warning(f"Skipping correlation for {model}/{task} due to zero or near-zero variance in aggregated x or y values (x_std={np.std(x_array):.2e}, y_std={np.std(y_array):.2e}).")
                correlation, p_value = np.nan, np.nan
            else:
                correlation, p_value = stats.pearsonr(x_array, y_array)

            # CI calculation proceeds using n_for_correlation (number of aggregated points)
            if np.isnan(correlation):
                logger.warning(f"Pearson correlation resulted in NaN for {model}/{task}. Skipping CI calculation.")
                r_low, r_high = np.nan, np.nan
            elif n_for_correlation <= 3: # Fisher's Z requires n > 3
                logger.warning(f"Not enough samples ({n_for_correlation}) for Fisher's Z transformation for {model}/{task}. CI will be NaN.")
                r_low, r_high = np.nan, np.nan
            else:
                # Calculate confidence intervals using Fisher's Z transformation
                z = 0.5 * np.log((1 + correlation) / (1 - correlation)) # Can be NaN if corr is +/-1
                if np.isnan(z): # Handle cases where correlation is exactly 1 or -1
                    if correlation == 1.0 or correlation == -1.0: # Perfect correlation
                         # CI might be considered same as correlation or handled as undefined by some packages
                         # For simplicity, let's set CI to correlation itself, or handle as per stats convention
                         # A more robust CI for perfect correlation might involve specific handling or assumptions
                         logger.debug(f"Perfect correlation ({correlation}) for {model}/{task}. CI bounds set to correlation.")
                         r_low, r_high = correlation, correlation
                    else: # z is NaN for other reasons
                         logger.warning(f"Fisher's Z is NaN for {model}/{task} (corr={correlation}). CI will be NaN.")
                         r_low, r_high = np.nan, np.nan
                else:
                    se = 1 / np.sqrt(n_for_correlation - 3)
                    z_crit = stats.norm.ppf(0.975)  # 95% CI
                    z_low = z - z_crit * se
                    z_high = z + z_crit * se

                    # Transform back to correlation scale
                    r_low = (np.exp(2 * z_low) - 1) / (np.exp(2 * z_low) + 1)
                    r_high = (np.exp(2 * z_high) - 1) / (np.exp(2 * z_high) + 1)

            all_correlations.append({
                "model": model,
                "task": task,
                "correlation": correlation,
                "p_value": p_value,
                "sample_size": n_for_correlation, # Use the actual number of pairs in correlation
                "ci_low": r_low,
                "ci_high": r_high,
                "significant": p_value < 0.05 and not np.isnan(p_value) # ensure p_value is not NaN
            })

    if not all_correlations:
        logger.warning("No valid correlations could be calculated.")
        return {}

    # Convert to DataFrame
    corr_df = pd.DataFrame(all_correlations)

    # Create helper function for truncating task names
    def truncate_task(task_name, max_length=max_task_label_length):
        if len(task_name) <= max_length:
            return task_name
        return task_name[:max_length-3] + "..."

    # Apply truncation to task names
    corr_df["task_display"] = corr_df["task"].apply(truncate_task)

    # Create dictionary to store figures
    figures = {}

    # Create separate plot for each model
    for model in models:
        model_corr_df = corr_df[corr_df["model"] == model].copy()

        if len(model_corr_df) == 0:
            continue

        # Sort results as requested
        if sort_by == "correlation":
            model_corr_df = model_corr_df.sort_values("correlation")
        elif sort_by == "task":
            model_corr_df = model_corr_df.sort_values(["task", "correlation"])
        elif sort_by == "significance":
            # Sort by significance and then by correlation magnitude
            model_corr_df = model_corr_df.sort_values(["significant", "correlation"], ascending=[False, True])
        else:
            logger.warning(f"Invalid sort_by value: {sort_by}, defaulting to correlation")
            model_corr_df = model_corr_df.sort_values("correlation")

        # Create labels
        labels = model_corr_df["task_display"].tolist()

        # Calculate figure height based on number of tasks
        # Ensure a minimum height but also scale with the number of tasks
        fig_height = max(6, min(len(model_corr_df) * 0.5, 30)) # Increased multiplier and max height

        # Determine left margin based on the longest task name
        max_label_len = max(len(label) for label in labels)
        left_margin = max(0.25, min(0.5, 0.012 * max_label_len)) # Adjusted calculation

        # Calculate right margin needed for the annotations
        right_margin = 0.25  # Adjusted based on expected width of annotations

        # Create plot with adjusted margins
        fig, ax = plt.subplots(figsize=(10, fig_height))
        plt.subplots_adjust(left=left_margin, right=(1-right_margin))

        # Plot bars with different colors for significant vs non-significant
        colors = [
            "darkgreen" if row["correlation"] >= 0 and row["significant"] else
            "lightgreen" if row["correlation"] >= 0 else
            "darkred" if row["significant"] and row["correlation"] < 0 else
            "lightcoral"
            for _, row in model_corr_df.iterrows()
        ]

        # Plot horizontal bars
        y_pos = np.arange(len(model_corr_df))
        bars = ax.barh(
            y_pos,
            model_corr_df["correlation"],
            color=colors,
            alpha=0.8,
            height=0.6
        )

        # Add error bars for confidence intervals
        ax.errorbar(
            model_corr_df["correlation"],
            y_pos,
            xerr=np.array([
                model_corr_df["correlation"] - model_corr_df["ci_low"],
                model_corr_df["ci_high"] - model_corr_df["correlation"]
            ]),
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=3
        )

        # Add labels and ticks
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8) # Reduced y-tick label font size

        # Add a vertical line at x=0
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)

        # Set x-axis limits to ensure there's space for annotations
        x_max = max(0.1, model_corr_df["correlation"].max() + 0.35) # Increased space for annotations
        x_min = min(-0.1, model_corr_df["correlation"].min() - 0.35) # Increased space for annotations
        ax.set_xlim(x_min, x_max)

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.3, axis="x")

        # Add axis labels and title
        xlabel_for_plot = f"Pearson Correlation ({y_label} vs. Aggregated {effective_x_correlation_label})"
        title_for_plot = f"Correlation between Aggregated {effective_x_correlation_label} and {y_label} for {model}"
        if log_transform_x_for_corr and "Log(" not in effective_x_correlation_label: # If transformation failed
            title_for_plot += " (Original X)"
            xlabel_for_plot += " (Original X)"

        ax.set_xlabel(xlabel_for_plot)
        ax.set_title(title_for_plot)

        # Add values at the end of each bar
        for i, bar in enumerate(bars):
            width = bar.get_width()
            p_value = model_corr_df.iloc[i]["p_value"]
            sample_size = model_corr_df.iloc[i]["sample_size"]

            # Format p-value with asterisks for significance
            if p_value < 0.001:
                p_text = "***"
            elif p_value < 0.01:
                p_text = "**"
            elif p_value < 0.05:
                p_text = "*"
            else:
                p_text = ""

            label_text = f"r={width:.2f}{p_text} (n={sample_size})"

            # Position text based on correlation direction
            if width >= 0:
                x_pos = width + 0.02
                ha = "left"
            else:
                x_pos = width - 0.02
                ha = "right"

            ax.text(
                x_pos,
                bar.get_y() + bar.get_height() / 2,
                label_text,
                ha=ha,
                va="center",
                fontsize=7  # Reduced annotation font size
            )

        # Add legend with improved positioning
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, color="darkgreen", alpha=0.8, label="Positive significant"),
            plt.Rectangle((0, 0), 1, 1, color="lightgreen", alpha=0.8, label="Positive non-significant"),
            plt.Rectangle((0, 0), 1, 1, color="darkred", alpha=0.8, label="Negative significant"),
            plt.Rectangle((0, 0), 1, 1, color="lightcoral", alpha=0.8, label="Negative non-significant")
        ]

        # Position legend at the bottom of the plot
        ax.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05 / (fig_height / 10)), # Adjust legend based on figure height
            ncol=2,
            fontsize=8 # Reduced legend font size
        )

        # Add note about significance
        ax.text(
            x_max - 0.02,
            -0.8,
            "* p<0.05, ** p<0.01, *** p<0.001",
            transform=ax.transAxes,
            fontsize=7, # Reduced significance note font size
            ha="right",
            alpha=0.7
        )

        # Adjust layout
        fig.tight_layout()

        # Store figure
        figures[f"{model}_token_correlation"] = fig

        # Save if output_dir is specified
        if output_dir is not None:
            output_dir.mkdir(exist_ok=True, parents=True)
            fig_name = f"{model}_token_correlations_{plot_type}_vs_{x_axis_type}"
            if not include_zero_tokens:
                fig_name += "_nonzero_tokens_only"

            fig.savefig(
                output_dir / f"{fig_name}.pdf", # Changed to PDF
                bbox_inches="tight"
            )

    # Create combined plot if requested
    if combined_plot:
        # Sort results
        if sort_by == "correlation":
            corr_df = corr_df.sort_values("correlation")
        elif sort_by == "model":
            corr_df = corr_df.sort_values(["model", "correlation"])
        elif sort_by == "task":
            corr_df = corr_df.sort_values(["task", "correlation"])
        elif sort_by == "significance":
            corr_df = corr_df.sort_values(["significant", "correlation"], ascending=[False, True])
        else:
            corr_df = corr_df.sort_values("correlation")

        # Create combined labels
        labels = [f"{row['model']} - {row['task_display']}" for _, row in corr_df.iterrows()]

        # Calculate figure height based on number of entries
        fig_height = max(10, min(len(corr_df) * 0.35, 40)) # Increased multiplier and max height

        # Determine left margin based on the longest label
        max_label_len = max(len(label) for label in labels)
        left_margin = max(0.25, min(0.5, 0.01 * max_label_len)) # Adjusted calculation

        # Create combined plot
        fig, ax = plt.subplots(figsize=(10, fig_height))
        plt.subplots_adjust(left=left_margin, right=0.8) # Adjusted right margin

        # Colors for bars
        colors = [
            "darkgreen" if row["correlation"] >= 0 and row["significant"] else
            "lightgreen" if row["correlation"] >= 0 else
            "darkred" if row["significant"] and row["correlation"] < 0 else
            "lightcoral"
            for _, row in corr_df.iterrows()
        ]

        # Plot horizontal bars
        y_pos = np.arange(len(corr_df))
        bars = ax.barh(
            y_pos,
            corr_df["correlation"],
            color=colors,
            alpha=0.8,
            height=0.6
        )

        # Add error bars
        ax.errorbar(
            corr_df["correlation"],
            y_pos,
            xerr=np.array([
                corr_df["correlation"] - corr_df["ci_low"],
                corr_df["ci_high"] - corr_df["correlation"]
            ]),
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=3
        )

        # Add labels and ticks
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7) # Reduced y-tick label font size

        # Add a vertical line at x=0
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)

        # Set x-axis limits
        x_max = max(0.1, corr_df["correlation"].max() + 0.35) # Increased space for annotations
        x_min = min(-0.1, corr_df["correlation"].min() - 0.35) # Increased space for annotations
        ax.set_xlim(x_min, x_max)

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.3, axis="x")

        # Add axis labels and title
        # Determine the final x_correlation_label based on whether transformation generally happened
        final_x_corr_label_display = x_correlation_label
        if log_transform_x_for_corr:
            # This is tricky because transformation might succeed for some model/task pairs and fail for others.
            # We'll assume if the flag is on, we intend to show "Log(...)" but note if any failed.
            # For combined plot, it's best to be generic or indicate variability.
            # For simplicity in combined plot, let's use a more general label if log_transform_x_for_corr is true.
            final_x_corr_label_display = f"Log({x_correlation_label})" if log_transform_x_for_corr else x_correlation_label
            # A more robust approach would be to check if ANY transformation occurred.

        xlabel_for_combined_plot = f"Pearson Correlation ({y_label} vs. Aggregated {final_x_corr_label_display})"
        title_for_combined_plot = f"Correlation between Aggregated {final_x_corr_label_display} and {y_label} for All Models"

        ax.set_xlabel(xlabel_for_combined_plot)
        ax.set_title(title_for_combined_plot)

        # Add values at the end of each bar
        for i, bar in enumerate(bars):
            width = bar.get_width()
            p_value = corr_df.iloc[i]["p_value"]
            sample_size = corr_df.iloc[i]["sample_size"]

            # Format p-value with asterisks for significance
            if p_value < 0.001:
                p_text = "***"
            elif p_value < 0.01:
                p_text = "**"
            elif p_value < 0.05:
                p_text = "*"
            else:
                p_text = ""

            label_text = f"r={width:.2f}{p_text} (n={sample_size})"

            # Position text based on correlation direction
            if width >= 0:
                x_pos = width + 0.02
                ha = "left"
            else:
                x_pos = width - 0.02
                ha = "right"

            ax.text(
                x_pos,
                bar.get_y() + bar.get_height() / 2,
                label_text,
                ha=ha,
                va="center",
                fontsize=7 # Reduced annotation font size
            )

        # Add legend with improved positioning
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, color="darkgreen", alpha=0.8, label="Positive significant"),
            plt.Rectangle((0, 0), 1, 1, color="lightgreen", alpha=0.8, label="Positive non-significant"),
            plt.Rectangle((0, 0), 1, 1, color="darkred", alpha=0.8, label="Negative significant"),
            plt.Rectangle((0, 0), 1, 1, color="lightcoral", alpha=0.8, label="Negative non-significant")
        ]

        # Position legend at the bottom of the plot for larger plots
        if len(corr_df) > 20:
            ax.legend(
                handles=legend_elements,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.03 / (fig_height / 15)), # Adjust legend based on figure height
                ncol=2,
                fontsize=8 # Reduced legend font size
            )
        else:
            # For smaller plots, position in the lower right
            ax.legend(
                handles=legend_elements,
                loc="lower right",
                fontsize=8 # Reduced legend font size
            )

        # Add note about significance
        ax.text(
            0.98, 0.015, # Adjusted y position
            "* p<0.05, ** p<0.01, *** p<0.001",
            transform=ax.transAxes,
            fontsize=7, # Reduced significance note font size
            ha="right",
            alpha=0.7
        )

        # Adjust layout
        fig.tight_layout()

        # Store combined figure
        figures["combined_token_correlation"] = fig

        # Save if output_dir is specified
        if output_dir is not None:
            fig_name = f"all_models_correlations_{plot_type}_vs_{x_axis_type}"
            if not include_zero_tokens:
                fig_name += "_nonzero_tokens_only"

            fig.savefig(
                output_dir / f"{fig_name}.pdf", # Changed to PDF
                bbox_inches="tight"
            )

    return figures


def plot_token_slopes(
    df: pd.DataFrame,
    models: List[str] = None,
    tasks: List[str] = None,
    output_dir: Optional[Path] = None,
    min_samples: int = 2,  # Minimum number of aggregated budget groups required to calculate slope
    sort_by: str = "slope",  # "slope", "task", "significance"
    include_zero_tokens: bool = True,  # Whether to include zero tokens in slope calculation
    plot_type: str = "accuracy",  # "accuracy", "cost", "latency", "mse", "relative_error"
    x_axis_type: str = "tokens",  # "tokens" or "budget"
    log_transform_x_for_slope: bool = True, # Whether to log-transform x-axis data for slope calculation
    combined_plot: bool = False,  # Whether to also create a combined plot with all models
    max_task_label_length: int = 25,  # Maximum length for task labels before truncation
    min_samples_per_budget_group: int = 5, # Min raw samples per budget group for it to be included
) -> Dict[str, plt.Figure]:
    """
    Plot linear regression slopes. Data is first aggregated by 'reasoning_budget'.
    The x-axis for slope calculation can be the mean 'reasoning_content_tokens' per budget group
    (if x_axis_type='tokens') or the 'reasoning_budget' value itself (if x_axis_type='budget').
    The y-axis is the mean of the specified metric (e.g., accuracy) per budget group.
    The slope represents the rate of change in the metric per unit change in x.

    Args:
        df: DataFrame with results.
        models: List of model IDs to include (default: all).
        tasks: List of task IDs to include (default: all).
        output_dir: Directory to save plots (default: don't save).
        min_samples: Minimum number of aggregated budget groups required to calculate slope
                     for a model-task pair. Default is 2.
        sort_by: How to sort the results ("slope", "task", "significance").
        include_zero_tokens: Whether to include samples with zero reasoning_content_tokens
                             when forming budget groups. This filter is applied to raw data.
        plot_type: Type of metric to calculate slope with respect to the x-axis variable
                   ("accuracy", "cost", "latency", "mse", "relative_error").
        x_axis_type: Variable for the x-axis of the slope calculation:
                     "tokens" (mean reasoning_content_tokens per budget group) or
                     "budget" (reasoning_budget value). Defaults to "tokens".
        log_transform_x_for_slope: If True, the x-axis data (tokens or budget)
                                   will be log-transformed before calculating slope.
                                   Requires x-values to be positive. Defaults to False.
        combined_plot: Whether to also create a combined plot with all models.
        max_task_label_length: Maximum length for task labels before truncation.
        min_samples_per_budget_group: Minimum number of raw data instances within a budget group
                                      for that group to be included in aggregation and slope calculation.
                                      Default is 5.

    Returns:
        Dictionary mapping plot names to Figure objects.
    """
    set_plotting_style()

    if models is None:
        models = df["model"].unique()
    if tasks is None:
        tasks = df["task_id"].unique()

    # Select the appropriate column based on plot_type
    if plot_type == "accuracy":
        y_column = "correct"
        y_label = "Accuracy"
        slope_unit = "per token" if x_axis_type == "tokens" else "per budget unit"
    elif plot_type == "cost":
        y_column = "cost"
        y_label = "Cost"
        slope_unit = "$ per token" if x_axis_type == "tokens" else "$ per budget unit"
    elif plot_type == "latency":
        y_column = "latency"
        y_label = "Latency"
        slope_unit = "s per token" if x_axis_type == "tokens" else "s per budget unit"
    elif plot_type == "mse":
        y_column = "squared_error"
        y_label = "Mean Squared Error"
        slope_unit = "MSE per token" if x_axis_type == "tokens" else "MSE per budget unit"
    elif plot_type == "relative_error":
        y_column = "relative_error"
        y_label = "Relative Error"
        slope_unit = "Relative Error per token" if x_axis_type == "tokens" else "Relative Error per budget unit"
    else:
        raise ValueError(f"Invalid plot_type: {plot_type}")

    # Determine x-axis column and label
    x_slope_column = ""
    x_slope_label = ""
    if x_axis_type == "tokens":
        x_slope_column = "reasoning_content_tokens"
        x_slope_label = "Reasoning Tokens"
    elif x_axis_type == "budget":
        x_slope_column = "reasoning_budget"
        x_slope_label = "Reasoning Budget (Tokens)"
        df[x_slope_column] = pd.to_numeric(df[x_slope_column], errors='coerce')
    else:
        raise ValueError(f"Invalid x_axis_type: {x_axis_type}. Must be 'tokens' or 'budget'.")

    # Ensure reasoning_content_tokens is present if needed for filtering or as the x-column
    if "reasoning_content_tokens" not in df.columns:
        df["reasoning_content_tokens"] = df.apply(_safe_count_tokens, axis=1)

    # Ensure reasoning_budget is numeric for grouping and potential use as x-column
    df["reasoning_budget"] = pd.to_numeric(df["reasoning_budget"], errors='coerce')

    # Filter out rows with missing essential values BEFORE aggregation
    df = df.dropna(subset=[x_slope_column, y_column, "reasoning_budget"])

    all_slopes = []

    for model in models:
        for task in tasks:
            model_task_df_initial = df[(df["model"] == model) & (df["task_id"] == task)].copy()

            # Apply include_zero_tokens filter to the raw data for this model-task pair
            if not include_zero_tokens:
                model_task_df_initial = model_task_df_initial[model_task_df_initial["reasoning_content_tokens"] > 0]

            if model_task_df_initial.empty:
                logger.debug(f"No data for {model}/{task} after initial filtering and zero token check.")
                continue

            # Group by 'reasoning_budget'
            grouped_by_budget = model_task_df_initial.groupby("reasoning_budget")

            aggregated_x_values = []
            aggregated_y_values = []

            for budget_value, budget_group_df in grouped_by_budget:
                if len(budget_group_df) < min_samples_per_budget_group:
                    logger.debug(f"Skipping budget group {budget_value} for {model}/{task} due to insufficient samples: {len(budget_group_df)} < {min_samples_per_budget_group}")
                    continue

                # Calculate Y value (mean of the metric for this budget group)
                current_y_metric_data = budget_group_df[y_column].astype(float).dropna()
                if len(current_y_metric_data) < min_samples_per_budget_group:
                    logger.debug(f"Skipping budget group {budget_value} for {model}/{task} due to insufficient y-metric samples after dropna: {len(current_y_metric_data)} < {min_samples_per_budget_group}")
                    continue

                mean_y_for_group = current_y_metric_data.mean()
                if pd.isna(mean_y_for_group):
                    logger.debug(f"Skipping budget group {budget_value} for {model}/{task} due to NaN mean y-metric.")
                    continue

                # Calculate X value based on x_axis_type
                current_x_val_for_group = np.nan
                if x_axis_type == "tokens":
                    tokens_for_group = budget_group_df[x_slope_column].astype(float).dropna()
                    if len(tokens_for_group) < min_samples_per_budget_group:
                        logger.debug(f"Skipping budget group {budget_value} for {model}/{task} due to insufficient token samples after dropna: {len(tokens_for_group)} < {min_samples_per_budget_group}")
                        continue
                    if not tokens_for_group.empty:
                        current_x_val_for_group = tokens_for_group.mean()
                    else:
                        logger.debug(f"Skipping budget group {budget_value} for {model}/{task} due to empty token data after filtering.")
                        continue
                elif x_axis_type == "budget":
                    current_x_val_for_group = float(budget_value)

                if pd.isna(current_x_val_for_group):
                    logger.debug(f"Skipping budget group {budget_value} for {model}/{task} due to NaN aggregated x-value.")
                    continue

                aggregated_x_values.append(current_x_val_for_group)
                aggregated_y_values.append(mean_y_for_group)

            n_for_slope = len(aggregated_x_values)

            if n_for_slope < min_samples:
                logger.warning(f"Not enough aggregated budget groups ({n_for_slope}) for {model}/{task} to calculate slope (min_samples: {min_samples}). Skipping.")
                continue

            # Convert aggregated lists to numpy arrays for linregress
            x_array = np.array(aggregated_x_values, dtype=float)
            y_array = np.array(aggregated_y_values, dtype=float)

            # Potentially log-transform x_array for slope calculation
            effective_x_slope_label = x_slope_label
            if log_transform_x_for_slope:
                if np.any(x_array < 0):
                    logger.warning(
                        f"Cannot apply log-like transformation to x-axis data for {model}/{task} because it contains negative values. "
                        "Slope will be calculated on original x-values."
                    )
                else:  # All values are >= 0, safe for np.log1p
                    x_array = np.log1p(x_array)  # Computes log(1+x)
                    effective_x_slope_label = f"Log(1+{x_slope_label})"
                    logger.info(f"Applied log1p (log(1+x)) transformation to x-axis data for slope calculation for {model}/{task}.")

            # Check for variance AFTER aggregation and potential transformation
            if np.std(x_array) < 1e-9 or np.std(y_array) < 1e-9:
                logger.warning(f"Skipping slope calculation for {model}/{task} due to zero or near-zero variance in aggregated x or y values (x_std={np.std(x_array):.2e}, y_std={np.std(y_array):.2e}).")
                slope, intercept, r_value, p_value, std_err = np.nan, np.nan, np.nan, np.nan, np.nan
            else:
                # Calculate linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_array, y_array)

            # Calculate confidence intervals for slope
            if np.isnan(slope) or np.isnan(std_err):
                logger.warning(f"Linear regression resulted in NaN slope or std_err for {model}/{task}. Skipping CI calculation.")
                slope_low, slope_high = np.nan, np.nan
            elif n_for_slope <= 2:
                logger.warning(f"Not enough samples ({n_for_slope}) for slope confidence interval for {model}/{task}. CI will be NaN.")
                slope_low, slope_high = np.nan, np.nan
            else:
                # Calculate 95% confidence interval for slope
                t_crit = stats.t.ppf(0.975, n_for_slope - 2)  # 95% CI with n-2 degrees of freedom
                margin_of_error = t_crit * std_err
                slope_low = slope - margin_of_error
                slope_high = slope + margin_of_error

            all_slopes.append({
                "model": model,
                "task": task,
                "slope": slope,
                "intercept": intercept,
                "r_value": r_value,
                "p_value": p_value,
                "std_err": std_err,
                "sample_size": n_for_slope,
                "slope_low": slope_low,
                "slope_high": slope_high,
                "significant": p_value < 0.05 and not np.isnan(p_value)
            })

    if not all_slopes:
        logger.warning("No valid slopes could be calculated.")
        return {}

    # Convert to DataFrame
    slope_df = pd.DataFrame(all_slopes)

    # Create helper function for truncating task names
    def truncate_task(task_name, max_length=max_task_label_length):
        if len(task_name) <= max_length:
            return task_name
        return task_name[:max_length-3] + "..."

    # Apply truncation to task names
    slope_df["task_display"] = slope_df["task"].apply(truncate_task)

    # Create dictionary to store figures
    figures = {}

    # Create separate plot for each model
    for model in models:
        model_slope_df = slope_df[slope_df["model"] == model].copy()

        if len(model_slope_df) == 0:
            continue

        # Sort results as requested
        if sort_by == "slope":
            model_slope_df = model_slope_df.sort_values("slope")
        elif sort_by == "task":
            model_slope_df = model_slope_df.sort_values(["task", "slope"])
        elif sort_by == "significance":
            model_slope_df = model_slope_df.sort_values(["significant", "slope"], ascending=[False, True])
        else:
            logger.warning(f"Invalid sort_by value: {sort_by}, defaulting to slope")
            model_slope_df = model_slope_df.sort_values("slope")

        # Create labels
        labels = model_slope_df["task_display"].tolist()

        # Calculate figure height based on number of tasks
        fig_height = max(6, min(len(model_slope_df) * 0.5, 30))

        # Determine left margin based on the longest task name
        max_label_len = max(len(label) for label in labels)
        left_margin = max(0.25, min(0.5, 0.012 * max_label_len))

        # Calculate right margin needed for the annotations
        right_margin = 0.25

        # Create plot with adjusted margins
        fig, ax = plt.subplots(figsize=(10, fig_height))
        plt.subplots_adjust(left=left_margin, right=(1-right_margin))

        # Plot bars with different colors for significant vs non-significant
        colors = [
            "darkgreen" if row["slope"] >= 0 and row["significant"] else
            "lightgreen" if row["slope"] >= 0 else
            "darkred" if row["significant"] and row["slope"] < 0 else
            "lightcoral"
            for _, row in model_slope_df.iterrows()
        ]

        # Plot horizontal bars
        y_pos = np.arange(len(model_slope_df))
        bars = ax.barh(
            y_pos,
            model_slope_df["slope"],
            color=colors,
            alpha=0.8,
            height=0.6
        )

        # Add error bars for confidence intervals
        ax.errorbar(
            model_slope_df["slope"],
            y_pos,
            xerr=np.array([
                model_slope_df["slope"] - model_slope_df["slope_low"],
                model_slope_df["slope_high"] - model_slope_df["slope"]
            ]),
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=3
        )

        # Add labels and ticks
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)

        # Add a vertical line at x=0
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)

        # Set x-axis limits to ensure there's space for annotations
        x_max = max(0.1, model_slope_df["slope"].max() + 0.35)
        x_min = min(-0.1, model_slope_df["slope"].min() - 0.35)
        ax.set_xlim(x_min, x_max)

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.3, axis="x")

        # Add axis labels and title
        xlabel_for_plot = f"Linear Regression Slope ({y_label} vs. Aggregated {effective_x_slope_label}) [{slope_unit}]"
        title_for_plot = f"Slope of {y_label} vs. Aggregated {effective_x_slope_label} for {model}"
        if log_transform_x_for_slope and "Log(" not in effective_x_slope_label:
            title_for_plot += " (Original X)"
            xlabel_for_plot += " (Original X)"

        ax.set_xlabel(xlabel_for_plot)
        ax.set_title(title_for_plot)

        # Add values at the end of each bar
        for i, bar in enumerate(bars):
            width = bar.get_width()
            p_value = model_slope_df.iloc[i]["p_value"]
            sample_size = model_slope_df.iloc[i]["sample_size"]
            r_value = model_slope_df.iloc[i]["r_value"]

            # Format p-value with asterisks for significance
            if p_value < 0.001:
                p_text = "***"
            elif p_value < 0.01:
                p_text = "**"
            elif p_value < 0.05:
                p_text = "*"
            else:
                p_text = ""

            label_text = f"m={width:.2e}{p_text} (r²={r_value**2:.2f}, n={sample_size})"

            # Position text based on slope direction
            if width >= 0:
                x_pos = width + 0.02
                ha = "left"
            else:
                x_pos = width - 0.02
                ha = "right"

            ax.text(
                x_pos,
                bar.get_y() + bar.get_height() / 2,
                label_text,
                ha=ha,
                va="center",
                fontsize=7
            )

        # Add legend with improved positioning
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, color="darkgreen", alpha=0.8, label="Positive significant"),
            plt.Rectangle((0, 0), 1, 1, color="lightgreen", alpha=0.8, label="Positive non-significant"),
            plt.Rectangle((0, 0), 1, 1, color="darkred", alpha=0.8, label="Negative significant"),
            plt.Rectangle((0, 0), 1, 1, color="lightcoral", alpha=0.8, label="Negative non-significant")
        ]

        # Position legend at the bottom of the plot
        ax.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05 / (fig_height / 10)),
            ncol=2,
            fontsize=8
        )

        # Add note about significance
        ax.text(
            0.98, 0.015,
            "* p<0.05, ** p<0.01, *** p<0.001",
            transform=ax.transAxes,
            fontsize=7,
            ha="right",
            alpha=0.7
        )

        # Adjust layout
        fig.tight_layout()

        # Store figure
        figures[f"{model}_token_slope"] = fig

        # Save if output_dir is specified
        if output_dir is not None:
            output_dir.mkdir(exist_ok=True, parents=True)
            fig_name = f"{model}_token_slopes_{plot_type}_vs_{x_axis_type}"
            if not include_zero_tokens:
                fig_name += "_nonzero_tokens_only"

            fig.savefig(
                output_dir / f"{fig_name}.pdf",
                bbox_inches="tight"
            )

    # Create combined plot if requested
    if combined_plot:
        # Sort results
        if sort_by == "slope":
            slope_df = slope_df.sort_values("slope")
        elif sort_by == "model":
            slope_df = slope_df.sort_values(["model", "slope"])
        elif sort_by == "task":
            slope_df = slope_df.sort_values(["task", "slope"])
        elif sort_by == "significance":
            slope_df = slope_df.sort_values(["significant", "slope"], ascending=[False, True])
        else:
            slope_df = slope_df.sort_values("slope")

        # Create combined labels
        labels = [f"{row['model']} - {row['task_display']}" for _, row in slope_df.iterrows()]

        # Calculate figure height based on number of entries
        fig_height = max(10, min(len(slope_df) * 0.35, 40))

        # Determine left margin based on the longest label
        max_label_len = max(len(label) for label in labels)
        left_margin = max(0.25, min(0.5, 0.01 * max_label_len))

        # Create combined plot
        fig, ax = plt.subplots(figsize=(10, fig_height))
        plt.subplots_adjust(left=left_margin, right=0.8)

        # Colors for bars
        colors = [
            "darkgreen" if row["slope"] >= 0 and row["significant"] else
            "lightgreen" if row["slope"] >= 0 else
            "darkred" if row["significant"] and row["slope"] < 0 else
            "lightcoral"
            for _, row in slope_df.iterrows()
        ]

        # Plot horizontal bars
        y_pos = np.arange(len(slope_df))
        bars = ax.barh(
            y_pos,
            slope_df["slope"],
            color=colors,
            alpha=0.8,
            height=0.6
        )

        # Add error bars
        ax.errorbar(
            slope_df["slope"],
            y_pos,
            xerr=np.array([
                slope_df["slope"] - slope_df["slope_low"],
                slope_df["slope_high"] - slope_df["slope"]
            ]),
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=3
        )

        # Add labels and ticks
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)

        # Add a vertical line at x=0
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)

        # Set x-axis limits
        x_max = max(0.1, slope_df["slope"].max() + 0.35)
        x_min = min(-0.1, slope_df["slope"].min() - 0.35)
        ax.set_xlim(x_min, x_max)

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.3, axis="x")

        # Add axis labels and title
        final_x_slope_label_display = x_slope_label
        if log_transform_x_for_slope:
            final_x_slope_label_display = f"Log({x_slope_label})" if log_transform_x_for_slope else x_slope_label

        xlabel_for_combined_plot = f"Linear Regression Slope ({y_label} vs. Aggregated {final_x_slope_label_display}) [{slope_unit}]"
        title_for_combined_plot = f"Slope of {y_label} vs. Aggregated {final_x_slope_label_display} for All Models"

        ax.set_xlabel(xlabel_for_combined_plot)
        ax.set_title(title_for_combined_plot)

        # Add values at the end of each bar
        for i, bar in enumerate(bars):
            width = bar.get_width()
            p_value = slope_df.iloc[i]["p_value"]
            sample_size = slope_df.iloc[i]["sample_size"]
            r_value = slope_df.iloc[i]["r_value"]

            # Format p-value with asterisks for significance
            if p_value < 0.001:
                p_text = "***"
            elif p_value < 0.01:
                p_text = "**"
            elif p_value < 0.05:
                p_text = "*"
            else:
                p_text = ""

            label_text = f"m={width:.2e}{p_text} (r²={r_value**2:.2f}, n={sample_size})"

            # Position text based on slope direction
            if width >= 0:
                x_pos = width + 0.02
                ha = "left"
            else:
                x_pos = width - 0.02
                ha = "right"

            ax.text(
                x_pos,
                bar.get_y() + bar.get_height() / 2,
                label_text,
                ha=ha,
                va="center",
                fontsize=7
            )

        # Add legend with improved positioning
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, color="darkgreen", alpha=0.8, label="Positive significant"),
            plt.Rectangle((0, 0), 1, 1, color="lightgreen", alpha=0.8, label="Positive non-significant"),
            plt.Rectangle((0, 0), 1, 1, color="darkred", alpha=0.8, label="Negative significant"),
            plt.Rectangle((0, 0), 1, 1, color="lightcoral", alpha=0.8, label="Negative non-significant")
        ]

        # Position legend at the bottom of the plot for larger plots
        if len(slope_df) > 20:
            ax.legend(
                handles=legend_elements,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.03 / (fig_height / 15)),
                ncol=2,
                fontsize=8
            )
        else:
            # For smaller plots, position in the lower right
            ax.legend(
                handles=legend_elements,
                loc="lower right",
                fontsize=8
            )

        # Add note about significance
        ax.text(
            0.98, 0.015,
            "* p<0.05, ** p<0.01, *** p<0.001",
            transform=ax.transAxes,
            fontsize=7,
            ha="right",
            alpha=0.7
        )

        # Adjust layout
        fig.tight_layout()

        # Store combined figure
        figures["combined_token_slope"] = fig

        # Save if output_dir is specified
        if output_dir is not None:
            fig_name = f"all_models_slopes_{plot_type}_vs_{x_axis_type}"
            if not include_zero_tokens:
                fig_name += "_nonzero_tokens_only"

            fig.savefig(
                output_dir / f"{fig_name}.pdf",
                bbox_inches="tight"
            )

    return figures


def plot_budget_length_boxplot(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    model_subset: Optional[List[str]] = None,
    task_subset: Optional[List[str]] = None,
    task_loader: Optional[TaskLoader] = None,
) -> Optional[plt.Figure]:
    """
    Plot a boxplot of actual reasoning content tokens for each requested reasoning budget.

    Args:
        df: DataFrame with results. Requires 'reasoning_budget' and 'reasoning_content'.
        output_dir: Directory to save plot (default: don't save).
        model_subset: Optional list of models to include. Plots all if None.
        task_subset: Optional list of task IDs to include. Plots all if None.

    Returns:
        Figure object or None if plotting is not possible.
    """
    set_plotting_style()

    plot_df = df.copy()

    # Filter by model and task if subsets are provided
    if model_subset:
        plot_df = plot_df[plot_df["model"].isin(model_subset)]
    # We will filter by task inside the loop
    # if task_subset:
    #     plot_df = plot_df[plot_df["task_id"].isin(task_subset)]

    if plot_df.empty:
        logger.warning("Initial DataFrame is empty or became empty after model filtering.")
        return None # Return None if no data at all

    # --- Determine Tasks to Plot ---
    all_tasks = plot_df["task_id"].unique()
    if task_subset:
        tasks_to_plot = [task for task in all_tasks if task in task_subset]
    else:
        tasks_to_plot = all_tasks

    if len(tasks_to_plot) == 0:
        logger.warning("No tasks selected or found in the data after filtering.")
        return {} # Return empty dict if no tasks to plot

    figures = {} # Dictionary to store figures per task

    # --- Loop Through Each Task ---
    for task_id in tasks_to_plot:
        logger.info(f"Generating budget vs length boxplot for task: {task_id}")
        task_df = plot_df[plot_df["task_id"] == task_id].copy()

        if task_df.empty:
            logger.warning(f"No data for task '{task_id}'. Skipping boxplot.")
            continue

        # Calculate reasoning tokens using the imported function
        # Use the module-level helper function
        task_df["reasoning_content_tokens"] = task_df.apply(_safe_count_tokens, axis=1)

        # Ensure budget is numeric, drop if not
        task_df["reasoning_budget"] = pd.to_numeric(
            task_df["reasoning_budget"], errors="coerce"
        )
        # Drop rows where budget or calculated tokens are missing FOR THIS TASK
        task_df = task_df.dropna(subset=["reasoning_budget", "reasoning_content_tokens"])

        if task_df.empty:
            logger.warning(f"No valid numeric budget/reasoning_token pairs found for task '{task_id}'. Skipping boxplot.")
            continue

        # Ensure budgets are treated as categories for plotting if few unique values
        # Or sort numerically if many budgets
        unique_budgets = sorted(task_df["reasoning_budget"].unique())
        # Check if there's enough data points for a meaningful boxplot
        if len(unique_budgets) < 1 or task_df.shape[0] < 5: # Arbitrary threshold, maybe adjust
            logger.warning(f"Not enough distinct budgets or data points for task '{task_id}'. Skipping boxplot.")
            continue

        budget_order = [str(int(b)) for b in unique_budgets] # Treat as categories for distinct boxes, ensure integer format
        task_df["reasoning_budget_cat"] = task_df["reasoning_budget"].astype(int).astype(str)

        # --- Plotting for the current task ---
        fig, ax = plt.subplots(figsize=(12, 7)) # Adjust size as needed

        sns.boxplot(
            x="reasoning_budget_cat",
            y="reasoning_content_tokens", # Use reasoning_content_tokens for Y-axis
            data=task_df,
            ax=ax,
            order=budget_order, # Ensure boxes are ordered numerically
            palette="viridis",
            showfliers=False, # Optionally hide outliers if plot is too cluttered
        )

        # Optional: Add stripplot for individual points (can be too dense)
        # sns.stripplot(x="reasoning_budget_cat", y="reasoning_content_tokens", data=plot_df, ax=ax,
        #               order=budget_order, color='.3', size=3, jitter=True, alpha=0.1)

        # Add y=x reference line for the budget value itself
        try:
            budget_values_numeric = [float(b) for b in budget_order]
            # Only plot if there are numeric budgets to plot against
            if budget_values_numeric:
                 ax.plot(range(len(budget_order)), budget_values_numeric, color="red", linestyle="--", alpha=0.8, marker='_', markersize=10, label="Requested Budget (Tokens)")
                 ax.legend()
        except ValueError:
            logger.warning(f"Could not plot y=x reference line for budgets in task '{task_id}'.")


        ax.set_xlabel("Requested Reasoning Budget (Tokens)")
        ax.set_ylabel("Actual Reasoning Content Tokens") # Updated label
        # Make title task-specific
        task_pretty_name = get_task_pretty_name(task_id, task_loader)
        ax.set_title(f"{task_pretty_name}", fontsize=16)
        
        # Improved x-axis tick handling to prevent overlapping
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        # Limit number of x-axis ticks if there are too many
        if len(budget_order) > 8:
            # Show every other tick if too many budget values
            tick_positions = range(0, len(budget_order), max(1, len(budget_order) // 6))
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([budget_order[i] for i in tick_positions], rotation=45, ha='right')
        else:
            ax.set_xticklabels(budget_order, rotation=45, ha='right')

        ax.grid(True, linestyle="--", alpha=0.6, axis='y') # Grid on y-axis is often helpful
        
        # Adjust layout to prevent label cutoff
        fig.tight_layout()

        # Save the figure for the current task
        if output_dir is not None:
            output_dir.mkdir(exist_ok=True, parents=True)
            # Make filename task-specific
            plot_filename = f"boxplot_budget_vs_tokens_{task_id}.pdf"
            fig.savefig(
                output_dir / plot_filename,
                dpi=300,
                bbox_inches="tight",
            )
            logger.info(f"Saved budget vs reasoning tokens boxplot to {output_dir / plot_filename}")

        # Store the figure in the dictionary
        figures[task_id] = fig
        plt.close(fig) # Close figure to prevent displaying it inline if not desired and save memory

    return figures # Return the dictionary of figures


def plot_token_scaling_curves_improved(
    df: pd.DataFrame,
    task_loader: TaskLoader,
    models: List[str] = None,
    tasks: List[str] = None,
    output_dir: Optional[Path] = None,
    plot_type: str = "accuracy",
    x_axis_type: str = "tokens",  # "tokens" or "budget"
    min_samples_per_point: int = 5,  # Minimum samples required for a point
    scaling_columns: Optional[Dict[str, str]] = None,
) -> Dict[str, plt.Figure]:
    """
    Plot improved scaling curves for tasks, aggregating by reasoning_budget.
    X-axis can be actual mean tokens used or the requested budget.
    Y-axis shows mean performance (accuracy, cost, latency, or -RMSE for MSE, or -Relative Error for relative_error).
    Error bars represent 95% CI for accuracy, and SEM for other metrics.
    """
    set_plotting_style()

    # Use crest colormap for scaling factors from seaborn
    # import matplotlib.cm as cm # Already imported globally in script
    import matplotlib.colors as mcolors # Already imported globally

    # Color palette for models
    color_palette_sns = sns.color_palette() # Renamed to avoid conflict if 'color_palette' is used later
    model_id_to_color = {
        "claude-sonnet-4-20250514": color_palette_sns[1],
        "claude-opus-4-20250514": color_palette_sns[7],
        "claude-3-7-sonnet-20250219": color_palette_sns[8],
        "claude-3-7-sonnet-20250219_natural_overthinking": color_palette_sns[7],
        "claude-3-7-sonnet-20250219_not_use_all_budget": color_palette_sns[8],
        "o3-2025-04-16": color_palette_sns[2],
        "o3-mini-2025-01-31": color_palette_sns[9],
        "O4-mini": color_palette_sns[3],
        "o4-mini-2025-04-16": color_palette_sns[3],
        "deepseek-reasoner": color_palette_sns[0],
        "Qwen3-8B": color_palette_sns[4],
        "Qwen3-14B": color_palette_sns[5],
        "Qwen3-32B": color_palette_sns[6],
    }

    # Mappings for prettier names (as provided by user)
    model_to_model_name = MODEL_TO_PRETTY_NAME
    
    # Create task name mapping from TaskLoader
    task_to_name = {}
    for task_id in tasks:
        task_to_name[task_id] = get_task_pretty_name(task_id, task_loader)


    if models is None:
        models = df["model"].unique().tolist()
    if tasks is None:
        tasks = df["task_id"].unique().tolist()

    df_enriched = df.copy()

    if scaling_columns is None:
        logger.info("scaling_columns not provided, attempting to infer based on task names.")
        scaling_columns = {}
        for task_id in tasks:
            if "misleading" in task_id: scaling_columns[task_id] = "num_distractors"
            elif "adversarial_pattern" in task_id: scaling_columns[task_id] = "sequence_length"
            elif "impossible_pattern" in task_id: scaling_columns[task_id] = "sequence_length"
            elif "symbol_guessing" in task_id: scaling_columns[task_id] = "examples_per_prompt"
            elif "number_sorting" in task_id: scaling_columns[task_id] = "actual_decimal_ratio"
            elif "regression" in task_id: scaling_columns[task_id] = "examples_per_prompt"
            elif "native_language_prediction" in task_id: scaling_columns[task_id] = "examples_per_prompt"
            elif "wordle" in task_id: scaling_columns[task_id] = "correct_letters"
            elif "zebra_puzzles" in task_id: scaling_columns[task_id] = "grid_size"
            elif "scheduling" in task_id: scaling_columns[task_id] = "target_clue_count"
            elif "smith_jones_robinson" in task_id: scaling_columns[task_id] = "target_clue_count"
            elif "multiplication" in task_id: scaling_columns[task_id] = "digit_count"
            elif "pigeonhole" in task_id: scaling_columns[task_id] = "num_items"
            elif "n_queens" in task_id: scaling_columns[task_id] = "board_size"
            # Add more auto-detection if needed

    df_enriched = _add_scaling_data_to_results(
        df_enriched, task_loader=task_loader, scaling_columns=scaling_columns
    )

    # --- X-axis setup ---
    x_col_for_grouping = "reasoning_budget" # Always group by budget for x-points
    x_label_final = ""
    if x_axis_type == "tokens":
        if "reasoning_content_tokens" not in df_enriched.columns:
            df_enriched["reasoning_content_tokens"] = df_enriched.apply(_safe_count_tokens, axis=1)
        x_col_for_mean_calc = "reasoning_content_tokens"
        x_label_final = "Avg Reasoning Tokens"
        df_enriched = df_enriched.dropna(subset=[x_col_for_mean_calc])
    elif x_axis_type == "budget":
        x_label_final = "Reasoning Budget (Tokens)"
        df_enriched = df_enriched.dropna(subset=[x_col_for_grouping]) # Ensure budget is not NaN
    else:
        raise ValueError(f"Invalid x_axis_type: {x_axis_type}")

    df_enriched[x_col_for_grouping] = pd.to_numeric(df_enriched[x_col_for_grouping], errors='coerce')
    df_enriched = df_enriched.dropna(subset=[x_col_for_grouping])


    # --- Y-axis and plot type setup ---
    y_data_col = ""
    y_label_final = ""
    y_lim_final = None

    if plot_type == "accuracy":
        y_data_col = "correct"
        y_label_final = "Accuracy"
        y_lim_final = (0, 1.05)
        df_enriched[y_data_col] = pd.to_numeric(df_enriched[y_data_col], errors='coerce')
    elif plot_type == "cost":
        y_data_col = "cost"
        y_label_final = "Mean Cost ($)"
        df_enriched[y_data_col] = pd.to_numeric(df_enriched[y_data_col], errors='coerce')
    elif plot_type == "latency":
        y_data_col = "latency"
        y_label_final = "Mean Latency (s)"
        df_enriched[y_data_col] = pd.to_numeric(df_enriched[y_data_col], errors='coerce')
    elif plot_type == "mse":
        y_data_col = "squared_error"
        y_label_final = "Negative RMSE"
        df_enriched[y_data_col] = pd.to_numeric(df_enriched[y_data_col], errors='coerce')
    elif plot_type == "relative_error":
        y_data_col = "relative_error"
        y_label_final = "Relative Error"
        df_enriched[y_data_col] = pd.to_numeric(df_enriched[y_data_col], errors='coerce')
    else:
        raise ValueError(f"Invalid plot_type: {plot_type}")

    df_enriched = df_enriched.dropna(subset=[y_data_col])

    figures = {}

    for task_idx, task_id in enumerate(tasks):
        task_df = df_enriched[df_enriched["task_id"] == task_id]
        if task_df.empty:
            logger.warning(f"No data for task {task_id} after initial filtering.")
            continue

        num_task_models = task_df["model"].nunique()
        if num_task_models == 0:
            logger.warning(f"No models with data for task {task_id}.")
            continue

        current_models_for_task = task_df["model"].unique().tolist()

        cols = min(3, num_task_models)
        rows = math.ceil(num_task_models / cols)
        fig, fig_axes_array = plt.subplots(
            rows, cols, figsize=(4 * cols, 3.5 * rows), sharey=False, layout="compressed" # Adjusted figsize
        )
        axes_list = np.array(fig_axes_array).reshape(-1)

        norm = None
        scaling_col_this_task = scaling_columns.get(task_id, None)
        has_scaling_this_task = scaling_col_this_task is not None and scaling_col_this_task in task_df.columns

        unique_scaling_values = []
        cmap = sns.color_palette("crest", as_cmap=True)

        if has_scaling_this_task:
            unique_scaling_values = sorted(task_df[scaling_col_this_task].dropna().unique())
            if len(unique_scaling_values) > 1:
                norm = mcolors.Normalize(vmin=min(unique_scaling_values), vmax=max(unique_scaling_values))
            else: # Only one or zero unique scaling values
                has_scaling_this_task = False

        num_models_plotted_for_task = 0
        for model_idx, model_name in enumerate(current_models_for_task):
            ax = axes_list[model_idx]
            model_specific_df = task_df[task_df["model"] == model_name]

            if model_specific_df.empty:
                ax.text(0.5, 0.5, f"No data for {model_to_model_name.get(model_name, model_name)}", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{model_to_model_name.get(model_name, model_name)}", fontsize=16)
                continue

            # Determine groups to plot (either different scaling values or one 'all_data' group)
            data_groups_to_plot = [] # List of (label_for_line, df_for_line, color_for_line)
            if has_scaling_this_task and norm: # norm implies multiple unique_scaling_values
                for scaling_val in unique_scaling_values:
                    df_for_val = model_specific_df[model_specific_df[scaling_col_this_task] == scaling_val]
                    if not df_for_val.empty:
                        data_groups_to_plot.append((str(scaling_val), df_for_val, cmap(norm(scaling_val))))
            else:
                if not model_specific_df.empty:
                    # Use model_id_to_color for this model's line
                    model_specific_color = model_id_to_color.get(model_name, None) # Get custom color or MPL default
                    data_groups_to_plot.append((model_name, model_specific_df, model_specific_color))

            any_line_plotted_for_model = False
            for line_label, line_df, line_color_from_group_setup in data_groups_to_plot: # line_color_from_group_setup now holds the correct color
                grouped_by_budget = line_df.groupby(x_col_for_grouping)

                plot_x_values = []
                plot_y_values = []
                plot_y_errors = []

                for budget_value, budget_group_df in grouped_by_budget:
                    if len(budget_group_df) < min_samples_per_point:
                        continue

                    # Calculate X value
                    current_x_val = 0
                    if x_axis_type == "tokens":
                        current_x_val = budget_group_df[x_col_for_mean_calc].mean()
                    else: # x_axis_type == "budget"
                        current_x_val = budget_value

                    if pd.isna(current_x_val): continue

                    # Calculate Y value and error
                    current_y_val = 0
                    current_y_err = 0 # Keep original variable name
                    metric_data = budget_group_df[y_data_col].astype(float).dropna()

                    if len(metric_data) < min_samples_per_point : continue # Check again after dropna for metric_data

                    count = len(metric_data)

                    if plot_type == "accuracy":
                        current_y_val = metric_data.mean()
                        if count > 0:
                            std_dev = metric_data.std()
                            current_y_err = 1.96 * std_dev / np.sqrt(count) if count > 0 else 0
                    elif plot_type == "mse":
                        instance_rmses = np.sqrt(metric_data) # metric_data is 'squared_error'
                        instance_rmses = np.clip(instance_rmses, a_min=None, a_max=1000) # Clip from reference
                        current_y_val = -instance_rmses.mean() # Negative Mean RMSE
                        if count > 0:
                            std_rmse = instance_rmses.std()
                            current_y_err = std_rmse / np.sqrt(count) if count > 0 else 0
                    elif plot_type == "relative_error":
                        current_y_val = -metric_data.mean() # Negative Mean Relative Error
                        if count > 0:
                            current_y_err = metric_data.std() / np.sqrt(count) if count > 0 else 0
                    elif plot_type == "cost":
                        current_y_val = metric_data.mean()
                        if count > 0:
                            current_y_err = metric_data.std() / np.sqrt(count) if count > 0 else 0
                    elif plot_type == "latency":
                        current_y_val = metric_data.mean()
                        if count > 0:
                            current_y_err = metric_data.std() / np.sqrt(count) if count > 0 else 0

                    if pd.isna(current_y_val): 
                        logger.warning(f"NaN y-value for model {model_name}, task {task_id}, budget {budget_value}. Skipping point.")
                        continue

                    plot_x_values.append(current_x_val)
                    plot_y_values.append(current_y_val)
                    plot_y_errors.append(current_y_err)

                if plot_x_values:
                    # Sort by x_values before plotting to ensure lines are drawn correctly
                    sorted_indices = np.argsort(plot_x_values)
                    x_sorted = np.array(plot_x_values)[sorted_indices]
                    y_sorted = np.array(plot_y_values)[sorted_indices]
                    y_err_sorted = np.array(plot_y_errors)[sorted_indices]

                    final_label = line_label if has_scaling_this_task and norm else None # Only label scaling lines

                    ax.errorbar(
                        x_sorted, y_sorted, yerr=y_err_sorted,
                        fmt="o-", capsize=3, elinewidth=1, markersize=5,
                        color=line_color_from_group_setup, # Use the color determined when setting up data_groups_to_plot
                        label=final_label
                    )
                    any_line_plotted_for_model = True

            if any_line_plotted_for_model:
                num_models_plotted_for_task +=1

            # Axis settings from reference script
            if x_axis_type == "tokens" or x_axis_type == "budget": # Apply consistently
                # Model specific scaling from reference script
                if "claude" in model_name.lower(): # More robust check
                    ax.set_xscale("symlog", base=10, linthresh=1000, linscale=0.2)
                elif "deepseek" in model_name.lower() or "qwen" in model_name.lower():
                    ax.set_xscale("symlog", base=10, linthresh=1000, linscale=0.1)
                else:
                    ax.set_xscale("log") # Default from reference

            ax.set_xlabel(x_label_final, fontsize=16)
            if model_idx % cols == 0: # Only for leftmost plots
                 ax.set_ylabel(y_label_final, fontsize=16)
            # if y_lim_final:
            #     ax.set_ylim(y_lim_final)

            # Improved x-axis tick handling to prevent overlapping
            ax.tick_params(axis="x", rotation=45, labelsize=9, which='both')
            ax.tick_params(axis="y", labelsize=10)
            
            # Format x-axis ticks to prevent overcrowding
            if x_axis_type == "tokens" or x_axis_type == "budget":
                # Only limit ticks for linear scales, log scales handle this automatically
                if ax.get_xscale() == 'linear':
                    # Limit number of x-axis ticks to prevent overcrowding
                    ax.locator_params(axis='x', nbins=6)
                    # Use scientific notation for large numbers to save space (only for linear scales)
                    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,3))
            
            ax.set_title(f"{model_to_model_name.get(model_name, model_name)}", fontsize=16)

        # Hide unused subplots
        for k_ax in range(num_models_plotted_for_task, len(axes_list)):
            axes_list[k_ax].set_visible(False)

        # fig.suptitle(f"{task_to_name.get(task_id.replace('-', '_'), task_id)}", fontsize=16, y=1.04, weight="bold")
        # New title logic with wrapping
        raw_title = task_to_name.get(task_id.replace('-', '_'), None)
        if raw_title is None:
            raw_title = task_id.replace("-", " ")
            # Convert to title case
            raw_title = raw_title.title()
        # Wrap title if it's too long
        wrapped_title = "\n".join(textwrap.wrap(raw_title, width=40)) # Adjust width as needed
        fig.suptitle(wrapped_title, fontsize=16, y=1.02, weight="bold")
        # fig.suptitle(wrapped_title, fontsize=16, y=1.10, weight="bold") # Increased y to prevent overlap

        if has_scaling_this_task and norm and num_models_plotted_for_task > 0:
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

            # Attempt to add colorbar to the right of the grid of subplots
            # This might need adjustment based on the exact layout and number of rows/cols
            try:
                # Use tight_layout to avoid layout engine conflicts
                plt.tight_layout()
                
                # Position colorbar to the right of all subplots
                cbar_ax_list = [ax for i, ax in enumerate(axes_list) if i < num_models_plotted_for_task and (i % cols == cols -1 or i == num_models_plotted_for_task -1) and ax.get_visible()]
                if cbar_ax_list:
                    cbar = fig.colorbar(sm, ax=cbar_ax_list, orientation='vertical', pad=0.02, aspect=30, shrink=0.8, label=scaling_col_this_task.replace('_', ' ').title() if scaling_col_this_task else '')
                else: # Fallback if specific axes selection is tricky
                    cbar = fig.colorbar(sm, ax=axes_list[:num_models_plotted_for_task], orientation='vertical', pad=0.1, aspect=20, shrink=0.7, label=scaling_col_this_task.replace('_', ' ').title() if scaling_col_this_task else '')

            except Exception as e:
                 logger.warning(f"Could not create colorbar for task {task_id}: {e}")
                 # Try a simpler colorbar approach
                 try:
                     plt.tight_layout()
                     cbar = plt.colorbar(sm, ax=fig.get_axes(), orientation='vertical', label=scaling_col_this_task.replace('_', ' ').title() if scaling_col_this_task else '')
                 except Exception as e2:
                     logger.warning(f"Fallback colorbar also failed for task {task_id}: {e2}")

        if num_models_plotted_for_task > 0 : # Only save if something was plotted
            figures[f"{task_id}_{x_axis_type}_{plot_type}"] = fig
            if output_dir is not None:
                output_dir.mkdir(exist_ok=True, parents=True)
                plot_filename = f"{task_id}_{x_axis_type}_{plot_type}.pdf" # Save as PDF
                try:
                    # Adjust layout to prevent title overlap and label cutoff
                    plt.tight_layout(rect=[0, 0.05, 1, 0.90])  # Leave space for title and bottom labels
                    fig.savefig(output_dir / plot_filename, dpi=300, bbox_inches='tight')
                    logger.info(f"Saved plot to {output_dir / plot_filename}")
                except Exception as e:
                    logger.error(f"Failed to save plot {output_dir / plot_filename}: {e}")
            plt.close(fig) # Close figure to free memory
        else:
            plt.close(fig) # Close even if nothing was plotted to free the figure object
            logger.warning(f"No data plotted for task {task_id}; plot not saved.")

    return figures


def plot_token_scaling_curves_single_plot(
    df: pd.DataFrame,
    task_loader: TaskLoader,
    models: List[str] = None,
    tasks: List[str] = None,
    output_dir: Optional[Path] = None,
    plot_type: str = "accuracy",
    x_axis_type: str = "tokens",  # "tokens" or "budget"
    min_samples_per_point: int = 5,  # Minimum samples required for a point
    show_legend: bool = False,  # Whether to show legend on individual plots
) -> Dict[str, plt.Figure]:
    """
    Plot token scaling curves with all models on a single plot (no scaling data).
    Each model gets a different colored line for easy comparison.
    Now properly handles reasoning tokens = 0 by creating separate buckets.
    
    Args:
        df: DataFrame with results
        task_loader: TaskLoader instance for getting task names
        models: List of model IDs to include (default: all)
        tasks: List of task IDs to include (default: all)
        output_dir: Directory to save plots (default: don't save)
        plot_type: Type of metric ("accuracy", "cost", "latency", "mse", "relative_error")
        x_axis_type: Variable for x-axis ("tokens" or "budget")
        min_samples_per_point: Minimum samples required for a point
        show_legend: Whether to show legend on individual plots (default: False)
        
    Returns:
        Dictionary mapping plot names to Figure objects
    """
    set_plotting_style()

    # Color palette for models
    color_palette_sns = sns.color_palette() 
    model_id_to_color = {
        "claude-sonnet-4-20250514": color_palette_sns[1],
        "claude-opus-4-20250514": color_palette_sns[7],
        "claude-3-7-sonnet-20250219": color_palette_sns[8],
        "claude-3-7-sonnet-20250219_natural_overthinking": color_palette_sns[7],
        "claude-3-7-sonnet-20250219_not_use_all_budget": color_palette_sns[8],
        "o3-2025-04-16": color_palette_sns[2],
        "o3-mini-2025-01-31": color_palette_sns[9],
        "O4-mini": color_palette_sns[3],
        "o4-mini-2025-04-16": color_palette_sns[3],
        "deepseek-reasoner": color_palette_sns[0],
        "Qwen3-8B": color_palette_sns[4],
        "Qwen3-14B": color_palette_sns[5],
        "Qwen3-32B": color_palette_sns[6],
    }

    # Mappings for prettier names
    model_to_model_name = MODEL_TO_PRETTY_NAME
    
    # Create task name mapping from TaskLoader
    task_to_name = {}
    if tasks:
        for task_id in tasks:
            task_to_name[task_id] = get_task_pretty_name(task_id, task_loader)

    if models is None:
        models = df["model"].unique().tolist()
    if tasks is None:
        tasks = df["task_id"].unique().tolist()

    df_enriched = df.copy()

    # --- X-axis setup ---
    x_col_for_grouping = "reasoning_budget" 
    x_label_final = ""
    if x_axis_type == "tokens":
        if "reasoning_content_tokens" not in df_enriched.columns:
            df_enriched["reasoning_content_tokens"] = df_enriched.apply(_safe_count_tokens, axis=1)
        x_col_for_mean_calc = "reasoning_content_tokens"
        x_label_final = "Reasoning Tokens"
        df_enriched = df_enriched.dropna(subset=[x_col_for_mean_calc])
    elif x_axis_type == "budget":
        x_label_final = "Reasoning Budget (Tokens)"
        df_enriched = df_enriched.dropna(subset=[x_col_for_grouping])
    else:
        raise ValueError(f"Invalid x_axis_type: {x_axis_type}")

    df_enriched[x_col_for_grouping] = pd.to_numeric(df_enriched[x_col_for_grouping], errors='coerce')
    df_enriched = df_enriched.dropna(subset=[x_col_for_grouping])

    # --- Y-axis and plot type setup ---
    y_data_col = ""
    y_label_final = ""

    if plot_type == "accuracy":
        y_data_col = "correct"
        y_label_final = "Accuracy"
        df_enriched[y_data_col] = pd.to_numeric(df_enriched[y_data_col], errors='coerce')
    elif plot_type == "cost":
        y_data_col = "cost"
        y_label_final = "Mean Cost ($)"
        df_enriched[y_data_col] = pd.to_numeric(df_enriched[y_data_col], errors='coerce')
    elif plot_type == "latency":
        y_data_col = "latency"
        y_label_final = "Mean Latency (s)"
        df_enriched[y_data_col] = pd.to_numeric(df_enriched[y_data_col], errors='coerce')
    elif plot_type == "mse":
        y_data_col = "squared_error"
        y_label_final = "Negative RMSE"
        df_enriched[y_data_col] = pd.to_numeric(df_enriched[y_data_col], errors='coerce')
    elif plot_type == "relative_error":
        y_data_col = "relative_error"
        y_label_final = "Relative Error"
        df_enriched[y_data_col] = pd.to_numeric(df_enriched[y_data_col], errors='coerce')
    else:
        raise ValueError(f"Invalid plot_type: {plot_type}")

    df_enriched = df_enriched.dropna(subset=[y_data_col])

    figures = {}

    for task_id in tasks:
        task_df = df_enriched[df_enriched["task_id"] == task_id]
        if task_df.empty:
            logger.warning(f"No data for task {task_id} after initial filtering.")
            continue

        # Create single plot for all models
        fig, ax = plt.subplots(figsize=(3, 3))
        
        models_plotted = 0
        
        for model_name in models:
            model_specific_df = task_df[task_df["model"] == model_name]
            
            if model_specific_df.empty:
                logger.debug(f"No data for model {model_name} in task {task_id}")
                continue

            # Prepare data for processing - similar to user's example
            plot_x_values = []
            plot_y_values = []
            plot_y_errors = []

            # Handle zero tokens explicitly (similar to user's zero cost handling)
            if x_axis_type == "tokens":
                # First, handle zero reasoning tokens as a separate point
                zero_token_df = model_specific_df[model_specific_df[x_col_for_mean_calc] == 0]
                if len(zero_token_df) >= min_samples_per_point:
                    metric_data_zero = zero_token_df[y_data_col].astype(float).dropna()
                    if len(metric_data_zero) >= min_samples_per_point:
                        # Additional check: ensure this is genuine zero reasoning token data
                        # Check if these are truly zero reasoning tokens or just calculation artifacts
                        genuine_zero_tokens = True
                        
                        # For reasoning models like o3/o4, they typically always have reasoning
                        # Only include zero points if they genuinely have reasoning_budget = 0 entries
                        if model_name in OPENAI_REASONING_MODELS:
                            # For OpenAI reasoning models, check if any of the zero token entries
                            # actually have reasoning_budget = 0 (not just calculated as 0 tokens)
                            zero_budget_count = len(zero_token_df[zero_token_df[x_col_for_grouping] == 0])
                            if zero_budget_count == 0:
                                genuine_zero_tokens = False
                                logger.debug(f"Skipping zero token point for {model_name} - no genuine zero budget entries")
                        
                        if genuine_zero_tokens:
                            current_x_val = 0.0
                            count = len(metric_data_zero)
                            
                            if plot_type == "accuracy":
                                current_y_val = metric_data_zero.mean()
                                std_dev = metric_data_zero.std()
                                current_y_err = 1.96 * std_dev / np.sqrt(count) if count > 0 else 0
                            elif plot_type == "mse":
                                instance_rmses = np.sqrt(metric_data_zero)
                                instance_rmses = np.clip(instance_rmses, a_min=None, a_max=1000)
                                current_y_val = -instance_rmses.mean()
                                std_rmse = instance_rmses.std()
                                current_y_err = std_rmse / np.sqrt(count) if count > 0 else 0
                            elif plot_type == "relative_error":
                                current_y_val = -metric_data_zero.mean()
                                current_y_err = metric_data_zero.std() / np.sqrt(count) if count > 0 else 0
                            elif plot_type in ["cost", "latency"]:
                                current_y_val = metric_data_zero.mean()
                                current_y_err = metric_data_zero.std() / np.sqrt(count) if count > 0 else 0

                            if not pd.isna(current_y_val):
                                plot_x_values.append(current_x_val)
                                plot_y_values.append(current_y_val)
                                plot_y_errors.append(current_y_err)
                                logger.debug(f"Added zero token point for {model_name} with {count} samples")

                # Then process non-zero token data by budget groups
                non_zero_df = model_specific_df[model_specific_df[x_col_for_mean_calc] > 0]
            else:
                # For budget-based x-axis, use all data
                non_zero_df = model_specific_df.copy()

            # Process remaining data by budget groups
            if not non_zero_df.empty:
                grouped_by_budget = non_zero_df.groupby(x_col_for_grouping)
                
                for budget_value, budget_group_df in grouped_by_budget:
                    if len(budget_group_df) < min_samples_per_point:
                        continue

                    # Calculate X value
                    current_x_val = 0
                    if x_axis_type == "tokens":
                        # Only include non-zero tokens in mean calculation
                        tokens_in_group = budget_group_df[x_col_for_mean_calc].astype(float)
                        tokens_in_group = tokens_in_group[tokens_in_group > 0]  # Exclude zeros from mean
                        if len(tokens_in_group) < min_samples_per_point:
                            continue
                        current_x_val = tokens_in_group.mean()
                    else:  # x_axis_type == "budget"
                        current_x_val = budget_value

                    if pd.isna(current_x_val):
                        continue

                    # Calculate Y value and error
                    metric_data = budget_group_df[y_data_col].astype(float).dropna()

                    if len(metric_data) < min_samples_per_point:
                        continue

                    count = len(metric_data)

                    if plot_type == "accuracy":
                        current_y_val = metric_data.mean()
                        if count > 0:
                            std_dev = metric_data.std()
                            current_y_err = 1.96 * std_dev / np.sqrt(count) if count > 0 else 0
                    elif plot_type == "mse":
                        instance_rmses = np.sqrt(metric_data)
                        instance_rmses = np.clip(instance_rmses, a_min=None, a_max=1000)
                        current_y_val = -instance_rmses.mean()
                        if count > 0:
                            std_rmse = instance_rmses.std()
                            current_y_err = std_rmse / np.sqrt(count) if count > 0 else 0
                    elif plot_type == "relative_error":
                        current_y_val = -metric_data.mean()
                        if count > 0:
                            current_y_err = metric_data.std() / np.sqrt(count) if count > 0 else 0
                    elif plot_type in ["cost", "latency"]:
                        current_y_val = metric_data.mean()
                        if count > 0:
                            current_y_err = metric_data.std() / np.sqrt(count) if count > 0 else 0

                    if pd.isna(current_y_val):
                        logger.warning(f"NaN y-value for model {model_name}, task {task_id}, budget {budget_value}. Skipping point.")
                        continue

                    plot_x_values.append(current_x_val)
                    plot_y_values.append(current_y_val)
                    plot_y_errors.append(current_y_err)

            if plot_x_values:
                # Sort by x_values before plotting
                sorted_indices = np.argsort(plot_x_values)
                x_sorted = np.array(plot_x_values)[sorted_indices]
                y_sorted = np.array(plot_y_values)[sorted_indices]
                y_err_sorted = np.array(plot_y_errors)[sorted_indices]

                # Get model color and pretty name
                model_color = model_id_to_color.get(model_name, None)
                model_pretty_name = model_to_model_name.get(model_name, model_name)

                ax.errorbar(
                    x_sorted, y_sorted, yerr=y_err_sorted,
                    fmt="o-", capsize=3, elinewidth=1, markersize=5,
                    color=model_color,
                    label=model_pretty_name
                )
                models_plotted += 1

        if models_plotted == 0:
            plt.close(fig)
            logger.warning(f"No models plotted for task {task_id}")
            continue

        # Set axis scale - use symlog to handle zero values properly (like in user's example)
        if x_axis_type == "tokens":
            # Use symlog scale to handle both zero and non-zero values
            ax.set_xscale("symlog", linthresh=100, linscale=0.2)
        elif x_axis_type == "budget":
            # For budget, can still use log if all values are positive, otherwise symlog
            min_x = ax.get_xlim()[0]
            if min_x <= 0:
                ax.set_xscale("symlog", linthresh=100, linscale=0.2)
            else:
                ax.set_xscale("log")

        # Format axes
        ax.set_xlabel(x_label_final, fontsize=14)
        ax.set_ylabel(y_label_final, fontsize=14)
        
        # Improved x-axis tick handling
        ax.tick_params(axis="x", rotation=45, labelsize=9, which='both')
        ax.tick_params(axis="y", labelsize=10)
        
        # Add legend only if requested
        if show_legend:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set title
        task_pretty_name = get_task_pretty_name(task_id, task_loader)
        ax.set_title(f"{task_pretty_name}", fontsize=16)

        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Adjust layout to accommodate legend
        plt.tight_layout()
        
        # Save figure
        if models_plotted > 0:
            figures[f"{task_id}_{x_axis_type}_{plot_type}"] = fig
            if output_dir is not None:
                output_dir.mkdir(exist_ok=True, parents=True)
                plot_filename = f"{task_id}_{plot_type}_single_plot_{x_axis_type}.pdf"
                try:
                    fig.savefig(output_dir / plot_filename, dpi=300, bbox_inches='tight')
                    logger.info(f"Saved single plot: {output_dir / plot_filename}")
                except Exception as e:
                    logger.error(f"Failed to save plot {output_dir / plot_filename}: {e}")
            plt.close(fig)
        else:
            plt.close(fig)
            logger.warning(f"No data plotted for task {task_id}; plot not saved.")

    return figures


# Helper function to get number of classes for a task
def get_num_classes(df: pd.DataFrame, task_id: str) -> Optional[int]:
    """Get the number of classes for a given task from the DataFrame."""
    # If there is no 'classes' column, this is likely an open-ended task; skip random chance line
    if 'classes' not in df.columns:
        return None
    task_subset_df = df[df['task_id'] == task_id]
    if not task_subset_df.empty:
        # Try getting the first non-null 'classes' value
        first_valid_instance = task_subset_df[task_subset_df['classes'].notna()].iloc[0] if task_subset_df['classes'].notna().any() else None

        if first_valid_instance is not None and 'classes' in first_valid_instance:
            classes_val = first_valid_instance['classes']
            # Handle cases where 'classes' might be stored as a string representation of a list
            if isinstance(classes_val, str):
                try:
                    import ast
                    classes_val = ast.literal_eval(classes_val)
                except (ValueError, SyntaxError, TypeError):
                    logger.warning(f"Could not parse 'classes' string for task {task_id}: {str(classes_val)[:100]}")
                    return None
            if isinstance(classes_val, list):
                return len(classes_val)
            else:
                logger.warning(f"'classes' column for task {task_id} is not a list or parsable string: {type(classes_val)}")
                return None
        else:
            logger.warning(f"Could not find valid 'classes' entry for task {task_id}")
            return None
    else:
        logger.warning(f"No data found for task {task_id} to determine number of classes.")
        return None


def create_model_legend(
    models: List[str] = None,
    output_dir: Optional[Path] = None,
    legend_filename: str = "model_legend.pdf",
) -> plt.Figure:
    """
    Create a standalone legend figure showing model names and their corresponding colors.
    This can be used as a reference for all plots that use the same color scheme.
    
    Args:
        models: List of model IDs to include in legend (default: all models in color mapping)
        output_dir: Directory to save legend PDF (default: don't save)
        legend_filename: Filename for the legend PDF
        
    Returns:
        Figure object containing the legend
    """
    set_plotting_style()
    
    # Color palette for models (same as in plotting function)
    color_palette_sns = sns.color_palette() 
    model_id_to_color = {
        "claude-sonnet-4-20250514": color_palette_sns[1],
        "claude-opus-4-20250514": color_palette_sns[7],
        "claude-3-7-sonnet-20250219": color_palette_sns[8],
        "claude-3-7-sonnet-20250219_natural_overthinking": color_palette_sns[7],
        "claude-3-7-sonnet-20250219_not_use_all_budget": color_palette_sns[8],
        "o3-2025-04-16": color_palette_sns[2],
        "o3-mini-2025-01-31": color_palette_sns[9],
        "O4-mini": color_palette_sns[3],
        "o4-mini-2025-04-16": color_palette_sns[3],
        "deepseek-reasoner": color_palette_sns[0],
        "Qwen3-8B": color_palette_sns[4],
        "Qwen3-14B": color_palette_sns[5],
        "Qwen3-32B": color_palette_sns[6],
    }
    
    # Mappings for prettier names
    model_to_model_name = MODEL_TO_PRETTY_NAME
    
    # Use provided models or all models in color mapping
    if models is None:
        models = list(model_id_to_color.keys())
    
    # Filter to only models that have colors defined
    models = [m for m in models if m in model_id_to_color]
    
    if not models:
        logger.warning("No models with defined colors found for legend.")
        return None
    
    # Determine optimal layout - prioritize horizontal expansion
    num_models = len(models)
    if num_models <= 3:
        ncol = num_models
        nrow = 1
    elif num_models <= 6:
        ncol = 3
        nrow = 2
    elif num_models <= 9:
        ncol = 3
        nrow = 3
    else:
        # For larger numbers, use up to 4 columns
        ncol = min(4, num_models)
        nrow = math.ceil(num_models / ncol)
    
    # Create figure for legend only - adjust size based on layout
    fig_width = max(3, ncol * 2.5)  # Wider for horizontal layout
    fig_height = max(1.5, nrow * 0.8 + 1)  # Height based on rows
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create legend elements
    legend_elements = []
    for model_id in models:
        model_color = model_id_to_color.get(model_id)
        model_pretty_name = model_to_model_name.get(model_id, model_id)
        
        # Create a line element for the legend
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color=model_color, 
                      linestyle='-', markersize=8, linewidth=2,
                      label=model_pretty_name)
        )
    
    # Hide the axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Create the legend with horizontal priority
    legend = ax.legend(
        handles=legend_elements,
        loc='center',
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=14,
        title="Models",
        title_fontsize=16,
        ncol=ncol  # Number of columns for horizontal expansion
    )
    
    # Style the legend
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output_dir is specified
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)
        fig.savefig(output_dir / legend_filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved model legend to {output_dir / legend_filename}")
    
    return fig
