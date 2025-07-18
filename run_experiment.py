#!/usr/bin/env python
"""
Script to run inverse scaling experiments with various configurations.
"""
# Standard Libraries
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
import os
from typing import List
import yaml

# Hydra and OmegaConf
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from hydra.core.hydra_config import HydraConfig

# Weights & Biases
import wandb

# Local imports
from safetytooling.utils import utils
from src.batch_model_interface import BatchModelInterface
from src.evaluator import Evaluator
from src.model_interface import ModelInterface
from src.vllm_model_interface import VLLMModelInterface, cleanup_all_vllm_interfaces
from src.results_manager import ResultsManager
from src.task_loader import TaskLoader
from src.config_utils import register_config_schemas

# Analysis/plotting imports
from src.utils.analysis import analyze_inverse_scaling
from src.utils.plotting import (
    plot_token_correlations,
    plot_budget_length_boxplot,
    plot_token_scaling_curves_improved,
    plot_token_slopes,
)

# Create logs directory for Hydra if it doesn't exist.
# This needs to be done before Hydra initializes logging, as Hydra's FileHandler
# can't create directories and will raise a FileNotFoundError.
logs_dir = Path(".logs")
logs_dir.mkdir(exist_ok=True)

# Use Hydra's logger (configured in config.yaml)
log = logging.getLogger(__name__)


def load_yaml_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Define an async helper function to manage async operations
async def _run_async_parts(
    cfg: DictConfig,
    evaluator: Evaluator,
    results_manager: ResultsManager,
    models_to_run: List,
    tasks_to_run: List,
    reasoning_budgets: List,
    validation_mode: bool,
    validation_samples: int,
    validation_runs: int,
    validation_seed: int,
):
    log.info(
        f"Starting evaluation (async): {len(models_to_run)} models, {len(tasks_to_run)} tasks, {len(reasoning_budgets)} budgets"
    )
    
    # Get any ICL-related configuration from the evaluation settings
    icl_config = None
    if "shot_count" in cfg.evaluation and cfg.evaluation.shot_count is not None:
        icl_config = {"enabled": True, "num_examples": cfg.evaluation.shot_count}
    
    # Get seeds from config if provided
    seeds = cfg.evaluation.get('seeds', None) if hasattr(cfg.evaluation, 'seeds') else None
    
    # Pass validation parameters down to run_evaluations
    summary, loaded_results_list = await evaluator.run_evaluations(
        models_to_run,
        tasks_to_run,
        reasoning_budgets,
        validation_mode=validation_mode,
        validation_samples=validation_samples,
        validation_runs=validation_runs,
        validation_seed=validation_seed,
        icl_config=icl_config,
        seeds=seeds,
    )
    log.info(f"Evaluation completed (async).")

    log.info("Generating summary (async)...")
    return summary, loaded_results_list


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def run_experiment_hydra(cfg: DictConfig) -> None:
    """Run the experiment using Hydra configuration."""
    # Register config schemas before loading
    register_config_schemas()

    # --- WandB Initialization ---
    if cfg.wandb.enabled:
        try:
            config_dict = OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            )
            entity = cfg.wandb.entity or os.environ.get("WANDB_ENTITY")
            run = wandb.init(
                project=cfg.wandb.project,
                entity=entity,
                config=config_dict,
                name=cfg.wandb.name,
                group=cfg.wandb.group,
                tags=list(cfg.wandb.tags),
                job_type="evaluation",
                reinit=True,
                save_code=True,
            )
            log.info(f"WandB Run URL: {run.url}")
        except Exception as e:
            log.error(f"Failed to initialize WandB: {e}", exc_info=True)
            log.warning("Proceeding without WandB logging.")
            cfg.wandb.enabled = False
    else:
        log.info("WandB logging disabled.")
        run = None

    # --- Configuration Processing ---
    log.info("Starting experiment run...")
    log.info(f"Hydra output directory: {HydraConfig.get().runtime.output_dir}")
    log.info(f"Experiment configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Load model and task configurations from YAML files
    # Look for models in the directory specified by paths.models_config
    models_config_path = Path(cfg.paths.models_config)
    if not models_config_path.is_dir():
        # For backward compatibility, try loading as a file
        log.warning(f"{models_config_path} is not a directory, trying to load as a file")
        models_config = load_yaml_config(models_config_path)
    else:
        # Load from all YAML files in the models_config directory
        models_config = {}
        for model_file in models_config_path.glob("*.yaml"):
            model_yaml = load_yaml_config(model_file)
            if isinstance(model_yaml, dict) and "id" in model_yaml:
                # Extract from @package model structure
                models_config[model_yaml["id"]] = model_yaml
            else:
                log.warning(f"Skipping {model_file} as it doesn't contain a valid model configuration")
    
    # Similar approach for tasks
    tasks_config_path = Path(cfg.paths.tasks_config)
    if not tasks_config_path.is_dir():
        # For backward compatibility, try loading as a file
        log.warning(f"{tasks_config_path} is not a directory, trying to load as a file")
        tasks_config = load_yaml_config(tasks_config_path)
    else:
        # Load task definitions from all YAML files in the tasks_config directory
        tasks_config = {}
        for task_file in tasks_config_path.glob("*.yaml"):
            task_yaml = load_yaml_config(task_file)
            if isinstance(task_yaml, dict) and "task_definitions" in task_yaml:
                # Extract from @package task structure - combine task definitions
                tasks_config.update(task_yaml["task_definitions"])
            else:
                log.warning(f"Skipping {task_file} as it doesn't contain valid task definitions")

    # Use the evaluation parameters from the experiment config
    current_eval_cfg = cfg.evaluation
    log.info(f"Using evaluation config from experiment '{cfg.experiment.name}'")

    def parse_list_or_string(config_val, all_keys):
        if config_val is None:
            return list(all_keys)
        elif isinstance(config_val, (list, ListConfig)):
            return list(config_val)
        elif isinstance(config_val, str):
            return [item.strip() for item in config_val.split(",")]
        else:
            raise ValueError(f"Invalid format for models/tasks: {config_val}")

    # Parse models, tasks, budgets from the experiment's evaluation block
    # Always use single model from override - no multi-model support needed
    if hasattr(cfg, 'model') and cfg.model is not None and hasattr(cfg.model, 'id'):
        models_to_run = [cfg.model.id]
        log.info(f"Using model from override: {cfg.model.id}")
        
        # Warn if evaluation.models is also specified (redundant)
        if current_eval_cfg.models is not None and len(current_eval_cfg.models or []) > 0:
            log.warning(f"Both model override ({cfg.model.id}) and evaluation.models ({current_eval_cfg.models}) specified. Using model override. Consider removing evaluation.models.")
    else:
        raise ValueError("No model specified. Please use 'override /model: model_name' in your experiment config defaults.")
    
    tasks_to_run = parse_list_or_string(current_eval_cfg.tasks, tasks_config.keys())
    reasoning_budgets = list(current_eval_cfg.reasoning_budgets)

    # --- Component Initialization ---
    exp_dir = Path(HydraConfig.get().runtime.output_dir)
    log.info(f"Experiment results will be saved in: {exp_dir}")

    # Find the latest previous run directory if it exists
    previous_results_dir = None
    if cfg.experiment.name:
        # Look for previous runs in the results directory
        results_base_dir = Path("results") / cfg.experiment.name
        if results_base_dir.exists():
            # Get all timestamp directories and sort them
            prev_runs = sorted([d for d in results_base_dir.iterdir() if d.is_dir() and d.name != exp_dir.name])
            if prev_runs:
                previous_results_dir = prev_runs[-1]  # Get the most recent run
                log.info(f"Found previous run directory: {previous_results_dir}")

    task_loader = TaskLoader(tasks_config)
    utils.setup_environment(anthropic_tag=cfg.api.anthropic_api_key_tag)
    
    # Determine interface type - check for VLLM models or explicit interface type
    interface_type = getattr(cfg, 'model_interface_type', 'default')
    
    # Auto-detect VLLM if any model has type 'vllm'
    if interface_type == 'default':
        vllm_models = [model_id for model_id, model_cfg in models_config.items() 
                      if model_cfg.get('type') == 'vllm' and model_id in models_to_run]
        if vllm_models:
            interface_type = 'vllm'
            log.info(f"Auto-detected VLLM interface for models: {vllm_models}")
    
    if interface_type == 'vllm':
        model_interface = VLLMModelInterface(
            models_config,
            current_eval_cfg,
            use_cache=not cfg.api.no_cache,
            anthropic_api_key_tag=cfg.api.anthropic_api_key_tag,
            batch_size=current_eval_cfg.batch_size,
        )
    elif cfg.api.use_batch_api:
        model_interface = BatchModelInterface(
            models_config,
            current_eval_cfg,
            use_cache=not cfg.api.no_cache,
            anthropic_api_key_tag=cfg.api.anthropic_api_key_tag,
            batch_size=current_eval_cfg.batch_size,  # Use batch_size from experiment config
        )
    else:
        model_interface = ModelInterface(
            models_config,
            use_cache=not cfg.api.no_cache,
            anthropic_api_key_tag=cfg.api.anthropic_api_key_tag,
            # Note: Non-batch interface might not use batch_size from config anyway
        )
    results_manager = ResultsManager(exp_dir, previous_results_dir)
    evaluator = Evaluator(task_loader, model_interface, results_manager)

    # Pass the correct validation config if needed
    validation_cfg = (
        cfg.validation if hasattr(cfg.experiment, "validation") and cfg.validation is not None
        else cfg.validation
    )

    # --- Run Async Operations ---
    try:
        summary, loaded_results_list = asyncio.run(
            _run_async_parts(
                cfg,  # Pass the whole cfg down if helper needs other parts
                evaluator,
                results_manager,
                models_to_run,
                tasks_to_run,
                reasoning_budgets,
                # Pass validation params explicitly from resolved validation_cfg
                validation_mode=validation_cfg.enabled,
                validation_samples=validation_cfg.samples,
                validation_runs=validation_cfg.runs,
                validation_seed=validation_cfg.seed,
            )
        )
    except Exception as e:
        log.error(f"Error during async execution: {e}", exc_info=True)
        
        # Cleanup model interface even on failure
        if hasattr(model_interface, 'close'):
            try:
                log.info("Cleaning up model interface after async failure...")
                model_interface.close()
            except Exception as cleanup_error:
                log.warning(f"Error during cleanup after async failure: {cleanup_error}")
        
        if cfg.wandb.enabled and run:
            wandb.finish(exit_code=1)
        raise

    # --- Save Summary (Sync) ---
    summary_path = results_manager.save_summary(summary)
    log.info(f"Summary saved to {summary_path}")

    # --- Create and Save DataFrame (Sync) ---
    log.info("Creating DataFrame...")
    df = results_manager.create_dataframe(results_list=loaded_results_list)
    df_path = exp_dir / "results_df.csv"
    df.to_csv(df_path, index=False)
    log.info(f"DataFrame saved to {df_path}")

    # --- WandB Logging (Summary & Artifacts) ---
    if cfg.wandb.enabled and run:
        log.info("Logging summary results to WandB...")
        wandb_summary_log = {}
        wandb_summary_log["total_loaded_evaluations"] = summary.get(
            "total_loaded_evaluations"
        )
        wandb_summary_log["total_filtered_evaluations"] = summary.get(
            "total_filtered_evaluations"
        )
        if "model_stats" in summary:
            for model_id, stats in summary["model_stats"].items():
                safe_model_id = model_id.replace("/", "_").replace(":", "-")
                wandb_summary_log[f"model_{safe_model_id}_accuracy"] = stats.get(
                    "accuracy"
                )
                wandb_summary_log[f"model_{safe_model_id}_total_cost"] = stats.get(
                    "total_cost"
                )
                wandb_summary_log[f"model_{safe_model_id}_total_evaluations"] = (
                    stats.get("total_evaluations")
                )
        wandb.log(wandb_summary_log)
        try:
            log.info("Logging artifacts to WandB...")
            summary_artifact = wandb.Artifact(f"summary-{run.id}", type="summary")
            summary_artifact.add_file(str(summary_path))
            run.log_artifact(summary_artifact)
            results_artifact = wandb.Artifact(f"results_df-{run.id}", type="results")
            results_artifact.add_file(str(df_path))
            run.log_artifact(results_artifact)
        except Exception as e:
            log.error(f"Failed to log artifacts to WandB: {e}", exc_info=True)

    # --- Run Analysis (Sync) ---
    analysis_results_path = None
    if cfg.analysis.run_analysis:
        log.info("Running analysis...")
        analysis_dir = exp_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        plot_dir = analysis_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        try:
            # --- Check available metrics in the DataFrame ---
            available_metrics = []
            if 'correct' in df.columns and df['correct'].notna().any():
                available_metrics.append('accuracy')
            if 'squared_error' in df.columns and df['squared_error'].notna().any():
                available_metrics.append('mse')
            if 'relative_error' in df.columns and df['relative_error'].notna().any():
                available_metrics.append('relative_error')
            log.info(f"Metrics found in results: {available_metrics}")

            # --- Generate plots for each available metric ---
            for metric_type in available_metrics:
                log.info(f"Generating {metric_type.upper()} plots...")
                
                log.info(f"... {metric_type} vs. token scaling curves...")
                plot_token_scaling_curves_improved(
                    df,
                    task_loader,
                    models_to_run,
                    tasks_to_run,
                    plot_dir,
                    plot_type=metric_type, # Use dynamic metric type
                    min_samples_per_point=3,
                )
                
                log.info(f"... {metric_type} token slope...")
                plot_token_slopes(
                    df,
                    models_to_run,
                    tasks_to_run,
                    plot_dir,
                    plot_type=metric_type, # Use dynamic metric type
                )

            # --- Generate metric-independent plots ---
            log.info("Generating budget vs length boxplot...")
            plot_budget_length_boxplot(df, plot_dir, task_loader=task_loader)

            log.info(f"Analysis complete. Plots saved to {plot_dir}")

            # WandB analysis logging block
            if cfg.wandb.enabled and run:
                log.info("Logging analysis artifacts to WandB...")
                try:
                    analysis_artifact = wandb.Artifact(
                        f"analysis-{run.id}", type="analysis"
                    )
                    if analysis_results_path and analysis_results_path.is_file():
                        analysis_artifact.add_file(str(analysis_results_path))
                    # Add all generated plots
                    if plot_dir.is_dir():
                        for plot_file in plot_dir.glob("*.png"):
                            analysis_artifact.add_file(
                                str(plot_file), name=f"plots/{plot_file.name}"
                            )
                    run.log_artifact(analysis_artifact)
                except Exception as e:
                    log.error(
                        f"Failed to log analysis artifacts to WandB: {e}", exc_info=True
                    )

        except Exception as e:
            log.error(f"Error during analysis: {e}", exc_info=True)
            
            # Cleanup model interface even on analysis failure
            if hasattr(model_interface, 'close'):
                try:
                    log.info("Cleaning up model interface after analysis failure...")
                    model_interface.close()
                except Exception as cleanup_error:
                    log.warning(f"Error during cleanup after analysis failure: {cleanup_error}")
            
            if cfg.wandb.enabled and run:
                wandb.finish(exit_code=1)
            raise

    log.info(
        f"Experiment finished successfully. Results saved to {exp_dir}"
    )
    
    # Cleanup model interface if it's a VLLM interface
    if hasattr(model_interface, 'close'):
        try:
            log.info("Cleaning up model interface...")
            model_interface.close()
        except Exception as e:
            log.warning(f"Error during model interface cleanup: {e}")
    
    # Final cleanup of all VLLM interfaces (in case some weren't captured above)
    try:
        log.info("Performing final VLLM cleanup...")
        cleanup_all_vllm_interfaces()
    except Exception as e:
        log.warning(f"Error during final VLLM cleanup: {e}")
    
    if cfg.wandb.enabled and run:
        wandb.finish()


# Keep the __main__ block
if __name__ == "__main__":
    run_experiment_hydra()
