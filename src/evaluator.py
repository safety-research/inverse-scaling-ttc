"""
Evaluator for running inverse scaling evaluations.
"""

import logging
import random
from itertools import groupby
from typing import Any, Dict, List, Optional, Union, Tuple
import asyncio

from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from src.batch_model_interface import BatchModelInterface
from src.model_interface import ModelInterface, console
from src.results_manager import ResultsManager
from src.task_loader import TaskLoader

# Import for VLLM-specific handling
try:
    from src.vllm_model_interface import VLLMModelInterface
except ImportError:
    VLLMModelInterface = None

# Set up rich logging if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
logger = logging.getLogger(__name__)


class Evaluator:
    """Main evaluator for inverse scaling tasks."""

    def __init__(
        self,
        task_loader: TaskLoader,
        model_interface: Union[ModelInterface, BatchModelInterface],
        results_manager: ResultsManager,
    ):
        """
        Initialize the evaluator.

        Args:
            task_loader: TaskLoader instance for loading task data
            model_interface: ModelInterface or BatchModelInterface for interacting with models
            results_manager: ResultsManager for handling results
        """
        self.task_loader = task_loader
        self.model_interface = model_interface
        self.results_manager = results_manager
        self.console = console

        self.console.print("✨ Evaluator initialized successfully", style="green")

    def _display_initial_config(
        self,
        models: List[str],
        tasks: List[str],
        reasoning_budgets: List[int],
        validation_mode: bool,
        validation_samples: int,
        validation_runs: int,
        validation_seed: int,
        icl_config: Dict[str, Any],
    ) -> None:
        """Displays the initial evaluation configuration."""
        config_table = Table(title="📋 Evaluation Configuration", show_header=True)
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="green")
        config_table.add_row("Models", ", ".join(models))
        config_table.add_row("Tasks", ", ".join(tasks))
        config_table.add_row(
            "Reasoning Budgets", ", ".join(map(str, reasoning_budgets))
        )
        if validation_mode:
            config_table.add_row("Validation Mode", "✓")
            config_table.add_row("Validation Samples", str(validation_samples))
            config_table.add_row("Validation Runs", str(validation_runs))
            config_table.add_row("Validation Seed", str(validation_seed))
            self.console.print("🔍 Running in validation mode", style="green")
        if icl_config:
            if icl_config.get("enabled", False):
                config_table.add_row("In-Context Learning (ICL)", "✓")
                config_table.add_row("ICL Examples (k)", str(icl_config.get("num_examples", 0)))
                self.console.print(f"🧠 Using {icl_config.get('num_examples', 0)}-shot In-Context Learning", style="blue")
        else:
            config_table.add_row("In-Context Learning (ICL)", "✗")

        self.console.print(config_table)

    def _generate_evaluation_configs(
        self,
        models: List[str],
        tasks: List[str],
        reasoning_budgets: List[int],
        validation_mode: bool,
        validation_samples: int,
        validation_runs: int,
        validation_seed: int,
        icl_config: Dict[str, Any],
        seeds: Optional[List[int]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
        """Generates the list of evaluation configurations to run, skipping completed ones."""
        if validation_mode:
            random.seed(validation_seed)

        all_eval_configs_to_run = []
        skipped_configs = []
        total_configs = 0

        if icl_config:
            icl_enabled = icl_config.get("enabled", False)
            icl_num_examples = icl_config.get("num_examples", 0)
        else:
            icl_enabled = False
            icl_num_examples = 0

        if icl_enabled:
            icl_random_gen = random.Random(validation_seed + 1)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            config_task = progress.add_task(
                "Generating configurations...", total=len(models) * len(tasks)
            )

            for model_id in models:
                for task_id in tasks:
                    progress.update(
                        config_task,
                        description=f"Loading task {task_id} for model {model_id}",
                    )
                    full_task_data = self.task_loader.get_task_data(task_id, external_progress=progress)
                    if full_task_data is None:
                        logger.error(f"Could not load task data for {task_id}. Skipping.")
                        continue

                    existing_results = self.results_manager.load_raw_results_for_task(model_id, task_id)
                    logger.info(f"Loaded {len(existing_results)} existing results for {model_id}/{task_id}.")

                    task_instances_to_evaluate = self._apply_validation_sampling(
                        full_task_data, validation_mode, validation_samples, validation_runs
                    )

                    for instance_index, instance in enumerate(task_instances_to_evaluate):
                        # Group reasoning budgets and track repetitions for statistical robustness
                        budget_counts = {}
                        global_run_index = 0  # Track run index across all budgets
                        for reasoning_budget in reasoning_budgets:
                            if reasoning_budget not in budget_counts:
                                budget_counts[reasoning_budget] = 0
                            budget_counts[reasoning_budget] += 1
                            
                            # Create unique instance ID for repeated budgets
                            run_number = budget_counts[reasoning_budget]
                            if run_number > 1:
                                unique_instance_id = f"{instance['id']}_run{run_number}"
                            else:
                                unique_instance_id = instance["id"]
                            
                            # Get seed for this run if seeds are provided
                            run_seed = None
                            if seeds and global_run_index < len(seeds):
                                run_seed = seeds[global_run_index]
                            global_run_index += 1
                            
                            total_configs += 1
                            result_exists_and_valid = self.results_manager.check_result_exists(
                                existing_results,
                                unique_instance_id,
                                reasoning_budget,
                                icl_enabled=icl_enabled,
                                icl_num_examples=icl_num_examples if icl_enabled else 0,
                            )

                            if result_exists_and_valid:
                                skipped_configs.append(
                                    {
                                        "model_id": model_id,
                                        "task_id": task_id,
                                        "instance_id": unique_instance_id,
                                        "reasoning_budget": reasoning_budget,
                                        "icl_enabled": icl_enabled,
                                        "icl_num_examples": icl_num_examples if icl_enabled else 0,
                                    }
                                )
                                continue

                            icl_examples = []
                            if icl_enabled and icl_num_examples > 0:
                                # Use the original instance ID base (without run suffix) for ICL exclusion
                                original_instance_id_base = instance["id"].split("_run")[0]
                                potential_indices = [
                                    idx for idx, data in enumerate(full_task_data)
                                    if data["id"] == original_instance_id_base
                                ]
                                if not potential_indices:
                                     logger.warning(f"Could not find original instance {original_instance_id_base} in full_task_data for ICL exclusion. Skipping ICL for this instance.")
                                else:
                                    current_original_index = potential_indices[0]

                                    candidate_indices = list(range(len(full_task_data)))
                                    if current_original_index < len(candidate_indices):
                                         del candidate_indices[current_original_index]

                                    if len(candidate_indices) >= icl_num_examples:
                                        sampled_indices = icl_random_gen.sample(candidate_indices, icl_num_examples)
                                        icl_examples = [full_task_data[i] for i in sampled_indices]
                                    else:
                                        logger.warning(f"Not enough unique examples in task {task_id} ({len(candidate_indices)}) to sample {icl_num_examples} for ICL. Skipping ICL for instance {unique_instance_id}.")

                            # Create modified instance with unique ID for repeated budget runs
                            instance_for_config = instance.copy()
                            instance_for_config["id"] = unique_instance_id

                            config = self._build_single_config(model_id, task_id, instance_for_config, reasoning_budget, icl_examples, run_seed)
                            if config:
                                all_eval_configs_to_run.append(config)

                    progress.advance(config_task)

        return all_eval_configs_to_run, skipped_configs, total_configs

    def _apply_validation_sampling(
        self,
        task_data: List[Dict[str, Any]],
        validation_mode: bool,
        validation_samples: int,
        validation_runs: int,
    ) -> List[Dict[str, Any]]:
        """Applies validation sampling to task data if validation mode is enabled."""
        if validation_mode and len(task_data) > validation_samples:
            sampled_instances = random.sample(task_data, validation_samples)
            task_instances = []
            for run in range(validation_runs):
                for instance in sampled_instances:
                    instance_copy = instance.copy()
                    instance_copy["id"] = f"{instance['id']}_run{run+1}"
                    task_instances.append(instance_copy)
            return task_instances
        else:
            return task_data

    def _build_single_config(
        self,
        model_id: str,
        task_id: str,
        instance: Dict[str, Any],
        reasoning_budget: int,
        icl_examples: List[Dict[str, Any]],
        seed: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Builds a single evaluation configuration dictionary."""
        # Extract metric from task metadata, default to 'accuracy' if not found
        metric = instance.get("metric", "accuracy") # Defaulting here for safety, can be made stricter later
        if metric not in ["accuracy", "mse", "relative_error"]:
            logger.warning(f"Instance {instance['id']} in task {task_id} has unsupported metric '{metric}' in metadata. Defaulting to accuracy.")
            metric = "accuracy"

        base_config = {
            "model_id": model_id,
            "task_id": task_id,
            "instance_id": instance["id"],
            "reasoning_budget": reasoning_budget,
            "prompt": instance["prompt"],
            "icl_examples": icl_examples,
            "metric": metric, # Explicitly add the determined metric
        }
        
        # Add seed if provided (for VLLM models)
        if seed is not None:
            base_config["seed"] = seed

        if "classes" in instance and "answer_index" in instance:
            base_config["classes"] = instance["classes"]
            base_config["answer_index"] = instance["answer_index"]
            return base_config
        elif "answer" in instance:
            base_config["answer"] = instance["answer"]
            return base_config
        else:
            logger.warning(
                f"Instance {instance['id']} in task {task_id} missing required fields ('classes'/'answer_index' or 'answer'). Skipping."
            )
            return None

    def _display_progress_summary(
        self, total_configs: int, skipped_configs: List[Dict], configs_to_run: List[Dict]
    ) -> None:
        """Displays the summary of configurations to be processed."""
        summary = Table(title="📊 Evaluation Progress", show_header=True)
        summary.add_column("Status", style="cyan")
        summary.add_column("Count", style="green", justify="right")
        summary.add_row("Total Configurations", str(total_configs))
        summary.add_row("Already Completed", str(len(skipped_configs)))
        summary.add_row("Remaining to Evaluate", str(len(configs_to_run)))
        self.console.print(summary)

    def _process_single_result(
        self,
        config: Dict[str, Any],
        model_result_raw: Optional[Dict[str, Any]],
        model_id: str,
        reasoning_budget: int,
    ) -> Dict[str, Any]:
        """Processes a single raw result from the model interface, calculating correctness or error."""
        # Determine ICL status from the input config for this result
        icl_enabled = bool(config.get("icl_examples"))
        icl_num_examples = len(config.get("icl_examples", []))

        try:
            metric = config["metric"]
        except KeyError:
             logger.error(f"Metric key missing in config for instance {config.get('instance_id', 'N/A')}. This should not happen.")
             # Decide on fallback behavior: raise error or default? For now, default to accuracy and log error.
             metric = "accuracy"

        if model_result_raw and model_result_raw.get("error", None) is None:
            processed_result = model_result_raw.copy()
            processed_result.update({
                "model": model_id,
                "task_id": config["task_id"],
                "instance_id": config["instance_id"],
                "prompt": config["prompt"],
                "metric": metric,
                "reasoning_budget": reasoning_budget,
                "icl_enabled": icl_enabled,
                "icl_num_examples": icl_num_examples,
            })

            config_classes = config.get("classes")
            config_answer_index = config.get("answer_index")
            config_answer_text = config.get("answer")
            model_extracted_answer = processed_result.get("extracted_answer")
            instance_id = config.get("instance_id", "N/A")

            predicted_index = None
            is_correct = None
            squared_error = None
            relative_error = None

            if metric == "accuracy":
                if config_classes is not None and config_answer_index is not None:
                    is_correct, predicted_index = self._check_mcq_correctness(
                        config_classes, config_answer_index, model_extracted_answer, instance_id
                    )
                    processed_result["correct_answer_index"] = config_answer_index
                elif config_answer_text is not None:
                    is_correct = self._check_open_ended_correctness(
                        config_answer_text, model_extracted_answer, instance_id
                    )
                    predicted_index = None
                    processed_result["correct_answer"] = config_answer_text
                else:
                    logger.error(f"Accuracy task type unknown or missing ground truth for instance {instance_id}")
                    is_correct = False
            elif metric == "mse":
                if config_answer_text is not None and model_extracted_answer is not None:
                    processed_result["correct_answer"] = config_answer_text
                    try:
                        ground_truth_float = float(config_answer_text)
                        model_float = float(model_extracted_answer)
                        squared_error = (model_float - ground_truth_float) ** 2
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Could not convert answers to float for MSE calculation. "
                            f"Ground truth: '{config_answer_text}', Model answer: '{model_extracted_answer}'. "
                            f"Instance: {instance_id}"
                        )
                else:
                    logger.warning(
                        f"Missing ground truth ('{config_answer_text}') or model answer ('{model_extracted_answer}') "
                        f"for MSE calculation. Instance: {instance_id}"
                    )
                # 'correct' remains None for MSE tasks
            elif metric == "relative_error":
                if config_answer_text is not None and model_extracted_answer is not None:
                    processed_result["correct_answer"] = config_answer_text
                    try:
                        # Clean numeric strings by removing thousands separators and whitespace
                        clean_ground_truth = str(config_answer_text).replace(',', '').replace(' ', '')
                        clean_model_answer = str(model_extracted_answer).replace(',', '').replace(' ', '')
                        
                        ground_truth_float = float(clean_ground_truth)
                        model_float = float(clean_model_answer)
                        if ground_truth_float == 0:
                            # Handle division by zero - use absolute error when ground truth is 0
                            relative_error = abs(model_float - ground_truth_float)
                            logger.debug(f"Ground truth is 0, using absolute error instead of relative error for instance {instance_id}")
                        else:
                            relative_error = abs(model_float - ground_truth_float) / abs(ground_truth_float)
                    except (ValueError, TypeError, ZeroDivisionError):
                        logger.warning(
                            f"Could not convert answers to float for relative error calculation. "
                            f"Ground truth: '{config_answer_text}', Model answer: '{model_extracted_answer}'. "
                            f"Instance: {instance_id}"
                        )
                else:
                    logger.warning(
                        f"Missing ground truth ('{config_answer_text}') or model answer ('{model_extracted_answer}') "
                        f"for relative error calculation. Instance: {instance_id}"
                    )
                # 'correct' remains None for relative error tasks
            else:
                logger.error(f"Unsupported metric type '{metric}' for instance {instance_id}")

            processed_result["predicted_index"] = predicted_index
            processed_result["correct"] = is_correct
            processed_result["squared_error"] = squared_error
            processed_result["relative_error"] = relative_error
            return processed_result

        else:
            error_msg = model_result_raw.get("error", "Unknown processing error") if model_result_raw else "Model returned None result"
            logger.warning(f"Error in model response for {config.get('instance_id', 'N/A')} (Budget: {reasoning_budget}): {error_msg}")
            cost = model_result_raw.get("cost", 0) if model_result_raw else 0
            latency = model_result_raw.get("latency", 0) if model_result_raw else 0
            # Get metric from config, fallback if missing (though it shouldn't be)
            metric_in_error = config.get("metric", "accuracy")
            return {
                "model": model_id,
                "task_id": config.get("task_id", "N/A"),
                "instance_id": config.get("instance_id", "N/A"),
                "reasoning_budget": reasoning_budget,
                "icl_enabled": icl_enabled,
                "icl_num_examples": icl_num_examples,
                "metric": metric_in_error, # Use the retrieved metric
                "error": error_msg,
                "correct": None, # Set 'correct' to None in error cases
                "squared_error": None, # Add squared_error field, set to None
                "cost": cost,
                "latency": latency,
            }

    def _check_mcq_correctness(
        self, classes: List[Any], correct_answer_index: int, model_answer: Optional[str], instance_id: str
    ) -> Tuple[bool, Optional[int]]:
        """Checks correctness for a multiple-choice question."""
        predicted_index = None
        if model_answer is None:
             return False, None

        try:
            model_answer_norm = str(model_answer).strip().lower()
            found_match = False
            for i, class_option in enumerate(classes):
                class_option_str = str(class_option).strip().lower() if class_option is not None else ""
                if model_answer_norm == class_option_str:
                    predicted_index = i
                    found_match = True
                    break

            if not found_match and len(model_answer_norm) == 1:
                logger.debug(f"MCQ Fallback: Answer '{model_answer}' not directly in classes: {classes}. Trying A/B/C index. Instance: {instance_id}")
                try:
                    idx = ord(model_answer_norm.upper()) - ord('A')
                    if 0 <= idx < len(classes):
                        predicted_index = idx
                        logger.debug(f"MCQ Fallback: Matched index {idx}. Instance: {instance_id}")
                    else:
                        logger.warning(f"MCQ Fallback: Calculated index {idx} out of range ({len(classes)} classes). Instance: {instance_id}")
                        predicted_index = None
                except (TypeError, ValueError):
                    logger.warning(f"MCQ Fallback: Invalid character '{model_answer}' for A/B/C index. Instance: {instance_id}")
                    predicted_index = None

        except (ValueError, TypeError, IndexError) as e:
            logger.warning(f"MCQ Error processing answer '{model_answer}' for classes '{classes}'. Error: {e}. Instance: {instance_id}")
            predicted_index = None

        is_correct = (predicted_index == correct_answer_index) if predicted_index is not None else False
        return is_correct, predicted_index

    def _check_open_ended_correctness(
        self, ground_truth_answer: Any, model_answer: Optional[str], instance_id: str
    ) -> bool:
        """Checks correctness for an open-ended question."""
        if model_answer is None:
            return False

        try:
            ground_truth_norm = str(ground_truth_answer).strip().lower()
            model_output_norm = str(model_answer).strip().lower()

            if not ground_truth_norm:
                is_correct = not model_output_norm
                logger.debug(f"Open-Ended (Empty GT): Model output '{model_output_norm}'. Correct: {is_correct}. Instance: {instance_id}")
                return is_correct

            try:
                ground_truth_float = float(ground_truth_norm)
                model_output_float = float(model_output_norm)
                is_correct = abs(ground_truth_float - model_output_float) < 1e-6
                logger.debug(f"Open-Ended (Numeric): Comparing {ground_truth_float} == {model_output_float} -> {is_correct}. Instance: {instance_id}")
                return is_correct
            except (ValueError, TypeError):
                is_correct = (ground_truth_norm == model_output_norm)
                logger.debug(f"Open-Ended (String): Comparing '{ground_truth_norm}' == '{model_output_norm}' -> {is_correct}. Instance: {instance_id}")
                return is_correct

        except Exception as e:
            logger.error(f"Open-Ended Error: Comparing model answer '{model_answer}' to ground truth '{ground_truth_answer}'. Error: {e}. Instance: {instance_id}")
            return False

    async def _process_evaluation_batch(
        self,
        model_id: str,
        reasoning_budget: int,
        batch_configs: List[Dict[str, Any]],
        progress: Progress,
        eval_task_id: TaskID,
    ) -> Tuple[int, float, int, List[Dict[str, Any]]]:
        """Processes a single batch of evaluations for a given model and budget."""
        processed_count = 0
        batch_cost = 0.0
        correct_count = 0
        batch_new_results = []

        prompt_texts = [config["prompt"] for config in batch_configs]
        possible_answers_list = [config.get("classes") for config in batch_configs]
        budgets_list = [reasoning_budget] * len(batch_configs)
        icl_examples_list = [config["icl_examples"] for config in batch_configs]
        seeds_list = [config.get("seed") for config in batch_configs]  # Extract seeds if present

        try:
            if isinstance(self.model_interface, BatchModelInterface):
                model_results_raw = await self.model_interface.evaluate_prompts_batch(
                    model_id,
                    prompt_texts,
                    possible_answers_list,
                    budgets_list,
                    icl_examples_list=icl_examples_list,
                    external_progress=progress,
                    seeds_list=seeds_list,
                )
            else:
                logger.warning("Running batch processing logic with non-batch model interface. This might be inefficient and requires the interface's evaluate_prompt method to support 'icl_examples'.")
                model_results_raw = []
                for idx, config in enumerate(batch_configs):
                    result = await self.model_interface.evaluate_prompt(
                        model_id=model_id,
                        prompt_text=config["prompt"],
                        possible_answers=config.get("classes"),
                        reasoning_budget=reasoning_budget,
                        icl_examples=config["icl_examples"],
                    )
                    model_results_raw.append(result)

            for i, config in enumerate(batch_configs):
                model_result_raw = model_results_raw[i] if i < len(model_results_raw) else None
                processed_result = self._process_single_result(config, model_result_raw, model_id, reasoning_budget)
                batch_new_results.append(processed_result)
                batch_cost += processed_result.get("cost", 0)
                if processed_result.get("correct"):
                    correct_count += 1
                processed_count += 1
                progress.advance(eval_task_id)

        except Exception as e:
            logger.error(f"❌ Error processing batch for model {model_id}, budget {reasoning_budget}: {e}", exc_info=True)
            for config in batch_configs:
                 error_result = self._create_error_result(config, model_id, reasoning_budget, f"Batch processing failed: {repr(e)}")
                 batch_new_results.append(error_result)
                 processed_count += 1
                 progress.advance(eval_task_id)

        return processed_count, batch_cost, correct_count, batch_new_results

    def _create_error_result(self, config: Dict[str, Any], model_id: str, reasoning_budget: int, error_message: str) -> Dict[str, Any]:
         """Creates a standardized result dictionary for an error case."""
         return {
             "model": model_id,
             "task_id": config["task_id"],
             "instance_id": config["instance_id"],
             "prompt": config["prompt"],
             "reasoning_budget": reasoning_budget,
             "icl_enabled": bool(config.get("icl_examples")),
             "icl_num_examples": len(config.get("icl_examples", [])),
             "error": error_message,
             "response": None,
             "extracted_answer": None,
             "correct": None,
             "latency": 0,
             "cost": 0,
             "raw_response": None,
         }

    def _display_final_statistics(self, results_count: int, total_correct: int, total_cost: float) -> None:
        """Displays the final evaluation statistics."""
        stats_table = Table(title="🎯 Final Evaluation Results", show_header=True)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        stats_table.add_row("Total Evaluations Run", str(results_count))
        if results_count > 0:
             accuracy = (total_correct / results_count * 100)
             stats_table.add_row(
                 "Correct Answers",
                 f"{total_correct}/{results_count} ({accuracy:.1f}%)",
             )
        else:
             stats_table.add_row("Correct Answers", "0/0 (N/A)")
        stats_table.add_row("Estimated Total Cost", f"${total_cost:.4f}")
        self.console.print(stats_table)

    async def run_evaluations(
        self,
        models: List[str],
        tasks: List[str],
        reasoning_budgets: List[int],
        validation_mode: bool = False,
        validation_samples: int = 20,
        validation_runs: int = 3,
        validation_seed: int = 42,
        icl_config: Optional[Dict[str, Any]] = None,
        seeds: Optional[List[int]] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run evaluations for the specified models, tasks, and reasoning budgets.
        Supports resuming from previous runs and real-time result saving.

        Args:
            models: List of model IDs to evaluate
            tasks: List of task IDs to evaluate on
            reasoning_budgets: List of reasoning budgets to use
            validation_mode: Whether to run in validation mode (sample instances)
            validation_samples: Number of instances to sample per task in validation mode
            validation_runs: Number of runs per sampled instance in validation mode
            validation_seed: Random seed for validation sampling and ICL sampling
            icl_config: Dictionary containing ICL configuration (enabled, num_examples)

        Returns:
            A tuple containing:
            - summary: Dictionary containing overall statistics
            - loaded_results_list: List of all processed results (loaded + new)
        """
        self._display_initial_config(
            models, tasks, reasoning_budgets,
            validation_mode, validation_samples, validation_runs, validation_seed,
            icl_config
        )

        configs_to_run, skipped_configs, total_configs = self._generate_evaluation_configs(
            models, tasks, reasoning_budgets,
            validation_mode, validation_samples, validation_runs, validation_seed,
            icl_config, seeds
        )
        self._display_progress_summary(total_configs, skipped_configs, configs_to_run)

        if not configs_to_run:
            self.console.print("✅ No remaining evaluations to run.", style="green")
            all_results = self.results_manager.load_all_results()
            summary_dict, filtered_results = self.results_manager.summarize_results(models, tasks, reasoning_budgets)
            return summary_dict, all_results

        configs_to_run.sort(key=lambda x: (x["model_id"], x["reasoning_budget"]))
        grouped_configs = groupby(configs_to_run, key=lambda x: (x["model_id"], x["reasoning_budget"]))

        all_new_results = []
        total_cost = 0.0
        total_new_evaluations = len(configs_to_run)

        logger.info(f"Total new evaluations: {total_new_evaluations}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            eval_task_id = progress.add_task(
                "Running evaluations...", total=total_new_evaluations
            )

            # Check if we're using VLLM interface - if so, process sequentially
            is_vllm_interface = VLLMModelInterface is not None and isinstance(self.model_interface, VLLMModelInterface)
            
            if is_vllm_interface:
                logger.info("🚀 VLLM interface detected - using sequential batch processing for optimal GPU utilization")
                batch_results_list = []
                for (model_id, reasoning_budget), group in grouped_configs:
                    batch_configs = list(group)
                    logger.info(f"Processing VLLM batch: {model_id}, budget {reasoning_budget}, {len(batch_configs)} prompts")
                    batch_result = await self._process_evaluation_batch(
                        model_id,
                        reasoning_budget,
                        batch_configs,
                        progress,
                        eval_task_id,
                    )
                    batch_results_list.append(batch_result)
            else:
                # For API-based models, use concurrent async processing
                logger.info("🌐 API-based interface detected - using concurrent async processing")
                evaluation_tasks = []
                for (model_id, reasoning_budget), group in grouped_configs:
                    batch_configs = list(group)
                    task = asyncio.create_task(
                        self._process_evaluation_batch(
                            model_id,
                            reasoning_budget,
                            batch_configs,
                            progress,
                            eval_task_id,
                        )
                    )
                    evaluation_tasks.append(task)

                logger.info(f"Evaluation tasks: {len(evaluation_tasks)}")
                batch_results_list = await asyncio.gather(*evaluation_tasks)

            for (
                processed_count,
                batch_cost,
                correct_count,
                batch_new_results,
            ) in batch_results_list:
                total_cost += batch_cost
                all_new_results.extend(batch_new_results)
                processed_count += processed_count
                progress.advance(eval_task_id)

        # Save results one by one using the existing save_result method
        self.console.print(f"💾 Saving {len(all_new_results)} new results...", style="yellow")
        for result in all_new_results:
            # Extract necessary fields for save_result
            # Handle potential missing keys gracefully although they should be present
            model_id = result.get("model", "unknown_model")
            task_id = result.get("task_id", "unknown_task")
            instance_id = result.get("instance_id", "unknown_instance")
            reasoning_budget = result.get("reasoning_budget", -1) # Use a placeholder if missing

            # Ensure reasoning_budget is an int, default to -1 if conversion fails
            try:
                reasoning_budget = int(reasoning_budget) if reasoning_budget is not None else -1
            except (ValueError, TypeError):
                 logger.warning(f"Could not convert reasoning_budget '{reasoning_budget}' to int for saving instance {instance_id}. Using -1.")
                 reasoning_budget = -1

            self.results_manager.save_result(
                 model_id=model_id,
                 task_id=task_id,
                 instance_id=instance_id,
                 reasoning_budget=reasoning_budget,
                 result=result, # Pass the full result dict
            )

        self.console.print("🔄 Reloading all results for final summary...", style="yellow")
        all_results_combined = self.results_manager.load_all_results()

        # Generate the final summary using the correct method and arguments
        final_summary, _ = self.results_manager.summarize_results(
            models=models,
            tasks=tasks,
            reasoning_budgets=reasoning_budgets
        )
        # final_summary = self.results_manager.generate_summary(all_results_combined) # Original incorrect line
        self.console.print(f"💰 Total cost for new evaluations: [blue]${total_cost:.4f}[/]")

        return final_summary, all_results_combined # Return summary and combined results

    async def evaluate_prompts(
        self,
        model_id: str,
        prompts: List[str],
        possible_answers: Optional[List[List[str]]] = None,
        reasoning_budgets: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a list of prompts using the model interface. (Helper/alternative entry point)

        Args:
            model_id: ID of the model to use
            prompts: List of prompts to evaluate
            possible_answers: Optional list of possible answers for multiple-choice tasks.
                           If None, treats the prompts as open-ended tasks.
            reasoning_budgets: Optional list of reasoning budgets for each prompt. Defaults to 0.

        Returns:
            List of dictionaries containing raw model responses and metadata.
            Note: This does not perform correctness checking or result saving.
        """
        if not prompts:
            return []
        if reasoning_budgets is None:
            reasoning_budgets = [0] * len(prompts)
        elif len(reasoning_budgets) != len(prompts):
             raise ValueError("Length of reasoning_budgets must match length of prompts")

        if possible_answers is not None and len(possible_answers) != len(prompts):
             raise ValueError("Length of possible_answers must match length of prompts if provided")

        self.console.print(f"🚀 Evaluating {len(prompts)} prompts directly with model [green]{model_id}[/]...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            eval_task_id = progress.add_task(
                f"Evaluating prompts with {model_id}", total=len(prompts)
            )

            results = await self.model_interface.evaluate_prompts_batch(
                model_id=model_id,
                prompt_texts=prompts,
                possible_answers_list=possible_answers,
                reasoning_budgets=reasoning_budgets,
                external_progress=None,
            )
            progress.update(eval_task_id, completed=len(prompts))

        self.console.print(f"✅ Finished direct evaluation of {len(prompts)} prompts.")
        return results
