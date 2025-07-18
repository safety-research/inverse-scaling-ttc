"""
Interface for accessing models using batch API operations.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from hydra.core.hydra_config import HydraConfig

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from safetytooling.apis.inference.anthropic import AnthropicModelBatch
from safetytooling.apis.inference.openai.batch_api import OpenAIModelBatch
from safetytooling.data_models import Prompt

from src.model_interface import ModelInterface
# Import shared helper functions
from src.utils.model_helpers import (
    _calculate_max_tokens,
    _extract_answer_tag,
    _get_reasoning_params,
    _get_reasoning_system_prompt,
    _prepare_prompt,
    import_time,
)

logger = logging.getLogger(__name__)


class BatchModelInterface(ModelInterface):
    """Interface for batch API access to models."""

    def __init__(
        self,
        models_config: Dict[str, Dict[str, Any]],
        evaluation_config: Dict[str, Any],
        use_cache: bool = True,
        anthropic_api_key_tag: str = "ANTHROPIC_API_KEY",
        batch_size: int = 100,  # Default batch size
        max_concurrent_batches: int = 20,  # Default max concurrent batches
    ):
        """
        Initialize the BatchModelInterface.

        Args:
            models_config: Configuration dictionary for supported models
            evaluation_config: Configuration for the evaluation run
            use_cache: Whether to use response caching
            anthropic_api_key_tag: Environment variable name for Anthropic API key
            batch_size: Number of prompts to process in each batch
            max_concurrent_batches: Maximum number of batches to process concurrently
        """
        super().__init__(models_config, use_cache, anthropic_api_key_tag, evaluation_config)
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches

        self.log_dir = Path(HydraConfig.get().runtime.output_dir) / "batch_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize batch API client
        logger.info("✨ Batch model interface initialized successfully")

    async def evaluate_prompt(
        self,
        model_id: str,
        prompt_text: str,
        possible_answers: List[str],
        reasoning_budget: int = 0,
        icl_examples: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single prompt using batch processing under the hood.

        Args:
            model_id: ID of the model to use
            prompt_text: The prompt to evaluate
            possible_answers: The possible classes/answers
            reasoning_budget: Reasoning budget in tokens
            icl_examples: Optional list of in-context learning examples

        Returns:
            Dict containing the model's response and metadata
        """
        # Create simpler progress display for single prompt evaluation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            progress.add_task(
                f"[cyan]Evaluating prompt with model {model_id} (budget: {reasoning_budget})",
                total=None,
            )

            # Copy possible_answers to avoid modifying the original
            possible_answers_copy = possible_answers.copy() if possible_answers else []

            # Create minimal copies of ICL examples if provided
            icl_examples_clean = None
            if icl_examples and len(icl_examples) > 0:
                icl_examples_clean = []
                for example in icl_examples:
                    # Only copy essential fields from each example
                    cleaned_example = {
                        k: v for k, v in example.items()
                        if k in {"input", "output", "task", "label"}
                    }
                    icl_examples_clean.append(cleaned_example)

            # Call the batch method with a single prompt
            results = await self.evaluate_prompts_batch(
                model_id,
                [prompt_text],  # Single prompt in a list
                [possible_answers_copy],  # Single set of possible answers in a list
                [reasoning_budget],  # Single budget in a list
                icl_examples_list=[icl_examples_clean] if icl_examples_clean is not None else None
            )

            # Extract the single result
            result = results[0] if results else {
                "model": model_id,
                "reasoning_budget": reasoning_budget,
                "error": "No response received",
                "response": None,
                "extracted_answer": None,
                "latency": 0,
                "cost": 0,
                "reasoning_content": None,
            }

            # Log the result
            if "error" not in result or result["error"] is None:
                logger.info(f"✅ Successfully evaluated prompt with model {model_id}")
                if result.get("extracted_answer"):
                    logger.info(f"📝 Answer: [green]{result['extracted_answer']}[/]")
                logger.info(f"⏱️  Latency: [yellow]{result.get('latency', 0):.2f}s[/]")
                logger.info(f"💰 Cost: [blue]${result.get('cost', 0):.4f}[/]")
            else:
                logger.error(f"❌ Error: {result['error']}")

            # Clean up temporary variables to aid garbage collection
            del possible_answers_copy
            del icl_examples_clean
            del results

            return result

    async def evaluate_prompts_batch(
        self,
        model_id: str,
        prompt_texts: List[str],
        possible_answers_list: List[List[str]],
        reasoning_budgets: List[int],
        external_progress: Optional[Progress] = None,
        icl_examples_list: Optional[List[Optional[List[Dict[str, Any]]]]] = None,
        seeds_list: Optional[List[Optional[int]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple prompts in batches, supporting different reasoning budgets per prompt.
        Correctly groups prompts by exact reasoning budget value for efficient batch processing.

        Args:
            model_id: ID of the model to use
            prompt_texts: List of prompts to evaluate
            possible_answers_list: List of possible answers for each prompt
            reasoning_budgets: List of reasoning budgets, one per prompt
            external_progress: Optional external progress bar to use
            icl_examples_list: Optional list of lists of ICL examples, one outer list entry per prompt

        Returns:
            List of dictionaries containing model responses and metadata

        Raises:
            ValueError: If input lists have different lengths or model_id is invalid
            RuntimeError: If batch API call fails
        """
        # Validate inputs
        if not (
            len(prompt_texts) == len(possible_answers_list) == len(reasoning_budgets)
        ):
            raise ValueError("Input lists must have the same length")

        if model_id not in self.models_config:
            raise ValueError(f"Invalid model_id: {model_id}")

        total_prompts = len(prompt_texts)
        model_config = self.models_config[model_id]

        # Validate ICL list length if provided
        if icl_examples_list is not None and len(icl_examples_list) != total_prompts:
            raise ValueError("Length of icl_examples_list must match length of prompt_texts")

        # If icl_examples_list is None, create a list of Nones for easier zipping
        if icl_examples_list is None:
            icl_examples_list = [None] * total_prompts

        # Configure progress bar context
        if external_progress is None:
            progress_context = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
            )
            use_external_progress = False
        else:
            progress_context = external_progress
            use_external_progress = True

        # Initialize results map to store final outputs
        all_results_map = {}  # Store results keyed by original_index
        final_responses = [None] * total_prompts  # Pre-allocate final response list

        with progress_context as progress:
            # --- PHASE 1: Prepare prompts in batches ---
            if not use_external_progress:
                prep_task = progress.add_task(
                    "Preparing prompts in parallel", total=total_prompts
                )

            # Process prompts in smaller batches to reduce memory usage
            batch_size = min(100, total_prompts)  # Smaller batch size for preparation
            prepared_prompts_map = {}  # Map original_index to prepared Prompt object

            # Group prompts by budget for more efficient processing
            budget_to_indices = {}
            for i, budget in enumerate(reasoning_budgets):
                if budget not in budget_to_indices:
                    budget_to_indices[budget] = []
                budget_to_indices[budget].append(i)

            # Process each budget group
            for budget, indices in budget_to_indices.items():
                base_system_prompt = model_config.get("system_prompt", "")

                # Prepare prompts in batches for this budget group
                for batch_start in range(0, len(indices), batch_size):
                    batch_indices = indices[batch_start:batch_start + batch_size]
                    prepare_tasks = []

                    # Create preparation tasks for this batch
                    for i in batch_indices:
                        # Get reasoning prompt part (same for all prompts with this budget)
                        is_multiple_choice = bool(possible_answers_list[i])
                        reasoning_prompt_part = _get_reasoning_system_prompt(
                            budget,
                            is_multiple_choice,
                            prompt_use_all_budget=self.evaluation_config.get("prompt_use_all_budget", True)
                        )
                        full_system_prompt = base_system_prompt + reasoning_prompt_part

                        # Create task to prepare this prompt
                        task = asyncio.create_task(self._prepare_prompt_async(
                            prompt_texts[i],
                            possible_answers_list[i],
                            model_id,
                            full_system_prompt,
                            i,  # original_index
                            icl_examples_list[i]
                        ))
                        prepare_tasks.append(task)

                    # Process this batch of prompts
                    prepared_results = await asyncio.gather(*prepare_tasks)

                    # Store results and update progress
                    for original_index, prepared_prompt in prepared_results:
                        prepared_prompts_map[original_index] = prepared_prompt
                        if not use_external_progress:
                            progress.advance(prep_task)

                    # Clear references to tasks to help garbage collection
                    prepare_tasks.clear()

            if not use_external_progress:
                # Mark prep task complete
                progress.update(prep_task, completed=total_prompts, visible=False)

            # --- PHASE 2: Group prompts by budget and create batches ---
            start_time = import_time()
            all_batch_configs = []  # Will store configs for all batches to be processed

            # Process each budget group
            for budget, indices in budget_to_indices.items():
                # Skip any indices where prompt preparation failed
                valid_indices = [i for i in indices if i in prepared_prompts_map]

                # Handle failed prompt preparations
                for i in indices:
                    if i not in prepared_prompts_map:
                        all_results_map[i] = {
                            "model": model_id,
                            "reasoning_budget": budget,
                            "error": "Prompt preparation failed",
                            "response": None,
                            "extracted_answer": None,
                            "latency": 0,
                            "cost": 0,
                            "reasoning_content": None,
                        }

                # Skip if no valid prompts for this budget
                if not valid_indices:
                    continue

                # Create batches for this budget
                num_batches = (len(valid_indices) + self.batch_size - 1) // self.batch_size
                logger.info(f"Creating {num_batches} batches for budget value {budget}.")

                for batch_idx in range(num_batches):
                    batch_start = batch_idx * self.batch_size
                    batch_end = min((batch_idx + 1) * self.batch_size, len(valid_indices))
                    batch_indices = valid_indices[batch_start:batch_end]

                    # Create minimal data structure for this batch
                    current_prompts = [prepared_prompts_map[i] for i in batch_indices]
                    current_metadata = [{"original_index": i, "budget": budget} for i in batch_indices]

                    # Configure API parameters
                    kwargs = {**model_config.get("api_params", {})}
                    is_thinking_model = self.evaluation_config.get("thinking", True)

                    if not is_thinking_model:
                        kwargs["max_tokens"] = budget if budget > 0 else model_config.get("default_max_tokens", 10000)
                        if budget == 0:
                            logger.warning(f"Budget is 0 for non-thinking model {model_id}. Using default max_tokens: {kwargs['max_tokens']}")
                    else:
                        if model_config.get("type", None) == "openai":
                            kwargs["max_completion_tokens"] = _calculate_max_tokens(budget, base_tokens=10000)
                        else:
                            kwargs["max_tokens"] = _calculate_max_tokens(budget, base_tokens=10000)

                    if is_thinking_model and budget != 0:
                        # If budget is 0, it means no reasoning is allowed
                        # If budget is negative, it means natural overthinking is allowed
                        # If budget is positive, it means reasoning is allowed with a forced budget
                        reasoning_params = _get_reasoning_params(model_id, budget, self.models_config)
                        kwargs.update(reasoning_params)

                    # Store all the configuration needed to create and process this batch
                    all_batch_configs.append({
                        "model_type": model_config["type"],
                        "model_name": model_config["model_name"],
                        "prompts": current_prompts,
                        "metadata": current_metadata,
                        "budget": budget,
                        "batch_idx": batch_idx,
                        "kwargs": kwargs,
                    })

            # Clear prepared_prompts_map to free memory after batch configs are created
            prepared_prompts_map.clear()

            # --- PHASE 3: Process batches in strictly limited groups ---
            total_batches_to_process = len(all_batch_configs)
            processed_batches_count = 0

            if total_batches_to_process > 0:
                if not use_external_progress:
                    batch_task_progress = progress.add_task(
                        f"Processing {total_batches_to_process} batches ({total_prompts} prompts)",
                        total=total_batches_to_process
                    )

                # Process batches in strictly limited groups
                for i in range(0, total_batches_to_process, self.max_concurrent_batches):
                    # Take the next batch of configs, limited by max_concurrent_batches
                    current_batch_configs = all_batch_configs[i:i + self.max_concurrent_batches]
                    current_tasks = []
                    current_metadata = []

                    # Now create the actual tasks for just this group
                    for config in current_batch_configs:
                        try:
                            if config["model_type"] == "anthropic":
                                batch_api = AnthropicModelBatch()
                            elif config["model_type"] == "openai":
                                batch_api = OpenAIModelBatch()
                            else:
                                raise ValueError(f"Invalid model type: {config['model_type']}")

                            task = asyncio.create_task(
                                batch_api(
                                    model_id=config["model_name"],
                                    prompts=config["prompts"],
                                    log_dir=self.log_dir,
                                    **config["kwargs"],
                                )
                            )
                            current_tasks.append(task)
                            current_metadata.append({
                                "budget": config["budget"],
                                "metadata": config["metadata"]
                            })
                        except Exception as e:
                            error_msg = f"Batch task creation failed: {str(e)}"
                            logger.error(f"❌ Failed to create batch task for budget {config['budget']}, batch {config['batch_idx']}: {error_msg}", exc_info=True)
                            for meta in config["metadata"]:
                                all_results_map[meta["original_index"]] = {
                                    "model": model_id,
                                    "reasoning_budget": meta["budget"],
                                    "error": error_msg,
                                    "response": None,
                                    "extracted_answer": None,
                                    "latency": 0,
                                    "cost": 0,
                                    "reasoning_content": None,
                                }

                    logger.info(f"Processing batch group {i//self.max_concurrent_batches + 1}/{(total_batches_to_process + self.max_concurrent_batches - 1)//self.max_concurrent_batches}: {len(current_tasks)} batches for budget {budget}")

                    # Process this limited group of batches
                    batch_results = await asyncio.gather(*current_tasks, return_exceptions=True)

                    # Handle batch results for this group
                    for j, (batch_result, meta_info) in enumerate(zip(batch_results, current_metadata)):
                        meta_data = meta_info["metadata"]
                        budget = meta_info["budget"]

                        if isinstance(batch_result, Exception):
                            error_msg = repr(batch_result)
                            logger.error(f"❌ Error in batch API call (budget {budget}): {error_msg}", exc_info=True)
                            for meta in meta_data:
                                all_results_map[meta["original_index"]] = {
                                    "model": model_id,
                                    "reasoning_budget": meta["budget"],
                                    "error": f"Batch API call failed: {error_msg}",
                                    "response": None,
                                    "extracted_answer": None,
                                    "latency": import_time() - start_time,
                                    "cost": 0,
                                    "reasoning_content": None,
                                }
                        else:
                            # Process individual responses in this batch
                            responses, batch_id = batch_result

                            # Process responses in smaller chunks
                            for k in range(0, len(responses), 20):
                                chunk_responses = responses[k:k + 20]
                                chunk_meta = meta_data[k:k + 20]

                                # Create processing tasks
                                process_tasks = [
                                    self.process_response(resp, model_id, meta["budget"], start_time)
                                    for resp, meta in zip(chunk_responses, chunk_meta)
                                ]

                                # Process this chunk
                                processed_results = await asyncio.gather(*process_tasks, return_exceptions=True)

                                # Store results
                                for processed_result, meta in zip(processed_results, chunk_meta):
                                    idx = meta["original_index"]
                                    if isinstance(processed_result, Exception):
                                        error_msg = repr(processed_result)
                                        logger.error(f"❌ Error processing response for index {idx}: {error_msg}", exc_info=True)
                                        all_results_map[idx] = {
                                            "model": model_id,
                                            "reasoning_budget": meta["budget"],
                                            "error": f"Response processing failed: {error_msg}",
                                            "response": None,
                                            "extracted_answer": None,
                                            "latency": import_time() - start_time,
                                            "cost": 0,
                                            "reasoning_content": None,
                                        }
                                    else:
                                        all_results_map[idx] = processed_result

                            # Clear references to responses
                            responses.clear()

                        processed_batches_count += 1
                        if not use_external_progress:
                            progress.update(
                                batch_task_progress,
                                advance=1,
                                description=f"Processed batch API call {processed_batches_count}/{total_batches_to_process}"
                            )

                    # Clear references to this batch group's tasks and configs to help garbage collection
                    current_tasks.clear()
                    current_metadata.clear()
            else:
                logger.info("No batch tasks were created to run.")
                if not use_external_progress:
                    batch_task_progress = progress.add_task("Processing batches", total=0)
                    progress.update(batch_task_progress, completed=0, visible=False)

            # Help garbage collection
            all_batch_configs.clear()

        # --- Build final responses ---
        for index in range(total_prompts):
            if index in all_results_map:
                final_responses[index] = all_results_map[index]
            else:
                # This case should ideally not happen if error handling above is correct
                logger.error(f"Result for original index {index} was unexpectedly missing from map.")

                # Find original budget for this missing index
                original_budget = next((budget for i, budget in enumerate(reasoning_budgets) if i == index), -1)

                final_responses[index] = {
                    "model": model_id,
                    "reasoning_budget": original_budget,
                    "error": "Result missing after processing",
                    "response": None,
                    "extracted_answer": None,
                    "latency": import_time() - start_time,
                    "cost": 0,
                    "reasoning_content": None,
                }

        # --- Logging Summary ---
        total_cost = sum(r.get("cost", 0) for r in final_responses if r)
        total_time = import_time() - start_time
        success_count = sum(1 for r in final_responses if r and r.get("error") is None)

        self.console.print("\n📊 Batch Processing Summary:")
        self.console.print(
            f"✅ Successfully processed: [green]{success_count}/{total_prompts}[/] prompts"
        )
        self.console.print(f"⏱️  Total time: [yellow]{total_time:.2f}s[/]")
        self.console.print(f"💰 Total cost: [blue]${total_cost:.4f}[/]")
        if success_count < total_prompts:
            self.console.print(
                f"⚠️  Failed prompts: [red]{total_prompts - success_count}[/]",
                style="yellow",
            )

        # Clear all_results_map to help garbage collection
        all_results_map.clear()

        return final_responses

    async def _prepare_prompt_async(
        self,
        prompt_text: str,
        possible_answers: Optional[List[str]],
        model_id: str,
        system_prompt: Optional[str],
        original_index: int,
        icl_examples: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[int, Prompt]:
        """
        Asynchronously prepare a single prompt and return it with its original index.
        Wraps the synchronous _prepare_prompt logic.
        """
        # possible_answers can be None for open-ended tasks now
        if possible_answers is None:
            possible_answers = []  # Use empty list for helper

        # Copy only necessary data to avoid memory leaks
        prompt_text_copy = prompt_text
        possible_answers_copy = possible_answers.copy() if possible_answers else []

        # For ICL examples, create a minimal copy with just the essential fields
        icl_examples_clean = None
        if icl_examples and len(icl_examples) > 0:
            icl_examples_clean = []
            for example in icl_examples:
                # Only copy the essential fields from each example
                cleaned_example = {
                    k: v for k, v in example.items()
                    if k in {"input", "output", "task", "label"}
                }
                icl_examples_clean.append(cleaned_example)

        # Run the synchronous preparation in executor to avoid blocking
        loop = asyncio.get_running_loop()
        prepared_prompt = await loop.run_in_executor(
            None,
            _prepare_prompt,
            prompt_text_copy,
            possible_answers_copy,
            model_id,
            system_prompt,
            icl_examples_clean
        )

        # Help garbage collection
        del prompt_text_copy
        del possible_answers_copy
        del icl_examples_clean

        return original_index, prepared_prompt

    async def process_response(
        self,
        response,
        model_id: str,
        reasoning_budget: int,
        start_time: float,
    ) -> Dict[str, Any]:
        """
        Process a single LLMResponse object received from safety-tooling.

        Args:
            response: The LLMResponse object (or None if safety-tooling skipped an error)
            model_id: Model identifier
            reasoning_budget: Reasoning budget in tokens
            start_time: Start time for latency calculation (Note: LLMResponse duration might be more accurate if available)

        Returns:
            Processed response dictionary compatible with our results format.
        """
        end_time = import_time()
        latency = end_time - start_time  # Or use response.duration if applicable

        # Handle cases where safety-tooling might return None for a failed request
        if response is None:
            logger.warning(
                f"Received None response object for budget {reasoning_budget}, likely a skipped error in safety-tooling."
            )
            return {
                "model": model_id,
                "reasoning_budget": reasoning_budget,
                "latency": latency,
                "error": "Processing skipped due to error in underlying API call.",
                "response": None,
                "extracted_answer": None,
                "cost": 0,
                "reasoning_content": None,
            }

        # Extract only necessary fields from LLMResponse to avoid storing the entire object
        result = {
            "model": getattr(response, "model_id", model_id),
            "reasoning_budget": reasoning_budget,
            "latency": getattr(response, "duration", latency),
            "error": None,  # Will be updated if we detect issues
            "response": None,  # Will be updated after validation
            "extracted_answer": None,
            "cost": getattr(response, "cost", 0),
            "stop_reason": getattr(response, "stop_reason", None),
            "input_tokens": getattr(response, "input_tokens", None),
            "output_tokens": getattr(response, "output_tokens", None),
            "reasoning_content": getattr(response, "reasoning_content", None),
            "usage": getattr(response, "usage", None),
        }

        # Get the completion text, handling potential ThinkingBlock issues
        completion_text = getattr(response, "completion", None)
        if completion_text is None:
            result["error"] = "No completion text found in response"
        elif not isinstance(completion_text, str):
            # Handle case where completion might be a ThinkingBlock or other object
            try:
                if hasattr(completion_text, "text"):
                    completion_text = completion_text.text
                elif hasattr(completion_text, "thinking"):
                    # If it's a ThinkingBlock without text but with thinking content
                    completion_text = str(completion_text.thinking)
                    result["reasoning_content"] = completion_text
                else:
                    result["error"] = f"Completion is not a string and has no text/thinking attribute: {type(completion_text)}"
                    completion_text = str(completion_text)  # Convert to string as fallback
            except Exception as e:
                result["error"] = f"Error processing completion object: {str(e)}"
                completion_text = None

        # Update result with processed completion text
        result["response"] = completion_text

        # Extract answer letter from the completion text if available
        if isinstance(completion_text, str):
            result["extracted_answer"] = _extract_answer_tag(completion_text)
        elif completion_text is not None:
            logger.warning(
                f"Completion content is not a string: {type(completion_text)}. Cannot extract answer letter."
            )

        # Handle usage for serializing (Object of type Usage is not JSON serializable)
        result["usage"] = result["usage"].model_dump()

        # Clear reference to large response object to aid garbage collection
        del response

        return result

