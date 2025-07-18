"""
Interface for accessing different reasoning models through the safetytooling APIs.
"""

import asyncio
import logging
import random # For jitter
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn, MofNCompleteColumn, TimeRemainingColumn
from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils

# Import shared helper functions
from src.utils.model_helpers import (
    _calculate_max_tokens,
    _extract_answer_tag,
    _get_reasoning_params,
    _get_reasoning_system_prompt,
    _prepare_prompt,
    import_time,
)

# Set up rich console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

# Default retry parameters
DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_BACKOFF = 1.0 # seconds
DEFAULT_MAX_BACKOFF = 60.0 # seconds
DEFAULT_CONCURRENCY_LIMIT = 20 # Default max concurrent requests for non-batch interfaces

class ModelInterface(ABC):
    """Base interface for accessing different reasoning models."""

    def __init__(
        self,
        models_config: Dict,
        use_cache: bool = True,
        anthropic_api_key_tag: str = "ANTHROPIC_API_KEY",
        evaluation_config: Optional[Dict[str, Any]] = None,
    ):
        self.models_config = models_config
        self.anthropic_api_key_tag = anthropic_api_key_tag
        self.console = console
        self.evaluation_config = evaluation_config or {}

        self.batch_size = DEFAULT_CONCURRENCY_LIMIT

        # Initialize API with caching
        self.cache_dir = Path.home() / ".cache" / "inverse-scaling-eval"
        self.prompt_history_dir = (
            Path.home() / ".prompt_history" / "inverse-scaling-eval"
        )
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.prompt_history_dir.mkdir(exist_ok=True, parents=True)

        # Load environment with appropriate API key
        utils.setup_environment(anthropic_tag=anthropic_api_key_tag)

        self.api = InferenceAPI(
            cache_dir=self.cache_dir if use_cache else None,
            prompt_history_dir=self.prompt_history_dir,
            print_prompt_and_response=False,
        )

        self.console.print("✨ Model interface initialized successfully", style="green")

    async def evaluate_prompt(
        self,
        model_id: str,
        prompt_text: str,
        possible_answers: List[str],
        reasoning_budget: int = 0,
        icl_examples: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single prompt using the standard InferenceAPI.

        Args:
            model_id: ID of the model to use
            prompt_text: The prompt to evaluate
            possible_answers: The possible classes/answers (can be empty for open-ended)
            reasoning_budget: Reasoning budget in tokens
            icl_examples: Optional list of in-context learning examples

        Returns:
            Dict containing the model's response and metadata
        """
        # Removed Progress bar context manager
        # task = progress.add_task(
        #     f"[cyan]Evaluating prompt with {model_id} (budget: {reasoning_budget})",
        #     total=None,
        # )

        if model_id not in self.models_config:
            logger.error(f"Model ID '{model_id}' not found in configuration.")
            # progress.update(task, completed=1, visible=False) # Removed
            raise ValueError(f"Invalid model_id: {model_id}")
        model_config = self.models_config[model_id]

        # Determine if it's multiple choice
        is_multiple_choice = bool(possible_answers)

        # Get the reasoning part of the system prompt using the helper
        reasoning_prompt_part = _get_reasoning_system_prompt(
            reasoning_budget,
            is_multiple_choice,
            prompt_use_all_budget=self.evaluation_config.get("prompt_use_all_budget", True),
            model_id=model_id,
            models_config=self.models_config,
        )

        # Combine with base system prompt from config
        base_system_prompt = model_config.get("system_prompt", "")
        full_system_prompt = base_system_prompt + reasoning_prompt_part

        # Prepare prompt using the helper
        # Ensure possible_answers is a list, even if empty
        prompt = _prepare_prompt(
            prompt_text,
            possible_answers if possible_answers else [],
            model_id,
            full_system_prompt,
            icl_examples=icl_examples,
            models_config=self.models_config,
            prefill_no_think=(model_id in ["deepseek-reasoner"] and reasoning_budget == 0)
        )

        # Start timer
        start_time = import_time()

        # Base API parameters from config
        kwargs = {**model_config.get("api_params", {})}

        # Add model-specific reasoning parameters using the helper
        if is_thinking_model and reasoning_budget != 0:
            reasoning_params = _get_reasoning_params(model_id, reasoning_budget, self.models_config)
            kwargs.update(reasoning_params)

        # Configure API parameters
        is_thinking_model = self.evaluation_config.get("thinking", True)

        if not is_thinking_model:
            max_tokens = reasoning_budget if reasoning_budget > 0 else model_config.get("default_max_tokens", 10000)
            if reasoning_budget == 0:
                logger.warning(f"Budget is 0 for non-thinking model {model_id}. Using default max_tokens: {max_tokens}")
        else:
            if model_config.get("type", None) == "openai":
                max_tokens = _calculate_max_tokens(reasoning_budget, base_tokens=10000)
                # Use max_completion_tokens for OpenAI models if needed
                kwargs["max_completion_tokens"] = max_tokens
            else:
                max_tokens = _calculate_max_tokens(reasoning_budget, base_tokens=10000)

        # --- Retry Logic ---
        max_retries = model_config.get("max_retries", DEFAULT_MAX_RETRIES)
        current_backoff = model_config.get("initial_backoff", DEFAULT_INITIAL_BACKOFF)
        max_backoff = model_config.get("max_backoff", DEFAULT_MAX_BACKOFF)
        last_exception = None

        for attempt in range(max_retries):
            try:
                # progress.update(task, description=f"[cyan]Evaluating {model_id} (budget: {reasoning_budget}) attempt {attempt+1}/{max_retries}") # Removed
                # Log attempt start instead
                logger.debug(f"Attempt {attempt+1}/{max_retries} for {model_id} (budget: {reasoning_budget})")
                # Call the standard InferenceAPI
                responses = await self.api(
                    model_id=model_config["model_name"],
                    prompt=prompt,
                    temperature=model_config.get("temperature", 0.0), # Default temp 0
                    max_tokens=max_tokens,
                    **kwargs,
                )

                # Process the first response (standard API usually returns one)
                response = responses[0] if responses else None
                latency = import_time() - start_time

                # Process response using similar logic to batch interface
                processed_result = await self.process_response(response, model_id, reasoning_budget, start_time)

                if processed_result.get("error") is None:
                    # Success, break retry loop
                    completion = processed_result.get("response")
                    extracted_answer = processed_result.get("extracted_answer")
                    cost = processed_result.get("cost", 0)
                    reasoning_content = processed_result.get("reasoning_content")
                    error = None
                    last_exception = None
                    break
                else:
                    # Error in processing, treat as retriable
                    completion = None
                    extracted_answer = None
                    cost = 0
                    reasoning_content = None
                    error = processed_result.get("error")
                    last_exception = RuntimeError(f"Processing Error: {error}")
                    logger.warning(f"Processing error on attempt {attempt+1} for {model_id}: {error}. Retrying...")

            except Exception as e:
                latency = import_time() - start_time # Update latency even on exception
                last_exception = e
                logger.warning(
                    f"Exception on attempt {attempt + 1} for {model_id}: {str(e)}. Retrying..."
                )
                # Fallthrough to backoff logic

            # If we are here, it means an error occurred or API call failed
            if attempt < max_retries - 1:
                # Calculate backoff with jitter
                jitter = random.uniform(0, current_backoff * 0.1) # 10% jitter
                sleep_time = current_backoff + jitter
                logger.info(f"Sleeping for {sleep_time:.2f} seconds before retry {attempt+2}")
                await asyncio.sleep(sleep_time)
                # Exponential backoff
                current_backoff = min(current_backoff * 2, max_backoff)
            else:
                logger.error(f"Max retries ({max_retries}) reached for {model_id}.")
                # Ensure error is set based on the last exception
                error = f"Max retries reached. Last error: {str(last_exception)}"
                completion = None
                extracted_answer = None
                cost = 0

        # Construct result outside the loop with consistent structure
        result = {
            "model": model_id,
            "reasoning_budget": reasoning_budget,
            "response": completion,
            "extracted_answer": extracted_answer,
            "latency": latency,
            "cost": cost,
            "error": error,
            "reasoning_content": reasoning_content if 'reasoning_content' in locals() else None,
            "stop_reason": getattr(response, "stop_reason", None) if 'response' in locals() and response else None,
            "input_tokens": getattr(response, "input_tokens", None) if 'response' in locals() and response else None,
            "output_tokens": getattr(response, "output_tokens", None) if 'response' in locals() and response else None,
            "usage": getattr(response, "usage", None) if 'response' in locals() and response else None,
        }

        # Handle usage serialization if present
        if result["usage"] and hasattr(result["usage"], "model_dump"):
            result["usage"] = result["usage"].model_dump()

        # progress.update(task, completed=1, visible=False) # Removed
        # Log final status after retries
        if result["error"]:
            logger.error(f"❌ Failed evaluating prompt with {model_id} after {max_retries} attempts: {result['error']}")
        else:
            logger.debug(f"✅ Successfully evaluated prompt with {model_id} (budget: {reasoning_budget})")
            if result['extracted_answer']:
                logger.debug(f"📝 Answer: [green]{result['extracted_answer']}[/]")
            logger.debug(f"⏱️  Latency: [yellow]{result['latency']:.2f}s[/] | 💰 Cost: [blue]${result['cost']:.4f}[/]")

        return result

    async def process_response(
        self,
        response,
        model_id: str,
        reasoning_budget: int,
        start_time: float,
    ) -> Dict[str, Any]:
        """
        Process a single response object from the InferenceAPI.

        Args:
            response: The response object from InferenceAPI
            model_id: Model identifier
            reasoning_budget: Reasoning budget in tokens
            start_time: Start time for latency calculation

        Returns:
            Processed response dictionary compatible with batch interface format.
        """
        end_time = import_time()
        latency = end_time - start_time

        # Handle cases where response might be None
        if response is None:
            logger.warning(
                f"Received None response object for budget {reasoning_budget}."
            )
            return {
                "model": model_id,
                "reasoning_budget": reasoning_budget,
                "latency": latency,
                "error": "No response received from API",
                "response": None,
                "extracted_answer": None,
                "cost": 0,
                "reasoning_content": None,
            }

        # Extract fields from response object, handling potential missing attributes
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

        # Check for API-level errors first
        if hasattr(response, "error") and response.error:
            result["error"] = response.error
            return result

        # Get the completion text, handling potential different response formats
        completion_text = getattr(response, "completion", None)
        if completion_text is None:
            result["error"] = "No completion text found in response"
        elif not isinstance(completion_text, str):
            # Handle case where completion might be a complex object
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
        else:
            # completion_text is already a string, clean it up
            completion_text = completion_text.strip()

        # Update result with processed completion text
        result["response"] = completion_text

        # Extract answer from the completion text if available
        if isinstance(completion_text, str) and completion_text:
            result["extracted_answer"] = _extract_answer_tag(completion_text)
        elif completion_text is not None:
            logger.warning(
                f"Completion content is not a string: {type(completion_text)}. Cannot extract answer."
            )

        return result

    async def _evaluate_prompt_with_semaphore(self, semaphore: asyncio.Semaphore, model_id: str, prompt_text: str, possible_answers: List[str], reasoning_budget: int, icl_examples: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Helper to wrap evaluate_prompt call with semaphore acquisition/release."""
        async with semaphore:
            return await self.evaluate_prompt(
                model_id,
                prompt_text,
                possible_answers,
                reasoning_budget,
                icl_examples
            )

    async def evaluate_prompts_batch(
        self,
        model_id: str,
        prompt_texts: List[str],
        possible_answers_list: List[List[str]],
        reasoning_budgets: List[int],
        external_progress: Optional[Progress] = None,
        icl_examples_list: Optional[List[Optional[List[Dict[str, Any]]]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple prompts concurrently using the standard InferenceAPI.
        This is the default implementation for models not using a specific batch API.

        Args:
            model_id: ID of the model to use
            prompt_texts: List of prompts to evaluate
            possible_answers_list: List of possible answers for each prompt
            reasoning_budgets: List of reasoning budgets, one per prompt
            external_progress: Optional external Rich progress bar to integrate with
            icl_examples_list: Optional list of lists of ICL examples, one per prompt

        Returns:
            List of dictionaries containing model responses and metadata
        """
        if not (len(prompt_texts) == len(possible_answers_list) == len(reasoning_budgets)):
            raise ValueError("Input lists must have the same length")

        if model_id not in self.models_config:
             raise ValueError(f"Invalid model_id: {model_id}")
        model_config = self.models_config[model_id]

        # Validate ICL list length if provided
        if icl_examples_list is not None and len(icl_examples_list) != len(prompt_texts):
            raise ValueError("Length of icl_examples_list must match length of prompt_texts")

        # If icl_examples_list is None, create a list of Nones for easier zipping
        if icl_examples_list is None:
            icl_examples_list = [None] * len(prompt_texts)

        # Get concurrency limit from config or use default
        concurrency_limit = model_config.get("concurrency_limit", DEFAULT_CONCURRENCY_LIMIT)
        semaphore = asyncio.Semaphore(concurrency_limit)

        # Create tasks for concurrent evaluation
        tasks = []
        for i in range(len(prompt_texts)):
            task = asyncio.create_task(
                self._evaluate_prompt_with_semaphore(
                    semaphore,
                    model_id,
                    prompt_texts[i],
                    possible_answers_list[i],
                    reasoning_budgets[i],
                    icl_examples_list[i]
                )
            )
            tasks.append(task)

        # Set up progress bar if not provided externally
        if external_progress is None:
            progress_context = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
                transient=False # Keep visible until all tasks are done
            )
            use_external_progress = False
        else:
            progress_context = external_progress
            use_external_progress = True

        results = []
        all_results = [None] * len(tasks) # Pre-allocate list to maintain order
        with progress_context as progress:
            if not use_external_progress:
                batch_task = progress.add_task(
                    f"[cyan]Evaluating {len(tasks)} prompts concurrently for {model_id} (limit: {concurrency_limit})",
                    total=len(tasks)
                )

            # Use asyncio.gather to run all tasks concurrently and wait for all
            # Set return_exceptions=True to get exceptions instead of raising them immediately
            gathered_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results, maintaining order
            for i, result_or_exception in enumerate(gathered_results):
                if isinstance(result_or_exception, Exception):
                    # Log the error and create a placeholder error dictionary
                    logger.error(f"Error evaluating prompt index {i} for {model_id}: {result_or_exception}", exc_info=True)
                    all_results[i] = {
                        "model": model_id,
                        "reasoning_budget": reasoning_budgets[i], # Get budget from original list
                        "error": f"Concurrency failure: {str(result_or_exception)}",
                        "response": None,
                        "extracted_answer": None,
                        "latency": 0, # Latency might be unknown if exception happened early
                        "cost": 0,
                        "reasoning_content": None,
                        "stop_reason": None,
                        "input_tokens": None,
                        "output_tokens": None,
                        "usage": None,
                    }
                else:
                    # Store the successful result
                    all_results[i] = result_or_exception

                # Advance progress bar after processing each result
                if not use_external_progress:
                    progress.advance(batch_task)

            # Ensure progress bar completes if gather finished early
            if not use_external_progress:
                progress.update(batch_task, completed=len(tasks))

        logger.info(f"Finished concurrent evaluation for {model_id}. Got {len(all_results)} results.")
        return all_results
