"""
VLLM-based model interface for open-source model inference.
"""

import asyncio
import atexit
import json
import logging
import os
import random
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Set environment variables to disable VLLM compilation features
os.environ["VLLM_USE_PRECOMPILED"] = "0"
os.environ["VLLM_DISABLE_CUSTOM_ALL_REDUCE"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn, MofNCompleteColumn, TimeRemainingColumn

# Import batch model interface
from src.batch_model_interface import BatchModelInterface

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
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_BACKOFF = 1.0
DEFAULT_MAX_BACKOFF = 30.0
DEFAULT_CONCURRENCY_LIMIT = 1  # VLLM typically runs single model at a time

# Models that support thinking
THINKING_MODELS = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", 
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "cognitivecomputations/DeepSeek-R1-0528-AWQ",
    "Qwen/QwQ-32B", 
    "Qwen/Qwen3-32B", 
    "Qwen/Qwen3-14B", 
    "Qwen/Qwen3-8B", 
    "Qwen/Qwen3-4B", 
    "Qwen/Qwen3-1.7B", 
    "Qwen/Qwen3-0.6B"
]

# Global registry to track VLLM interface instances for cleanup
_active_vllm_interfaces = []

def cleanup_all_vllm_interfaces():
    """Cleanup all active VLLM interfaces on exit."""
    for interface in _active_vllm_interfaces:
        try:
            interface.close()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    _active_vllm_interfaces.clear()

def signal_handler(signum, frame):
    """Handle signals by cleaning up VLLM interfaces."""
    logger.info(f"Received signal {signum}, cleaning up VLLM interfaces...")
    cleanup_all_vllm_interfaces()
    # Re-raise signal after cleanup
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)

# Register cleanup and signal handlers
atexit.register(cleanup_all_vllm_interfaces)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class VLLMModelInterface(BatchModelInterface):
    """VLLM-based model interface for open-source models."""

    def __init__(
        self,
        models_config: Dict,
        evaluation_config: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        anthropic_api_key_tag: str = "ANTHROPIC_API_KEY",
        batch_size: int = 100,
        max_concurrent_batches: int = 20,
        cache_dir: Optional[Path] = None,
    ):
        # Call parent constructor
        super().__init__(
            models_config=models_config,
            evaluation_config=evaluation_config or {},
            use_cache=use_cache,
            anthropic_api_key_tag=anthropic_api_key_tag,
            batch_size=batch_size,
            max_concurrent_batches=max_concurrent_batches,
        )
        
        self.loaded_models = {}  # Cache for loaded VLLM models
        
        # Override cache dir for VLLM-specific caching
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            self.cache_dir = Path.home() / ".cache" / "inverse-scaling-eval-vllm"
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # Register this interface for cleanup
        _active_vllm_interfaces.append(self)

        self.console.print("✨ VLLM Model interface initialized successfully", style="green")

    def close(self):
        """Cleanup VLLM models and worker processes."""
        if hasattr(self, 'loaded_models'):
            for model_name, model in self.loaded_models.items():
                try:
                    # Try to destroy VLLM model and its worker processes
                    if hasattr(model, 'destroy'):
                        model.destroy()
                    elif hasattr(model, 'stop_remote_worker_execution_loop'):
                        model.stop_remote_worker_execution_loop()
                    logger.info(f"Cleaned up VLLM model: {model_name}")
                except Exception as e:
                    logger.warning(f"Error cleaning up VLLM model {model_name}: {e}")
            self.loaded_models.clear()
        
        # Clear CUDA cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logger.warning(f"Error clearing CUDA cache: {e}")

        # Remove from global registry
        try:
            _active_vllm_interfaces.remove(self)
        except ValueError:
            pass  # Already removed

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        self.close()

    def _load_vllm_model(self, model_name: str, max_model_len: int = 20000) -> Any:
        """Load a VLLM model, caching for reuse."""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
            
        try:
            import torch
            from vllm import LLM, SamplingParams
            
            # Try with compilation disabled first
            try:
                model = LLM(
                    model=model_name, 
                    tokenizer=model_name, 
                    swap_space=50, 
                    gpu_memory_utilization=0.95, 
                    enable_lora=False, 
                    tensor_parallel_size=torch.cuda.device_count(), 
                    max_lora_rank=128, 
                    max_model_len=max_model_len,
                    # Disable torch.compile to avoid compilation errors
                    disable_custom_all_reduce=True,
                    enforce_eager=True,
                    # Use v0 engine to avoid v1 compilation issues
                    use_v2_block_manager=False
                )
                logger.info(f"✅ Loaded VLLM model with compilation disabled: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load with compilation disabled, trying minimal config: {e}")
                # Fallback to minimal configuration
                model = LLM(
                    model=model_name, 
                    tokenizer=model_name, 
                    gpu_memory_utilization=0.8,  # Reduced memory utilization
                    max_model_len=max_model_len,
                    enforce_eager=True,
                    trust_remote_code=True
                )
                logger.info(f"✅ Loaded VLLM model with minimal config: {model_name}")
            
            self.loaded_models[model_name] = model
            return model
        except ImportError as e:
            raise ImportError("VLLM dependencies not available. Please install vllm and torch.") from e
        except Exception as e:
            logger.error(f"Failed to load VLLM model {model_name}: {e}")
            raise

    def _format_vllm_prompt(self, prompt_obj, model_name: str, use_system_prompt: bool = True) -> str:
        """Convert a Prompt object to a string format for VLLM."""
        try:
            model = self.loaded_models.get(model_name)
            if not model:
                # Load model if not cached
                model = self._load_vllm_model(model_name)
                
            tokenizer = model.get_tokenizer()
            
            # Convert our Prompt object to the format expected by apply_chat_template
            messages = []
            for msg in prompt_obj.messages:
                role_str = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
                # Map our roles to standard chat template roles
                if role_str in ['system', 'developer']:
                    role_str = 'system'
                elif role_str == 'user':
                    role_str = 'user'
                elif role_str == 'assistant':
                    role_str = 'assistant'
                    
                messages.append({"role": role_str, "content": msg.content})
            
            # Handle special cases for thinking models
            enable_thinking = False
            if "qwen3" in model_name.lower() and model_name in THINKING_MODELS:
                enable_thinking = True
                
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=enable_thinking if enable_thinking else None
            )
            
            return formatted_prompt
            
        except Exception as e:
            logger.error(f"Error formatting prompt for {model_name}: {e}")
            # Fallback to simple concatenation
            full_text = ""
            for msg in prompt_obj.messages:
                full_text += f"{msg.role}: {msg.content}\n"
            return full_text

    async def evaluate_prompt(
        self,
        model_id: str,
        prompt_text: str,
        possible_answers: List[str],
        reasoning_budget: int = 0,
        icl_examples: Optional[List[Dict[str, Any]]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single prompt using VLLM.

        Args:
            model_id: ID of the model to use
            prompt_text: The prompt to evaluate
            possible_answers: The possible classes/answers (can be empty for open-ended)
            reasoning_budget: Reasoning budget in tokens
            icl_examples: Optional list of in-context learning examples

        Returns:
            Dict containing the model's response and metadata
        """
        if model_id not in self.models_config:
            logger.error(f"Model ID '{model_id}' not found in configuration.")
            raise ValueError(f"Invalid model_id: {model_id}")
        
        model_config = self.models_config[model_id]
        model_name = model_config["model_name"]

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
        prefill_no_think = (model_name in THINKING_MODELS and reasoning_budget == 0)
        prompt = _prepare_prompt(
            prompt_text,
            possible_answers if possible_answers else [],
            model_id,
            full_system_prompt,
            icl_examples=icl_examples,
            models_config=self.models_config,
            prefill_no_think=prefill_no_think
        )

        # Start timer
        start_time = import_time()

        # Configure generation parameters
        temperature = model_config.get("temperature", 0.6)
        max_tokens = _calculate_max_tokens(reasoning_budget, base_tokens=model_config.get("default_max_tokens", 10000))
        
        # Apply special handling for thinking models with 0 budget
        if prefill_no_think and "qwen3" not in model_name.lower():
            # For non-Qwen thinking models, add </think> suffix
            formatted_prompt = self._format_vllm_prompt(prompt, model_name) + "</think>"
        else:
            formatted_prompt = self._format_vllm_prompt(prompt, model_name)

        # Run inference
        try:
            model = self._load_vllm_model(model_name, max_model_len=max_tokens + 2000)
            
            from vllm import SamplingParams
            top_p = model_config.get("top_p", 0.95)
            sampling_params = SamplingParams(
                temperature=temperature, 
                max_tokens=max_tokens, 
                top_p=top_p,
                seed=seed  # Add seed if provided
            )
            
            outputs = model.generate([formatted_prompt], sampling_params)
            response_text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
            
            latency = import_time() - start_time

            # Extract answer using helper
            extracted_answer = _extract_answer_tag(response_text)

            # Extract reasoning content and calculate token usage properly
            tokenizer = model.get_tokenizer()
            reasoning_content = None
            thinking_tokens = 0
            
            if "<answer>" in response_text:
                # Find the last occurrence of <answer> tag
                last_answer_index = response_text.rfind("<answer>")
                if last_answer_index != -1:
                    # Get all text before the last answer tag
                    reasoning_content = response_text[:last_answer_index]
                    # Count tokens in the thinking part
                    thinking_tokens = len(tokenizer.encode(reasoning_content))
            else:
                # If no answer tag, the entire response is reasoning
                reasoning_content = response_text
                thinking_tokens = len(tokenizer.encode(response_text)) if response_text else 0

            # Calculate total output tokens
            total_output_tokens = len(tokenizer.encode(response_text)) if response_text else 0
            
            # Estimate input tokens from the formatted prompt
            input_tokens = len(tokenizer.encode(formatted_prompt)) if formatted_prompt else 0

            # Estimate cost (placeholder - VLLM doesn't have direct cost calculation)
            cost = 0.0

            # Create usage object compatible with BatchModelInterface
            usage_dict = {
                "thinking_tokens": thinking_tokens,
                "total_tokens": total_output_tokens,
                "input_tokens": input_tokens,
                "output_tokens": total_output_tokens
            }

            result = {
                "model": model_id,
                "reasoning_budget": reasoning_budget,
                "latency": latency,
                "error": None,
                "response": response_text,
                "extracted_answer": extracted_answer,
                "cost": cost,
                "stop_reason": outputs[0].outputs[0].finish_reason if outputs and outputs[0].outputs else None,
                "input_tokens": input_tokens,
                "output_tokens": total_output_tokens,
                "reasoning_content": reasoning_content,
                "usage": usage_dict,
            }

            logger.debug(f"✅ Successfully evaluated prompt with {model_id} (budget: {reasoning_budget})")
            if result['extracted_answer']:
                logger.debug(f"📝 Answer: [green]{result['extracted_answer']}[/]")
            logger.debug(f"⏱️  Latency: [yellow]{result['latency']:.2f}s[/]")

            return result

        except Exception as e:
            latency = import_time() - start_time
            error_msg = f"VLLM inference failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            return {
                "model": model_id,
                "reasoning_budget": reasoning_budget,
                "latency": latency,
                "error": error_msg,
                "response": None,
                "extracted_answer": None,
                "cost": 0.0,
                "stop_reason": None,
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_content": None,
                "usage": {
                    "thinking_tokens": 0,
                    "total_tokens": 0,
                    "input_tokens": 0,
                    "output_tokens": 0
                },
            }

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
        Evaluate multiple prompts using VLLM batch processing.

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

        # Validate ICL list length if provided
        if icl_examples_list is not None and len(icl_examples_list) != len(prompt_texts):
            raise ValueError("Length of icl_examples_list must match length of prompt_texts")

        # If icl_examples_list is None, create a list of Nones for easier processing
        if icl_examples_list is None:
            icl_examples_list = [None] * len(prompt_texts)

        # VLLM batch processing - prepare all prompts first
        model_config = self.models_config[model_id]
        model_name = model_config["model_name"]
        
        formatted_prompts = []
        metadata_list = []
        
        for i in range(len(prompt_texts)):
            # Determine if it's multiple choice
            is_multiple_choice = bool(possible_answers_list[i])

            # Get the reasoning part of the system prompt
            reasoning_prompt_part = _get_reasoning_system_prompt(
                reasoning_budgets[i],
                is_multiple_choice,
                prompt_use_all_budget=self.evaluation_config.get("prompt_use_all_budget", True),
                model_id=model_id,
                models_config=self.models_config,
            )

            # Combine with base system prompt from config
            base_system_prompt = model_config.get("system_prompt", "")
            full_system_prompt = base_system_prompt + reasoning_prompt_part

            # Prepare prompt using the helper
            prefill_no_think = (model_name in THINKING_MODELS and reasoning_budgets[i] == 0)
            prompt = _prepare_prompt(
                prompt_texts[i],
                possible_answers_list[i] if possible_answers_list[i] else [],
                model_id,
                full_system_prompt,
                icl_examples=icl_examples_list[i],
                models_config=self.models_config,
                prefill_no_think=prefill_no_think
            )

            # Format for VLLM
            if prefill_no_think and "qwen3" not in model_name.lower():
                formatted_prompt = self._format_vllm_prompt(prompt, model_name) + "</think>"
            else:
                formatted_prompt = self._format_vllm_prompt(prompt, model_name)
                
            formatted_prompts.append(formatted_prompt)
            metadata_list.append({
                "reasoning_budget": reasoning_budgets[i],
                "index": i
            })

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
                transient=False
            )
            use_external_progress = False
        else:
            progress_context = external_progress
            use_external_progress = True

        results = []
        start_time = import_time()

        with progress_context as progress:
            if not use_external_progress:
                batch_task = progress.add_task(
                    f"[cyan]Running VLLM batch inference for {model_id} ({len(formatted_prompts)} prompts)",
                    total=len(formatted_prompts)
                )

            try:
                # Load model
                max_tokens = max(_calculate_max_tokens(budget, base_tokens=model_config.get("default_max_tokens", 10000)) 
                               for budget in reasoning_budgets)
                model = self._load_vllm_model(model_name, max_model_len=max_tokens + 2000)
                
                # Configure sampling parameters
                from vllm import SamplingParams
                temperature = model_config.get("temperature", 0.6)
                top_p = model_config.get("top_p", 0.95)
                
                # Check if we have seeds and if they're all the same
                seed = None
                if seeds_list and any(s is not None for s in seeds_list):
                    unique_seeds = set(s for s in seeds_list if s is not None)
                    if len(unique_seeds) == 1:
                        # All prompts have the same seed, we can use it
                        seed = unique_seeds.pop()
                    elif len(unique_seeds) > 1:
                        logger.warning("VLLM batch processing does not support different seeds per prompt. Using first non-None seed.")
                        seed = next(s for s in seeds_list if s is not None)
                
                # Use the maximum tokens needed across all prompts
                sampling_params = SamplingParams(
                    temperature=temperature, 
                    max_tokens=max_tokens, 
                    top_p=top_p,
                    seed=seed  # Add seed if available
                )
                
                # Run batch inference
                outputs = model.generate(formatted_prompts, sampling_params)
                
                # Process results
                tokenizer = model.get_tokenizer()
                for i, (output, metadata) in enumerate(zip(outputs, metadata_list)):
                    response_text = output.outputs[0].text if output.outputs else ""
                    extracted_answer = _extract_answer_tag(response_text)
                    
                    # Extract reasoning content and calculate token usage properly
                    reasoning_content = None
                    thinking_tokens = 0
                    
                    if "<answer>" in response_text:
                        # Find the last occurrence of <answer> tag
                        last_answer_index = response_text.rfind("<answer>")
                        if last_answer_index != -1:
                            # Get all text before the last answer tag
                            reasoning_content = response_text[:last_answer_index]
                            # Count tokens in the thinking part
                            thinking_tokens = len(tokenizer.encode(reasoning_content))
                    else:
                        # If no answer tag, the entire response is reasoning
                        reasoning_content = response_text
                        thinking_tokens = len(tokenizer.encode(response_text)) if response_text else 0

                    # Calculate total output tokens
                    total_output_tokens = len(tokenizer.encode(response_text)) if response_text else 0
                    
                    # Calculate input tokens from the corresponding formatted prompt
                    input_tokens = len(tokenizer.encode(formatted_prompts[i])) if i < len(formatted_prompts) else 0
                    
                    # Create usage dict compatible with BatchModelInterface
                    usage_dict = {
                        "thinking_tokens": thinking_tokens,
                        "total_tokens": total_output_tokens,
                        "input_tokens": input_tokens,
                        "output_tokens": total_output_tokens
                    }
                    
                    result = {
                        "model": model_id,
                        "reasoning_budget": metadata["reasoning_budget"],
                        "latency": import_time() - start_time,  # Shared latency for batch
                        "error": None,
                        "response": response_text,
                        "extracted_answer": extracted_answer,
                        "cost": 0.0,  # Placeholder
                        "stop_reason": output.outputs[0].finish_reason if output.outputs else None,
                        "input_tokens": input_tokens,
                        "output_tokens": total_output_tokens,
                        "reasoning_content": reasoning_content,
                        "usage": usage_dict,
                    }
                    
                    results.append(result)
                    
                    if not use_external_progress:
                        progress.advance(batch_task)

            except Exception as e:
                logger.error(f"❌ VLLM batch inference failed: {e}", exc_info=True)
                # Create error results for all prompts
                for i, metadata in enumerate(metadata_list):
                    error_result = {
                        "model": model_id,
                        "reasoning_budget": metadata["reasoning_budget"],
                        "latency": import_time() - start_time,
                        "error": f"Batch inference failed: {str(e)}",
                        "response": None,
                        "extracted_answer": None,
                        "cost": 0.0,
                        "stop_reason": None,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "reasoning_content": None,
                        "usage": {
                            "thinking_tokens": 0,
                            "total_tokens": 0,
                            "input_tokens": 0,
                            "output_tokens": 0
                        },
                    }
                    results.append(error_result)
                    
                    if not use_external_progress:
                        progress.advance(batch_task)

            if not use_external_progress:
                progress.update(batch_task, completed=len(formatted_prompts))

        logger.info(f"Finished VLLM batch evaluation for {model_id}. Got {len(results)} results.")
        return results