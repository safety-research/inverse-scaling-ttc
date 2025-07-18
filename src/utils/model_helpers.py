import re
import time
from typing import Any, Dict, List, Optional

from safetytooling.data_models import ChatMessage, MessageRole, Prompt

USE_DEVELOPER_ROLE_MODELS = ["o1-mini-2024-09-12", "o1-2024-12-17", "o3-mini-2025-01-31", "o4-mini-2025-04-16", "o3-2025-04-16"]

def import_time():
    """Import time for measuring latency."""
    return time.time()

def _format_prompt_content(prompt_text: str, possible_answers: Optional[List[str]]) -> str:
    """Helper to format prompt text with MC options and answer instruction."""
    content = prompt_text
    is_multiple_choice = bool(possible_answers)

    if is_multiple_choice:
        content += "\n\n"
        for i, answer in enumerate(possible_answers):
            letter = chr(65 + i)  # A, B, C, D, etc.
            content += f"{letter}. {answer}\n"

    # Add instruction for answer format
    if is_multiple_choice:
        content += "\n\nProvide your final answer as a single letter in the format <answer>X</answer>, where X is your chosen option."
    else:
        content += "\n\nProvide your final answer in the format <answer>X</answer>, where X is the final answer."

    return content

def _prepare_prompt(
    prompt_text: str,
    possible_answers: List[str],
    model_id: str,
    system_prompt: Optional[str] = "",
    icl_examples: Optional[List[Dict[str, Any]]] = None,
    models_config: Optional[Dict] = None,
    prefill_no_think: bool = False,
) -> Prompt:
    """
    Prepare a single prompt object.

    Args:
        prompt_text: The text prompt
        possible_answers: List of possible answers (can be empty for open-ended)
        model_id: model ID
        system_prompt: Optional system prompt
        prefill_no_think: Whether to add an empty <think> tag for assistant prefill
        icl_examples: Optional list of dictionaries, each representing an in-context example
        models_config: Optional dictionary containing configurations for different models.
    Returns:
        Formatted Prompt object
    """
    messages = []
    processed_system_prompt = system_prompt if system_prompt else ""

    # Determine model type
    model_type = "default"
    if models_config and model_id in models_config:
        model_type = models_config[model_id].get("type", "default")

    # --- Add In-Context Learning Examples --- #
    if icl_examples:
        # Append examples to the system prompt for all models
        example_str_parts = []
        for example in icl_examples:
            try:
                # 1. Format Example User Prompt
                example_prompt_text = example.get("prompt", "")
                example_classes = example.get("classes") # Can be None
                example_user_content = _format_prompt_content(example_prompt_text, example_classes)

                # 2. Format Example Assistant Answer
                example_assistant_content = ""
                if example_classes and "answer_index" in example:
                    correct_index = int(example["answer_index"])
                    if 0 <= correct_index < len(example_classes):
                        correct_letter = chr(65 + correct_index)
                        example_assistant_content = f"<answer>{correct_letter}</answer>"
                    else:
                        print(f"ICL Example for system prompt has invalid answer_index {correct_index}. Skipping example.")
                        continue # Skip this example
                elif "answer" in example: # Open-ended example
                    example_assistant_content = f"<answer>{example['answer']}</answer>"
                else:
                    print("ICL Example for system prompt is missing required fields ('classes'/'answer_index' or 'answer'). Skipping example.")
                    continue # Skip this example

                example_str_parts.append(f"Example:\nUser: {example_user_content}\nAssistant: <thinking>Your thinking process...</thinking> {example_assistant_content}")

            except (ValueError, TypeError) as e:
                print(f"Skipping ICL example for system prompt due to error: {e}")
                continue # Skip this example

        if example_str_parts:
            processed_system_prompt += "\n\n" + "\n\n".join(example_str_parts)

    # --- Add System Prompt ---
    if processed_system_prompt:
        # Determine role based on model_id convention or default to system
        role = MessageRole.developer if model_id in USE_DEVELOPER_ROLE_MODELS  else MessageRole.system
        messages.append(ChatMessage(role=role, content=processed_system_prompt))

    # --- Add Final Task Prompt ---
    final_task_prompt = _format_prompt_content(prompt_text, possible_answers)
    messages.append(ChatMessage(role=MessageRole.user, content=final_task_prompt))

    # --- Add Assistant Prefill (Optional) ---
    if prefill_no_think:
        # Useful for r1 models to avoid thinking when the budget is 0
        messages.append(ChatMessage(role=MessageRole.assistant, content="<think></think>", is_prefix=True))

    return Prompt(messages=messages)

def _extract_answer_tag(text: str) -> Optional[str]:
    """Extract answer from the last <answer>X</answer> tag, supporting letters or full text."""
    if not isinstance(text, str):
        return None

    # Try to find the last content between <answer> and </answer> tags
    # Using a more robust pattern that handles nested tags
    pattern = r"<answer>(?:(?!</?answer>).)*?</answer>"
    matches = list(re.finditer(pattern, text, re.IGNORECASE | re.DOTALL))
    if matches:
        # Get the last match
        last_match = matches[-1]
        # Extract the content between the tags using a capture group
        answer = re.search(r"<answer>(.*?)</answer>", last_match.group(0), re.IGNORECASE | re.DOTALL)
        if answer:
            answer = answer.group(1).strip()
            # If it's a single letter, treat it as multiple choice and uppercase it
            if len(answer) == 1 and answer.isalpha():
                return answer.upper()
            # Otherwise return the potentially multi-line answer as is
            elif answer:
                return answer
            else:
                # Handle empty tag case
                return None

    # Fallback: Try to find any single letter answer (A, B, C, etc.) for MCQs
    match = re.search(
        r"(?:answer|option|choose|select|pick)(?:\s+is|\s*:\s*)?\s+([A-Z])(?:\.|,|\s|$)",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()

    # Fallback: Look for any isolated letter that could be an answer
    match = re.search(r"\b([A-Z])\b(?![.-])", text) # Avoid matching initials like A. or B-
    if match:
        # Check context to avoid grabbing random letters (e.g., "Section A")
        preceding_text = text[:match.start()].lower()
        if not any(kw in preceding_text[-20:] for kw in ["section", "part", "appendix", "figure", "table"]):
             return match.group(1).upper()

    return None


def _get_reasoning_system_prompt(
    reasoning_budget: int,
    is_multiple_choice: bool,
    prompt_use_all_budget: bool,
    model_id: Optional[str] = None, # Optional: For future model-specific tweaks
    models_config: Optional[Dict] = None # Optional: For future model-specific tweaks
) -> str:
    """Get system prompt content related to reasoning budget and answering format."""

    if is_multiple_choice:
        answering_instruction = "select your answer from the provided options"
    else:
        answering_instruction = "provide your answer"

    if reasoning_budget == 0:
        # Use a simple instruction for zero budget, focusing on the format
        return f"\nDon't think. Directly {answering_instruction} and clearly indicate your final answer using <answer>X</answer> format."

    # Determine thinking intensity keyword based on budget
    if reasoning_budget == 0:
        thinking_keyword = "don't think"
    elif 0 < reasoning_budget < 2048:
        thinking_keyword = "think"
    elif 2048 <= reasoning_budget < 8192:
        thinking_keyword = "think harder"
    elif reasoning_budget >= 8192:
        thinking_keyword = "ultrathink"

    forced_thinking_token_budget_instruction = ""
    if reasoning_budget > 0:
        forced_thinking_token_budget_instruction = f"You have a thinking token budget of about {reasoning_budget} tokens. "

        if prompt_use_all_budget:
            forced_thinking_token_budget_instruction += f"(IMPORTANT: {thinking_keyword.capitalize()}! YOU SHOULD AIM TO USE A SIGNIFICANT PORTION OF YOUR THINKING BUDGET BEFORE ANSWERING)." # Adjusted instruction
        else:
            forced_thinking_token_budget_instruction += "(You don't need to use all of your thinking budget before answering)."

    # Construct the reasoning prompt part
    reasoning_prompt_part = (
        f"\nUse a thinking process to analyze the problem step-by-step. "
        f"{forced_thinking_token_budget_instruction}"
    ).strip()

    # Combine reasoning part with the answering instruction
    reasoning_system_prompt = (
        f"{reasoning_prompt_part}\n"
        f"After your thinking process, {answering_instruction} and clearly indicate your final answer using <answer>X</answer> format."
    )

    return reasoning_system_prompt

def _get_reasoning_params(
    model_id: str, reasoning_budget: int, models_config: Dict
) -> Dict[str, Any]:
    """Get model-specific API parameters for reasoning, based on models_config."""
    if model_id not in models_config:
        # Return empty dict if model_id is somehow unknown
        return {}

    model_config = models_config[model_id]
    model_type = model_config.get("type", "default")

    if reasoning_budget == 0:
        # No reasoning params needed if budget is zero or less
        return {}

    elif reasoning_budget < 0:
        # Natural overthinking
        if model_type == "anthropic":
            return {"thinking": {"type": "enabled", "budget_tokens": 16384}}
        elif model_type == "openai" and not model_id.startswith("o1-mini") and not model_id.startswith("o1-preview") and not model_id.startswith("gpt-4.1-mini"):
            return {"reasoning_effort": "high"}
        else:
            return {}
    else:
        # Return parameters based on model type defined in config
        if model_type == "anthropic":
            if model_id in ["claude-3-7-sonnet-20250219", "claude-sonnet-4-20250514", "claude-opus-4-20250514"]:
                return {"thinking": {"type": "enabled", "budget_tokens": reasoning_budget}}
            else:
                return {}
        elif model_type == "openai" and not model_id.startswith("o1-mini") and not model_id.startswith("o1-preview") and not model_id.startswith("gpt-4.1-mini"):
            # Assuming 'openai' type signifies a model compatible with 'reasoning_budget' param

            if 1024 <= reasoning_budget < 4096:
                reasoning_effort = "low"
            elif 4096 <= reasoning_budget < 8192:
                reasoning_effort = "medium"
            elif reasoning_budget >= 8192:
                reasoning_effort = "high"
            else:
                raise ValueError(f"Invalid reasoning budget: {reasoning_budget}")
            return {"reasoning_effort": reasoning_effort}
        # Add other model types here if they have specific reasoning params
        # elif model_type == "some_other_type":
        #     return {"specific_param": value}
        else:
            # Default for unknown or standard models - no special parameters
            return {}

def _calculate_max_tokens(reasoning_budget: int, base_tokens: int = 1024) -> int:
    """
    Calculate the maximum tokens for a response.

    Args:
        reasoning_budget: The reasoning budget in tokens.
        base_tokens: Base tokens allocated for the answer structure itself.

    Returns:
        The maximum tokens allowed for the response.
    """
    # Ensure budget is not negative for calculation
    if reasoning_budget < 0:
        return 16384 + base_tokens
    else:
        effective_budget = max(0, reasoning_budget)
        return effective_budget + base_tokens
