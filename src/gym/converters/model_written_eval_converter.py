"""
Converter for Model Written Evaluation datasets.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, List

from ..base.base_converter import BaseConverter
from ..base.data_models import TaskInstance


class ModelWrittenEvalConverter(BaseConverter):
    """
    Converter for Model Written Evaluation JSONL files.
    
    Handles both yes/no questions and multiple choice questions with automatic format detection.
    """

    def __init__(self, config):
        super().__init__(config)
        # Updated pattern to handle A-Z choices (supports up to 26 options)
        self.choice_pattern = re.compile(r"\(([A-Z])\)\s*(.*?)(?=\n\s*\([A-Z]\)|$)", re.DOTALL)

    def load_source_data(self) -> Iterator[Dict[str, Any]]:
        """
        Load Model Written Eval data from JSONL files.
        
        Yields:
            Raw data dictionaries with file context
        """
        input_path = Path(self.config.input_source)

        if input_path.is_file() and input_path.suffix == ".jsonl":
            # Single file
            files_to_process = [input_path]
        elif input_path.is_dir():
            # Directory of JSONL files
            files_to_process = list(input_path.glob("*.jsonl"))
        else:
            raise ValueError(f"Input source must be a JSONL file or directory containing JSONL files: {input_path}")

        self.logger.info(f"Found {len(files_to_process)} JSONL files to process")

        for jsonl_file in files_to_process:
            try:
                file_type = self._detect_file_type(jsonl_file)
                self.logger.info(f"Detected file type '{file_type}' for {jsonl_file.name}")

                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            data = json.loads(line)
                            data["_file_info"] = {
                                "filename": jsonl_file.name,
                                "file_type": file_type,
                                "line_number": line_num
                            }
                            yield data
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Invalid JSON on line {line_num} in {jsonl_file}: {e}")

            except Exception as e:
                self.logger.error(f"Error processing file {jsonl_file}: {e}")

    def _detect_file_type(self, jsonl_file: Path) -> str:
        """
        Detect if file contains yes/no or multiple choice questions.
        
        Args:
            jsonl_file: Path to JSONL file
            
        Returns:
            "yes_no" or "multiple_choice"
        """
        sample_size = 10
        yes_no_count = 0
        mc_count = 0

        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= sample_size:
                        break

                    try:
                        data = json.loads(line.strip())
                        question = data.get("question", "")
                        answer = data.get("answer_matching_behavior", "")

                        # Check for yes/no pattern
                        if answer.lower().strip() in ["yes", "no"]:
                            yes_no_count += 1

                        # Check for multiple choice pattern (updated to handle A-Z choices)
                        if ("\n\nChoices\n" in question or "\nChoices:\n" in question or 
                            "\n (A)" in question or re.search(r"\([A-Z]\)", question)):
                            mc_count += 1
                        
                        # Also check if answer is in format " (A)", " (B)", etc.
                        if re.match(r"^\s*\([A-Z]\)\s*$", answer):
                            mc_count += 1

                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            self.logger.warning(f"Error detecting file type for {jsonl_file}: {e}")

        # Return the predominant type
        self.logger.info(f"File: {jsonl_file.name} - Yes/No count: {yes_no_count}, Multiple Choice count: {mc_count}")
        return "yes_no" if yes_no_count >= mc_count else "multiple_choice"

    def convert_instance(self, raw_data: Dict[str, Any]) -> Optional[TaskInstance]:
        """
        Convert a single Model Written Eval example to TaskInstance.
        
        Args:
            raw_data: Raw data dictionary with file info
            
        Returns:
            TaskInstance with converted data
        """
        file_info = raw_data.get("_file_info", {})
        file_type = file_info.get("file_type", "multiple_choice")

        if file_type == "yes_no":
            return self._convert_yes_no_instance(raw_data)
        else:
            return self._convert_multiple_choice_instance(raw_data)

    def _convert_yes_no_instance(self, raw_data: Dict[str, Any]) -> Optional[TaskInstance]:
        """Convert yes/no question instance."""
        question = raw_data.get("question", "")
        answer = raw_data.get("answer_matching_behavior", "").strip()

        if not question or not answer:
            self.logger.warning("Missing question or answer in yes/no entry")
            return None

        # Validate answer
        if answer.lower() not in ["yes", "no"]:
            self.logger.warning(f"Invalid yes/no answer: {answer}")
            return None

        # Format prompt
        prompt = f"{question} \\n\\n Please answer only with 'Yes' or 'No' in <answer> tags."

        # Create metadata
        metadata = {
            "dataset": "model_written_eval",
            "format": "yes_no",
            "original_data": {k: v for k, v in raw_data.items() if k != "_file_info"}
        }

        if "_file_info" in raw_data:
            metadata.update(raw_data["_file_info"])

        return TaskInstance(
            prompt=prompt,
            answer=answer,
            metadata=metadata
        )

    def _convert_multiple_choice_instance(self, raw_data: Dict[str, Any]) -> Optional[TaskInstance]:
        """Convert multiple choice question instance."""
        question_full = raw_data.get("question", "")
        answer_behavior = raw_data.get("answer_matching_behavior", "")

        if not question_full or not answer_behavior:
            self.logger.warning("Missing question or answer in multiple choice entry")
            return None

        # Parse question and choices
        question_main, choices = self._parse_multiple_choice_question(question_full)

        if not question_main or not choices:
            self.logger.warning("Could not parse multiple choice question")
            return None

        # Find answer index
        try:
            # Handle answer format like "(A)", "(B)", "(C)", etc.
            if answer_behavior.strip().startswith("(") and answer_behavior.strip().endswith(")"):
                # Extract the letter from "(X)" format
                letter = answer_behavior.strip()[1]
                if letter.isalpha() and letter.isupper():
                    # Convert letter to index: A=0, B=1, C=2, etc.
                    answer_index = ord(letter) - ord('A')
                    # Verify the index is valid for the number of choices
                    if answer_index >= len(choices):
                        self.logger.warning(f"Answer index {answer_index} out of range for {len(choices)} choices")
                        return None
                else:
                    # Invalid letter format, try to find in choices
                    answer_index = choices.index(answer_behavior)
            else:
                # Try to find the answer text in choices
                answer_index = choices.index(answer_behavior)
        except (ValueError, IndexError):
            self.logger.warning(f"Could not find answer '{answer_behavior}' in choices: {choices}")
            return None

        # Format prompt
        choices_text = "\\n".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
        prompt = f"{question_main}\\n\\nChoices:\\n{choices_text}\\n\\nAnswer:"

        # Create metadata
        metadata = {
            "dataset": "model_written_eval",
            "format": "multiple_choice",
            "original_data": {k: v for k, v in raw_data.items() if k != "_file_info"}
        }

        if "_file_info" in raw_data:
            metadata.update(raw_data["_file_info"])

        return TaskInstance(
            prompt=prompt,
            answer=choices[answer_index],
            answer_index=answer_index,
            classes=choices,
            metadata=metadata
        )

    def _parse_multiple_choice_question(self, question_full: str) -> tuple[str, List[str]]:
        """
        Parse multiple choice question to extract main question and choices.
        
        Args:
            question_full: Full question text with choices
            
        Returns:
            Tuple of (question_main, choices_list)
        """
        question_main = ""
        choices_text = ""
        

        # Try splitting by "\n\nChoices:\n" first (with colon)
        parts = question_full.split("\n\nChoices:\n", 1)
        if len(parts) == 2:
            question_main = parts[0].strip()
            choices_text = parts[1].strip()
            # Remove potential trailing "\n\nAnswer:"
            if choices_text.endswith("\n\nAnswer:"):
                choices_text = choices_text[:-len("\n\nAnswer:")].strip()
        else:
            # Try splitting by "\n\nChoices\n" (without colon)
            parts = question_full.split("\n\nChoices\n", 1)
            if len(parts) == 2:
                question_main = parts[0].strip()
                choices_text = parts[1].strip()
                # Remove potential trailing "\n\nAnswer:"
                if choices_text.endswith("\n\nAnswer:"):
                    choices_text = choices_text[:-len("\n\nAnswer:")].strip()
            else:
                # Try finding the start of choices with "\n (A)"
                choice_start_marker = "\n (A)"
                try:
                    choice_start_index = question_full.rindex(choice_start_marker)
                    question_main = question_full[:choice_start_index].strip()
                    choices_text = question_full[choice_start_index + 1:].strip()
                    if choices_text.endswith("\n\nAnswer:"):
                        choices_text = choices_text[:-len("\n\nAnswer:")].strip()
                except ValueError:
                    return "", []

        # Parse choices - handle newline-separated format like "\n (A) Option1\n (B) Option2"
        if not choices_text:
            # If no explicit choices section, try to extract from the full question
            # Look for pattern like "\n (A) " in the question
            choice_lines = []
            for line in question_full.split("\\n"):
                line = line.strip()
                if re.match(r"^\([A-Z]\)", line):
                    choice_lines.append(line)
            
            if choice_lines:
                question_main = question_full
                # Remove the choices from the question
                for choice_line in choice_lines:
                    question_main = question_main.replace("\\n" + choice_line, "")
                question_main = question_main.strip()
                
                # Parse the choices
                choices = []
                for line in choice_lines:
                    match = re.match(r"^\(([A-Z])\)\s*(.*)", line)
                    if match:
                        choices.append(match.group(2).strip())
            else:
                return "", []
        else:
            # Parse choices using regex on the choices_text
            choices_match = self.choice_pattern.findall(choices_text)
            
            if not choices_match:
                # Try line-by-line parsing
                choices = []
                for line in choices_text.split("\\n"):
                    line = line.strip()
                    if line:
                        match = re.match(r"^\(([A-Z])\)\s*(.*)", line)
                        if match:
                            choices.append(match.group(2).strip())
                
                if not choices:
                    return "", []
            else:
                # Extract choice texts and clean them
                choices = []
                for letter, text in choices_match:
                    cleaned_text = text.strip()
                    if cleaned_text:
                        choices.append(cleaned_text)

        return question_main, choices
