"""
Answer parsers for all datasets used in calibration experiments.
These parsers are shared between local and vLLM calibration scripts.
"""

import re
from typing import Optional

def parse_gsm8k_answer(output: str) -> Optional[int]:
    """Parse the answer from GSM8K responses."""
    final_answer_match = re.search(r"The final answer is \$?(\d+,?\d*)", output)
    if final_answer_match:
        return int(final_answer_match.group(1).replace(",", ""))
    matches = re.findall(r"\$?(\d+,?\d*)", output)
    return int(matches[-1].replace(",", "")) if matches else None

def parse_mmlu_answer(output: str) -> Optional[str]:
    """Parse the answer from MMLU responses."""
    match = re.search(r"The final answer is ([A-D])", output)
    return match.group(1) if match else None

def parse_medmcqa_answer(output: str) -> Optional[str]:
    """Parse the answer from MedMCQA responses."""
    match = re.search(r"The final answer is ([A-D])", output)
    return match.group(1) if match else None

def parse_simpleqa_answer(output: str) -> str:
    """Parse the answer from SimpleQA responses."""
    answer_match = re.search(
        r"The answer is: (.*?)(?:\.|$)", output, re.IGNORECASE | re.MULTILINE
    )
    return answer_match.group(1).strip() if answer_match else output.strip() 