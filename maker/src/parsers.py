"""
Parsers for MAKER - Extract move and state from LLM responses.

Two parser types:
1. Repairing parser: Attempts to fix common formatting errors
2. Red-flagging parser: Strict validation, discards malformed responses

Based on:
"Solving a Million-Step LLM Task with Zero Errors" (Meyerson et al., 2025)
"""

import re
import ast
from typing import Tuple, List, Optional


def extract_balanced_brackets(text: str, start_idx: int) -> str:
    """Extract a substring with balanced brackets [[...]] starting at start_idx."""
    bracket_stack = []
    i = start_idx
    while i < len(text):
        if text[i] == '[':
            bracket_stack.append('[')
        elif text[i] == ']':
            if not bracket_stack:
                break
            bracket_stack.pop()
            if not bracket_stack:
                return text[start_idx:i + 1]
        i += 1
    return text[start_idx:i] + ']'


def parse_move_state_repair(response_text: str) -> Tuple[List[int], List[List[int]]]:
    """
    Repairing parser - attempts to extract move and state even with formatting issues.

    This parser tries to be helpful by fixing common formatting errors.
    Use this for initial exploration/testing, not for production runs.

    Args:
        response_text: The raw LLM response text

    Returns:
        Tuple of (move, next_state) where:
        - move is [disk_id, from_peg, to_peg]
        - next_state is [[peg0], [peg1], [peg2]]

    Raises:
        ValueError: If parsing fails
    """
    # Parse move
    try:
        move_matches = re.findall(r"(?i)\bmove\b\s*=\s*(\[[^\[\]]*\])", response_text)
        if not move_matches:
            raise ValueError("No 'move' found in response.")
        move = ast.literal_eval(move_matches[-1].strip())
    except Exception as e:
        raise ValueError("Could not parse 'move' from response.") from e

    # Parse next_state
    try:
        # Match last occurrence of 'next_state = [ [' with any whitespace
        pattern = re.compile(r"(?i)\bnext_state\b\s*=\s*(\[\s*\[)", re.DOTALL)
        matches = list(pattern.finditer(response_text))
        if not matches:
            raise ValueError("No 'next_state' found in response.")
        start_idx = matches[-1].start(1)  # last match
        next_state_str = extract_balanced_brackets(response_text, start_idx).strip()
        next_state = ast.literal_eval(next_state_str)
    except Exception as e:
        raise ValueError("Could not parse 'next_state' from response.") from e

    return move, next_state


def _validate_move(move: list) -> List[int]:
    """Validate that move is a list of exactly 3 integers."""
    if not isinstance(move, list) or len(move) != 3 or not all(isinstance(x, int) for x in move):
        raise ValueError("'move' must be a list of exactly 3 integers.")
    return move


def _validate_state(state: list, num_disks: int = 20) -> List[List[int]]:
    """
    Validate that state is a valid Towers of Hanoi configuration.

    Args:
        state: The state to validate
        num_disks: Number of disks in the puzzle (default 20)

    Returns:
        The validated state

    Raises:
        ValueError: If state is invalid
    """
    if not (isinstance(state, list) and len(state) == 3 and all(isinstance(t, list) for t in state)):
        raise ValueError("'next_state' must be a list of three lists.")

    flat = [x for t in state for x in t]
    if not all(isinstance(x, int) for x in flat):
        raise ValueError("All entries in 'next_state' must be integers.")

    if len(flat) != num_disks or set(flat) != set(range(1, num_disks + 1)):
        missing = sorted(set(range(1, num_disks + 1)) - set(flat))
        extra = sorted(set(flat) - set(range(1, num_disks + 1)))
        raise ValueError(
            f"State must contain 1..{num_disks} exactly once. "
            f"Missing: {missing or '[]'}, Extras: {extra or '[]'}"
        )
    return state


def parse_move_state_flag(
    response_text: str,
    num_disks: int = 20
) -> Tuple[List[int], List[List[int]]]:
    """
    Red-flagging parser - strict validation, discards malformed responses.

    This parser is used in production runs. It flags and discards any response
    that doesn't perfectly match the expected format, as format errors often
    correlate with reasoning errors.

    Args:
        response_text: The raw LLM response text
        num_disks: Number of disks in the puzzle

    Returns:
        Tuple of (move, next_state)

    Raises:
        ValueError: If response doesn't match expected format exactly
    """
    # Match square brackets with strict format
    move_pat = re.compile(r"(?is)\bmove\b\s*=\s*(\[[^\[\]]*\])")
    state_pat = re.compile(
        r"(?is)\bnext_state\b\s*=\s*(\[\s*\[[^\[\]]*\]\s*,\s*\[[^\[\]]*\]\s*,\s*\[[^\[\]]*\]\s*\])"
    )

    move_matches = list(move_pat.finditer(response_text))
    if not move_matches:
        raise ValueError("No 'move = [...]' found.")
    move_str = move_matches[-1].group(1)  # last 'move'

    state_matches = list(state_pat.finditer(response_text))
    if not state_matches:
        raise ValueError("No 'next_state = [[...],[...],[...]]' found.")
    state_str = state_matches[-1].group(1)  # last 'next_state'

    try:
        move = ast.literal_eval(move_str)
    except Exception as e:
        raise ValueError("Could not parse 'move' as a Python list.") from e

    try:
        next_state = ast.literal_eval(state_str)
    except Exception as e:
        raise ValueError("Could not parse 'next_state' as Python lists.") from e

    return _validate_move(move), _validate_state(next_state, num_disks)


class RedFlagChecker:
    """
    Red flag detection for LLM responses.

    Red flags indicate potentially unreliable responses:
    1. Overly long responses (> max_tokens threshold)
    2. Incorrectly formatted responses

    From the paper: "bad behaviors are correlated in LLMs, so if an LLM produces
    a response that signals pathological behavior, the response should be flagged
    and simply discarded."
    """

    def __init__(self, max_tokens: int = 750, num_disks: int = 20):
        """
        Initialize the red flag checker.

        Args:
            max_tokens: Maximum allowed response length in tokens (approx)
            num_disks: Number of disks in the puzzle
        """
        self.max_tokens = max_tokens
        self.num_disks = num_disks

    def check_response(self, response_text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a response has any red flags.

        Args:
            response_text: The LLM response text

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if no red flags
            - error_message: Description of the red flag if invalid, None otherwise
        """
        # Check length (approximate token count by chars/4)
        approx_tokens = len(response_text) / 4
        if approx_tokens > self.max_tokens:
            return False, f"Response too long: ~{int(approx_tokens)} tokens > {self.max_tokens}"

        # Try to parse - format errors are red flags
        try:
            parse_move_state_flag(response_text, self.num_disks)
            return True, None
        except ValueError as e:
            return False, f"Format error: {str(e)}"

    def get_valid_response(
        self,
        response_text: str
    ) -> Optional[Tuple[List[int], List[List[int]]]]:
        """
        Get parsed response if valid, None if red-flagged.

        Args:
            response_text: The LLM response text

        Returns:
            Tuple of (move, next_state) if valid, None if red-flagged
        """
        is_valid, _ = self.check_response(response_text)
        if not is_valid:
            return None

        try:
            return parse_move_state_flag(response_text, self.num_disks)
        except ValueError:
            return None
