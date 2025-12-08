"""
LLM Client for MAKER - Handles API calls to various LLM providers.

Supports:
- OpenAI (gpt-4.1-mini, gpt-4.1-nano, o3-mini, etc.)
- Anthropic (claude-3.5-haiku, etc.)
- Together.ai (open-source models)

Based on:
"Solving a Million-Step LLM Task with Zero Errors" (Meyerson et al., 2025)
"""

import os
import time
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for an LLM model."""
    name: str
    provider: str
    input_cost_per_mtok: float  # $ per million input tokens
    output_cost_per_mtok: float  # $ per million output tokens
    max_output_tokens: int = 750
    default_temperature: float = 0.1


# Model configurations from the paper
MODEL_CONFIGS = {
    # OpenAI models
    "gpt-4.1-nano": ModelConfig(
        name="gpt-4.1-nano",
        provider="openai",
        input_cost_per_mtok=0.1,
        output_cost_per_mtok=0.4
    ),
    "gpt-4.1-mini": ModelConfig(
        name="gpt-4.1-mini",
        provider="openai",
        input_cost_per_mtok=0.4,
        output_cost_per_mtok=1.6
    ),
    "o3-mini": ModelConfig(
        name="o3-mini",
        provider="openai",
        input_cost_per_mtok=1.1,
        output_cost_per_mtok=4.4
    ),
    # Anthropic models
    "claude-3.5-haiku": ModelConfig(
        name="claude-3-5-haiku-latest",
        provider="anthropic",
        input_cost_per_mtok=1.0,
        output_cost_per_mtok=5.0
    ),
    # Open-source models via Together.ai
    "qwen-3": ModelConfig(
        name="Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
        provider="together",
        input_cost_per_mtok=0.2,
        output_cost_per_mtok=0.6
    ),
    "deepseek-v3.1": ModelConfig(
        name="deepseek-ai/DeepSeek-V3",
        provider="together",
        input_cost_per_mtok=0.6,
        output_cost_per_mtok=1.7
    ),
    "llama-3.2-3B": ModelConfig(
        name="meta-llama/Llama-3.2-3B-Instruct-Turbo",
        provider="together",
        input_cost_per_mtok=0.06,
        output_cost_per_mtok=0.06
    ),
}


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 750
    ) -> str:
        """Make an LLM API call and return the response text."""
        pass

    @abstractmethod
    def get_cost_per_call(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate the cost of a call given token counts."""
        pass


class OpenAIClient(LLMClient):
    """Client for OpenAI API."""

    def __init__(self, model: str = "gpt-4.1-mini", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")

        self.config = MODEL_CONFIGS.get(model, ModelConfig(
            name=model,
            provider="openai",
            input_cost_per_mtok=1.0,
            output_cost_per_mtok=2.0
        ))

        # Import here to make the dependency optional
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 750
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.config.name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    def get_cost_per_call(self, input_tokens: int, output_tokens: int) -> float:
        input_cost = (input_tokens / 1_000_000) * self.config.input_cost_per_mtok
        output_cost = (output_tokens / 1_000_000) * self.config.output_cost_per_mtok
        return input_cost + output_cost


class AnthropicClient(LLMClient):
    """Client for Anthropic API."""

    def __init__(self, model: str = "claude-3.5-haiku", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY env var.")

        self.config = MODEL_CONFIGS.get(model, ModelConfig(
            name=model,
            provider="anthropic",
            input_cost_per_mtok=1.0,
            output_cost_per_mtok=5.0
        ))

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 750
    ) -> str:
        response = self.client.messages.create(
            model=self.config.name,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature
        )
        return response.content[0].text

    def get_cost_per_call(self, input_tokens: int, output_tokens: int) -> float:
        input_cost = (input_tokens / 1_000_000) * self.config.input_cost_per_mtok
        output_cost = (output_tokens / 1_000_000) * self.config.output_cost_per_mtok
        return input_cost + output_cost


class TogetherClient(LLMClient):
    """Client for Together.ai API (open-source models)."""

    def __init__(self, model: str = "qwen-3", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("Together API key required. Set TOGETHER_API_KEY env var.")

        self.config = MODEL_CONFIGS.get(model, ModelConfig(
            name=model,
            provider="together",
            input_cost_per_mtok=0.5,
            output_cost_per_mtok=1.0
        ))

        try:
            from together import Together
            self.client = Together(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install together: pip install together")

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 750
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.config.name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    def get_cost_per_call(self, input_tokens: int, output_tokens: int) -> float:
        input_cost = (input_tokens / 1_000_000) * self.config.input_cost_per_mtok
        output_cost = (output_tokens / 1_000_000) * self.config.output_cost_per_mtok
        return input_cost + output_cost


class MockLLMClient(LLMClient):
    """Mock client for testing - returns correct Towers of Hanoi moves."""

    def __init__(self, error_rate: float = 0.0, num_disks: int = 5):
        """
        Initialize mock client.

        Args:
            error_rate: Probability of returning an incorrect response (0.0 to 1.0)
            num_disks: Number of disks in the puzzle
        """
        self.error_rate = error_rate
        self.num_disks = num_disks
        self.config = ModelConfig(
            name="mock",
            provider="mock",
            input_cost_per_mtok=0.0,
            output_cost_per_mtok=0.0
        )

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 750
    ) -> str:
        import random
        import re
        import ast

        # Parse current state and previous move from prompt
        try:
            state_match = re.search(r"Current State:\s*(\[\[.*?\]\])", user_prompt, re.DOTALL)
            move_match = re.search(r"Previous move:\s*(\[.*?\]|\w+)", user_prompt)

            if state_match:
                current_state = ast.literal_eval(state_match.group(1).replace("\n", ""))
            else:
                current_state = [list(range(self.num_disks, 0, -1)), [], []]

            if move_match:
                prev_move_str = move_match.group(1)
                if prev_move_str.lower() in ['none', 'null']:
                    prev_move = None
                else:
                    prev_move = ast.literal_eval(prev_move_str)
            else:
                prev_move = None

        except Exception:
            current_state = [list(range(self.num_disks, 0, -1)), [], []]
            prev_move = None

        # Calculate correct move using standard Tower of Hanoi algorithm
        move, next_state = self._calculate_correct_move(current_state, prev_move)

        # Introduce errors based on error_rate
        if random.random() < self.error_rate:
            # Make an error
            if random.random() < 0.5:
                # Wrong move
                move = [move[0], move[1], (move[2] + 1) % 3]
            else:
                # Wrong state
                if next_state[0]:
                    next_state[0][-1], next_state[1 if next_state[1] else 2].append(
                        next_state[0].pop() if next_state[0] else 1
                    ) if next_state[1] or next_state[2] else None

        return f"""
Let me solve this step by step.

Based on the Tower of Hanoi rules:
- Previous move: {prev_move}
- Current state: {current_state}

The next move should be:

move = {move}
next_state = {next_state}
"""

    def _calculate_correct_move(self, state: list, prev_move: Optional[list]):
        """Calculate the correct next move using standard algorithm."""
        import copy

        # Find where disk 1 is
        disk_1_peg = None
        for peg_idx, peg in enumerate(state):
            if peg and peg[-1] == 1:
                disk_1_peg = peg_idx
                break

        # Determine if previous move was disk 1
        prev_moved_disk_1 = prev_move is not None and prev_move[0] == 1

        if not prev_moved_disk_1:
            # Move disk 1 clockwise
            if disk_1_peg is not None:
                target_peg = (disk_1_peg + 1) % 3
                new_state = copy.deepcopy(state)
                disk = new_state[disk_1_peg].pop()
                new_state[target_peg].append(disk)
                return [1, disk_1_peg, target_peg], new_state
            else:
                # Disk 1 not found - shouldn't happen
                return [1, 0, 1], state
        else:
            # Make the only legal move not involving disk 1
            new_state = copy.deepcopy(state)

            # Find pegs without disk 1 on top
            candidates = []
            for peg_idx, peg in enumerate(state):
                if peg and peg[-1] != 1:
                    candidates.append((peg_idx, peg[-1]))

            if len(candidates) == 0:
                # No valid move - shouldn't happen in valid game
                return [2, 0, 1], state
            elif len(candidates) == 1:
                # Only one candidate - must move it
                from_peg, disk = candidates[0]
                # Find valid target
                for to_peg in range(3):
                    if to_peg != from_peg:
                        to_top = state[to_peg][-1] if state[to_peg] else float('inf')
                        if disk < to_top:
                            new_state[from_peg].pop()
                            new_state[to_peg].append(disk)
                            return [disk, from_peg, to_peg], new_state
            else:
                # Two candidates - find the legal move
                for from_peg, disk in candidates:
                    for to_peg in range(3):
                        if to_peg != from_peg:
                            to_top = state[to_peg][-1] if state[to_peg] else float('inf')
                            # Skip if target has disk 1 on top (we can't put larger on smaller)
                            if to_top == 1:
                                continue
                            if disk < to_top:
                                new_state = copy.deepcopy(state)
                                new_state[from_peg].pop()
                                new_state[to_peg].append(disk)
                                return [disk, from_peg, to_peg], new_state

            # Fallback
            return [2, 0, 1], state

    def get_cost_per_call(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0


def create_client(
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None
) -> LLMClient:
    """
    Factory function to create an LLM client.

    Args:
        provider: One of "openai", "anthropic", "together", "mock"
        model: Model name (provider-specific)
        api_key: API key (or use environment variable)

    Returns:
        An LLMClient instance
    """
    if provider == "openai":
        return OpenAIClient(model=model or "gpt-4.1-mini", api_key=api_key)
    elif provider == "anthropic":
        return AnthropicClient(model=model or "claude-3.5-haiku", api_key=api_key)
    elif provider == "together":
        return TogetherClient(model=model or "qwen-3", api_key=api_key)
    elif provider == "mock":
        return MockLLMClient()
    else:
        raise ValueError(f"Unknown provider: {provider}")
