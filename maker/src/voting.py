"""
First-to-ahead-by-k Voting Algorithm for MAKER.

This module implements the error correction mechanism based on multi-agent voting.
The approach is motivated by the Sequential Probability Ratio Test (SPRT).

From the paper:
"For simplicity, the error correction in this paper uses the statistical power of
independent samples from a stochastic process (here an LLM). To determine a winner
from these samples, a first-to-ahead-by-k voting process is used."

Based on:
"Solving a Million-Step LLM Task with Zero Errors" (Meyerson et al., 2025)
"""

from typing import Dict, Tuple, Any, Callable, Optional, List
from collections import defaultdict
import hashlib
import json


def hash_response(move: list, next_state: list) -> str:
    """
    Create a unique hash for a (move, next_state) pair.

    This is used to identify identical responses for voting.

    Args:
        move: The move [disk, from_peg, to_peg]
        next_state: The resulting state [[peg0], [peg1], [peg2]]

    Returns:
        A hash string uniquely identifying this response
    """
    # Convert to canonical JSON string for consistent hashing
    canonical = json.dumps({"move": move, "state": next_state}, sort_keys=True)
    return hashlib.md5(canonical.encode()).hexdigest()


class VotingResult:
    """Container for voting results and statistics."""

    def __init__(
        self,
        winner_move: list,
        winner_state: list,
        total_samples: int,
        valid_samples: int,
        vote_counts: Dict[str, int],
        winner_votes: int
    ):
        self.move = winner_move
        self.next_state = winner_state
        self.total_samples = total_samples
        self.valid_samples = valid_samples
        self.vote_counts = vote_counts
        self.winner_votes = winner_votes

    def __repr__(self):
        return (
            f"VotingResult(move={self.move}, "
            f"samples={self.total_samples}, valid={self.valid_samples}, "
            f"winner_votes={self.winner_votes})"
        )


def do_voting(
    get_vote_fn: Callable[[], Optional[Tuple[list, list]]],
    k: int = 3,
    max_samples: int = 100,
    first_to_k: bool = False
) -> VotingResult:
    """
    First-to-ahead-by-k voting algorithm.

    Samples responses until one candidate is ahead by k votes from all others.
    This implements Algorithm 2 from the paper.

    Args:
        get_vote_fn: Function that returns (move, next_state) or None if red-flagged
        k: The vote margin required to win (default 3)
        max_samples: Maximum samples before giving up (safety limit)
        first_to_k: If True, use simpler first-to-k voting (first to reach k wins)

    Returns:
        VotingResult containing the winning response and statistics

    Raises:
        RuntimeError: If no valid response found within max_samples
    """
    # Vote counts: hash -> count
    vote_counts: Dict[str, int] = defaultdict(int)
    # Map hash to actual response
    responses: Dict[str, Tuple[list, list]] = {}

    total_samples = 0
    valid_samples = 0

    while total_samples < max_samples:
        total_samples += 1

        # Get a vote (may return None if red-flagged)
        result = get_vote_fn()

        if result is None:
            # Red-flagged response, skip
            continue

        move, next_state = result
        valid_samples += 1

        # Hash the response for voting
        response_hash = hash_response(move, next_state)

        # Store the response if new
        if response_hash not in responses:
            responses[response_hash] = (move, next_state)

        # Increment vote count
        vote_counts[response_hash] += 1
        current_votes = vote_counts[response_hash]

        # Check win condition
        if first_to_k:
            # Simple first-to-k: first candidate to reach k votes wins
            if current_votes >= k:
                return VotingResult(
                    winner_move=move,
                    winner_state=next_state,
                    total_samples=total_samples,
                    valid_samples=valid_samples,
                    vote_counts=dict(vote_counts),
                    winner_votes=current_votes
                )
        else:
            # First-to-ahead-by-k: must be k votes ahead of all others
            max_other = max(
                (v for h, v in vote_counts.items() if h != response_hash),
                default=0
            )
            if current_votes >= k + max_other:
                return VotingResult(
                    winner_move=move,
                    winner_state=next_state,
                    total_samples=total_samples,
                    valid_samples=valid_samples,
                    vote_counts=dict(vote_counts),
                    winner_votes=current_votes
                )

    # Max samples reached - return the current leader
    if not vote_counts:
        raise RuntimeError(f"No valid responses after {max_samples} samples")

    best_hash = max(vote_counts, key=vote_counts.get)
    best_move, best_state = responses[best_hash]

    return VotingResult(
        winner_move=best_move,
        winner_state=best_state,
        total_samples=total_samples,
        valid_samples=valid_samples,
        vote_counts=dict(vote_counts),
        winner_votes=vote_counts[best_hash]
    )


def get_vote(
    llm_call_fn: Callable[[str, str, float], str],
    system_prompt: str,
    user_prompt: str,
    parser_fn: Callable[[str], Tuple[list, list]],
    temperature: float = 0.1
) -> Optional[Tuple[list, list]]:
    """
    Get a single vote by calling the LLM and parsing the response.

    This implements Algorithm 3 from the paper.

    Args:
        llm_call_fn: Function(system, user, temperature) -> response_text
        system_prompt: The system prompt
        user_prompt: The user prompt with current state
        parser_fn: Function to parse response (should raise ValueError on red flag)
        temperature: Sampling temperature

    Returns:
        Tuple of (move, next_state) if valid, None if red-flagged
    """
    try:
        response = llm_call_fn(system_prompt, user_prompt, temperature)
        return parser_fn(response)
    except (ValueError, Exception):
        # Red-flagged or error - return None
        return None


def calculate_k_min(
    num_steps: int,
    per_step_success_rate: float,
    target_success_prob: float = 0.95
) -> int:
    """
    Calculate the minimum k required for a given success probability.

    From the paper (Equation 14):
    k_min = ceil(ln(t^(-1/s) - 1) / ln((1-p)/p))

    Args:
        num_steps: Total number of steps (s)
        per_step_success_rate: Per-step success probability (p)
        target_success_prob: Target overall success probability (t)

    Returns:
        The minimum k value required
    """
    import math

    if per_step_success_rate <= 0.5:
        raise ValueError("Per-step success rate must be > 0.5 for voting to work")

    p = per_step_success_rate
    t = target_success_prob
    s = num_steps

    # t^(-1/s) - 1
    a = t ** (-1/s) - 1

    # ln((1-p)/p)
    b = math.log((1 - p) / p)

    k_min = math.ceil(math.log(a) / b)

    return max(1, k_min)


def estimate_cost(
    num_steps: int,
    per_step_success_rate: float,
    cost_per_sample: float,
    k: int
) -> float:
    """
    Estimate the expected cost of running MAKER.

    From the paper (Equation 18):
    E[cost] = O(s * ln(s)) when using MAD (m=1)

    More precisely (Equation 17):
    E[cost] ≈ c * s * k_min / (p^(m-1) * (2p - 1))

    For MAD (m=1):
    E[cost] ≈ c * s * k / (2p - 1)

    Args:
        num_steps: Total number of steps
        per_step_success_rate: Per-step success probability
        cost_per_sample: Cost per LLM call
        k: The voting parameter

    Returns:
        Estimated total cost
    """
    p = per_step_success_rate
    s = num_steps
    c = cost_per_sample

    # Expected samples per step (approximate)
    # In practice, most steps complete in k samples due to exponential convergence
    expected_samples_per_step = k / (2 * p - 1)

    return c * s * expected_samples_per_step
