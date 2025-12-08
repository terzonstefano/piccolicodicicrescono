"""
MAKER Solver - Main implementation of the MDAP framework for Towers of Hanoi.

This module implements Algorithm 1 (generate_solution) from the paper,
orchestrating the decomposed agents with voting-based error correction.

Based on:
"Solving a Million-Step LLM Task with Zero Errors" (Meyerson et al., 2025)
"""

import copy
import time
import json
from typing import Optional, List, Tuple, Dict, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path

from .prompts import SYSTEM_PROMPT, format_user_prompt
from .parsers import parse_move_state_flag, RedFlagChecker
from .voting import do_voting, VotingResult, calculate_k_min
from .llm_client import LLMClient, create_client


@dataclass
class StepResult:
    """Result of a single step execution."""
    step_number: int
    move: List[int]
    next_state: List[List[int]]
    voting_result: VotingResult
    elapsed_time: float
    is_correct: Optional[bool] = None  # If ground truth available


@dataclass
class SolverStats:
    """Statistics for a solver run."""
    total_steps: int = 0
    completed_steps: int = 0
    total_samples: int = 0
    total_valid_samples: int = 0
    total_red_flagged: int = 0
    errors_detected: int = 0
    total_time: float = 0.0
    estimated_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "total_samples": self.total_samples,
            "total_valid_samples": self.total_valid_samples,
            "total_red_flagged": self.total_red_flagged,
            "red_flag_rate": self.total_red_flagged / max(1, self.total_samples),
            "errors_detected": self.errors_detected,
            "total_time": self.total_time,
            "avg_time_per_step": self.total_time / max(1, self.completed_steps),
            "estimated_cost": self.estimated_cost
        }


class TowersOfHanoi:
    """
    Towers of Hanoi puzzle implementation.

    Provides the ground truth for validation and initial state generation.
    """

    def __init__(self, num_disks: int = 20):
        self.num_disks = num_disks
        self.total_steps = 2 ** num_disks - 1

    def get_initial_state(self) -> List[List[int]]:
        """Get the initial state with all disks on peg 0."""
        return [list(range(self.num_disks, 0, -1)), [], []]

    def get_final_state(self) -> List[List[int]]:
        """
        Get the final state.

        With the clockwise algorithm (disk 1 moves 0->1->2->0):
        - Even number of disks: ends on peg 2
        - Odd number of disks: ends on peg 1
        """
        if self.num_disks % 2 == 0:
            # Even: ends on peg 2
            return [[], [], list(range(self.num_disks, 0, -1))]
        else:
            # Odd: ends on peg 1
            return [[], list(range(self.num_disks, 0, -1)), []]

    def is_valid_move(
        self,
        state: List[List[int]],
        move: List[int]
    ) -> Tuple[bool, str]:
        """
        Check if a move is valid from the given state.

        Args:
            state: Current state [[peg0], [peg1], [peg2]]
            move: [disk, from_peg, to_peg]

        Returns:
            Tuple of (is_valid, error_message)
        """
        disk, from_peg, to_peg = move

        # Check peg indices
        if from_peg not in [0, 1, 2] or to_peg not in [0, 1, 2]:
            return False, "Invalid peg index"

        if from_peg == to_peg:
            return False, "Source and target peg are the same"

        # Check disk is on top of source peg
        if not state[from_peg] or state[from_peg][-1] != disk:
            return False, f"Disk {disk} is not on top of peg {from_peg}"

        # Check target peg constraint
        if state[to_peg] and state[to_peg][-1] < disk:
            return False, f"Cannot place disk {disk} on smaller disk {state[to_peg][-1]}"

        return True, ""

    def apply_move(
        self,
        state: List[List[int]],
        move: List[int]
    ) -> List[List[int]]:
        """Apply a move to a state and return the new state."""
        new_state = copy.deepcopy(state)
        disk, from_peg, to_peg = move
        new_state[from_peg].pop()
        new_state[to_peg].append(disk)
        return new_state

    def get_correct_move(
        self,
        state: List[List[int]],
        prev_move: Optional[List[int]]
    ) -> Tuple[List[int], List[List[int]]]:
        """
        Calculate the correct next move using standard algorithm.

        The algorithm:
        - If previous move didn't move disk 1: move disk 1 clockwise
        - If previous move moved disk 1: make the only legal non-disk-1 move
        """
        # Find disk 1
        disk_1_peg = None
        for peg_idx, peg in enumerate(state):
            if peg and peg[-1] == 1:
                disk_1_peg = peg_idx
                break

        prev_moved_disk_1 = prev_move is not None and prev_move[0] == 1

        if not prev_moved_disk_1:
            # Move disk 1 clockwise (0 -> 1 -> 2 -> 0)
            if disk_1_peg is not None:
                target_peg = (disk_1_peg + 1) % 3
                move = [1, disk_1_peg, target_peg]
                next_state = self.apply_move(state, move)
                return move, next_state
        else:
            # Make the only legal move not involving disk 1
            for from_peg, peg in enumerate(state):
                if not peg or peg[-1] == 1:
                    continue
                disk = peg[-1]
                for to_peg in range(3):
                    if to_peg == from_peg:
                        continue
                    # Check if move is legal
                    to_top = state[to_peg][-1] if state[to_peg] else float('inf')
                    if disk < to_top and to_top != 1:
                        move = [disk, from_peg, to_peg]
                        next_state = self.apply_move(state, move)
                        return move, next_state

        # Shouldn't reach here in valid game
        raise ValueError("No valid move found")

    def generate_solution(self) -> List[Tuple[List[int], List[List[int]]]]:
        """Generate the complete solution (for validation)."""
        solution = []
        state = self.get_initial_state()
        prev_move = None

        for _ in range(self.total_steps):
            move, next_state = self.get_correct_move(state, prev_move)
            solution.append((move, next_state))
            state = next_state
            prev_move = move

        return solution


class MAKERSolver:
    """
    MAKER Solver - Solves Towers of Hanoi using MDAP framework.

    Implements:
    - Maximal Agentic Decomposition (1 step per agent)
    - First-to-ahead-by-k Voting for error correction
    - Red-flagging for detecting unreliable responses
    """

    def __init__(
        self,
        num_disks: int = 20,
        k: int = 3,
        max_tokens: int = 750,
        llm_client: Optional[LLMClient] = None,
        provider: str = "mock",
        model: Optional[str] = None,
        validate: bool = True,
        verbose: bool = True,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize the MAKER solver.

        Args:
            num_disks: Number of disks (default 20 for million-step task)
            k: Voting parameter (votes ahead required to win)
            max_tokens: Maximum response tokens (for red-flagging)
            llm_client: Pre-configured LLM client (optional)
            provider: LLM provider if no client given
            model: Model name if no client given
            validate: Whether to validate moves against ground truth
            verbose: Whether to print progress
            checkpoint_dir: Directory for saving checkpoints
        """
        self.num_disks = num_disks
        self.k = k
        self.max_tokens = max_tokens
        self.validate = validate
        self.verbose = verbose
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # Initialize puzzle
        self.puzzle = TowersOfHanoi(num_disks)
        self.total_steps = self.puzzle.total_steps

        # Initialize LLM client
        if llm_client:
            self.llm_client = llm_client
        else:
            self.llm_client = create_client(provider=provider, model=model)

        # Initialize red flag checker
        self.red_flag_checker = RedFlagChecker(
            max_tokens=max_tokens,
            num_disks=num_disks
        )

        # Statistics
        self.stats = SolverStats(total_steps=self.total_steps)

        # Solution storage
        self.solution: List[StepResult] = []

        if self.verbose:
            print(f"MAKER Solver initialized:")
            print(f"  Disks: {num_disks}")
            print(f"  Total steps: {self.total_steps:,}")
            print(f"  Voting k: {k}")
            print(f"  Max tokens: {max_tokens}")

    def _get_vote(
        self,
        current_state: List[List[int]],
        previous_move: Optional[List[int]],
        temperature: float = 0.1
    ) -> Optional[Tuple[List[int], List[List[int]]]]:
        """
        Get a single vote by calling the LLM.

        Returns None if response is red-flagged.
        """
        # Format prompt
        user_prompt = format_user_prompt(
            previous_move=previous_move if previous_move else "None (first move)",
            current_state=current_state
        )

        try:
            # Call LLM
            response = self.llm_client.call(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=self.max_tokens
            )

            # Check for red flags and parse
            result = self.red_flag_checker.get_valid_response(response)
            return result

        except Exception as e:
            if self.verbose:
                print(f"    LLM call error: {e}")
            return None

    def _execute_step(
        self,
        step_number: int,
        current_state: List[List[int]],
        previous_move: Optional[List[int]],
        ground_truth: Optional[Tuple[List[int], List[List[int]]]] = None
    ) -> StepResult:
        """
        Execute a single step using voting.

        This implements the inner loop of Algorithm 1.
        """
        start_time = time.time()

        # Create vote function for this step
        sample_count = 0
        first_sample = True

        def get_vote_fn() -> Optional[Tuple[List[int], List[List[int]]]]:
            nonlocal sample_count, first_sample
            sample_count += 1

            # Use temperature 0 for first sample (greedy), then 0.1
            temp = 0.0 if first_sample else 0.1
            first_sample = False

            result = self._get_vote(current_state, previous_move, temperature=temp)

            if result is None:
                self.stats.total_red_flagged += 1

            return result

        # Run voting
        voting_result = do_voting(
            get_vote_fn=get_vote_fn,
            k=self.k,
            max_samples=100
        )

        elapsed = time.time() - start_time

        # Update stats
        self.stats.total_samples += voting_result.total_samples
        self.stats.total_valid_samples += voting_result.valid_samples

        # Check correctness if ground truth available
        is_correct = None
        if ground_truth:
            correct_move, correct_state = ground_truth
            is_correct = (
                voting_result.move == correct_move and
                voting_result.next_state == correct_state
            )
            if not is_correct:
                self.stats.errors_detected += 1

        return StepResult(
            step_number=step_number,
            move=voting_result.move,
            next_state=voting_result.next_state,
            voting_result=voting_result,
            elapsed_time=elapsed,
            is_correct=is_correct
        )

    def solve(
        self,
        max_steps: Optional[int] = None,
        start_step: int = 0,
        checkpoint_interval: int = 1000
    ) -> List[StepResult]:
        """
        Solve the Towers of Hanoi puzzle.

        This implements Algorithm 1 (generate_solution) from the paper.

        Args:
            max_steps: Maximum steps to execute (None for all)
            start_step: Step to start from (for resuming)
            checkpoint_interval: Save checkpoint every N steps

        Returns:
            List of StepResult objects
        """
        if max_steps is None:
            max_steps = self.total_steps

        # Generate ground truth if validating
        ground_truth = None
        if self.validate:
            if self.verbose:
                print("Generating ground truth solution...")
            ground_truth = self.puzzle.generate_solution()

        # Initialize state
        if start_step == 0:
            current_state = self.puzzle.get_initial_state()
            previous_move = None
            self.solution = []
        else:
            # Resume from checkpoint
            if self.solution and len(self.solution) >= start_step:
                last_result = self.solution[start_step - 1]
                current_state = last_result.next_state
                previous_move = last_result.move
            else:
                raise ValueError(f"Cannot resume from step {start_step}")

        start_time = time.time()

        if self.verbose:
            print(f"\nStarting solve from step {start_step}...")
            print(f"Target: {min(start_step + max_steps, self.total_steps):,} steps")

        try:
            for step in range(start_step, min(start_step + max_steps, self.total_steps)):
                # Get ground truth for this step
                gt = ground_truth[step] if ground_truth else None

                # Execute step
                result = self._execute_step(
                    step_number=step,
                    current_state=current_state,
                    previous_move=previous_move,
                    ground_truth=gt
                )

                self.solution.append(result)
                self.stats.completed_steps += 1

                # Update state for next step
                current_state = result.next_state
                previous_move = result.move

                # Progress output
                if self.verbose and (step + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = self.stats.completed_steps / elapsed
                    eta = (self.total_steps - step - 1) / rate if rate > 0 else 0

                    status = "OK" if result.is_correct else "ERR" if result.is_correct is False else "?"
                    print(
                        f"  Step {step + 1:,}/{self.total_steps:,} "
                        f"[{status}] "
                        f"samples={result.voting_result.total_samples} "
                        f"rate={rate:.1f}/s "
                        f"ETA={eta/60:.1f}m"
                    )

                # Checkpoint
                if self.checkpoint_dir and (step + 1) % checkpoint_interval == 0:
                    self._save_checkpoint(step + 1)

        except KeyboardInterrupt:
            if self.verbose:
                print("\nInterrupted by user")

        self.stats.total_time = time.time() - start_time

        if self.verbose:
            self._print_summary()

        return self.solution

    def _save_checkpoint(self, step: int):
        """Save a checkpoint of the current solution."""
        if not self.checkpoint_dir:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{step}.json"

        data = {
            "step": step,
            "num_disks": self.num_disks,
            "k": self.k,
            "stats": self.stats.to_dict(),
            "solution": [
                {
                    "step": r.step_number,
                    "move": r.move,
                    "next_state": r.next_state,
                    "is_correct": r.is_correct
                }
                for r in self.solution
            ]
        }

        with open(checkpoint_file, 'w') as f:
            json.dump(data, f)

        if self.verbose:
            print(f"  Checkpoint saved: {checkpoint_file}")

    def _print_summary(self):
        """Print a summary of the solve run."""
        stats = self.stats.to_dict()

        print("\n" + "=" * 50)
        print("MAKER Solver Summary")
        print("=" * 50)
        print(f"Steps completed: {stats['completed_steps']:,}/{stats['total_steps']:,}")
        print(f"Total samples: {stats['total_samples']:,}")
        print(f"Valid samples: {stats['total_valid_samples']:,}")
        print(f"Red-flagged: {stats['total_red_flagged']:,} ({stats['red_flag_rate']*100:.1f}%)")
        print(f"Errors detected: {stats['errors_detected']}")
        print(f"Total time: {stats['total_time']:.1f}s")
        print(f"Avg time/step: {stats['avg_time_per_step']*1000:.1f}ms")

        if stats['errors_detected'] == 0 and stats['completed_steps'] > 0:
            print("\n*** ZERO ERRORS - Success! ***")
        print("=" * 50)

    def verify_solution(self) -> bool:
        """Verify the solution is correct."""
        if not self.solution:
            return False

        puzzle = TowersOfHanoi(self.num_disks)
        state = puzzle.get_initial_state()

        for i, result in enumerate(self.solution):
            # Check move is valid
            is_valid, error = puzzle.is_valid_move(state, result.move)
            if not is_valid:
                print(f"Invalid move at step {i}: {error}")
                return False

            # Apply move
            new_state = puzzle.apply_move(state, result.move)

            # Check state matches
            if new_state != result.next_state:
                print(f"State mismatch at step {i}")
                return False

            state = new_state

        # Check final state
        if state != puzzle.get_final_state():
            print("Final state incorrect")
            return False

        return True
