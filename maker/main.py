#!/usr/bin/env python3
"""
MAKER - Solving Long-Horizon LLM Tasks with Zero Errors

Command-line interface for the MAKER framework.

Usage:
    python main.py --help
    python main.py solve --disks 10 --provider mock
    python main.py solve --disks 20 --provider openai --model gpt-4.1-mini
    python main.py estimate --disks 20 --error-rate 0.002

Based on:
"Solving a Million-Step LLM Task with Zero Errors" (Meyerson et al., 2025)
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.solver import MAKERSolver, TowersOfHanoi
from src.voting import calculate_k_min, estimate_cost
from src.llm_client import create_client, MODEL_CONFIGS


def cmd_solve(args):
    """Run the MAKER solver on Towers of Hanoi."""
    print("=" * 60)
    print("MAKER - Massively Decomposed Agentic Process")
    print("Solving Towers of Hanoi with Zero Errors")
    print("=" * 60)

    # Create solver
    solver = MAKERSolver(
        num_disks=args.disks,
        k=args.k,
        max_tokens=args.max_tokens,
        provider=args.provider,
        model=args.model,
        validate=not args.no_validate,
        verbose=not args.quiet,
        checkpoint_dir=args.checkpoint_dir
    )

    # Run solver
    solution = solver.solve(
        max_steps=args.max_steps,
        checkpoint_interval=args.checkpoint_interval
    )

    # Verify if requested
    if args.verify and solution:
        print("\nVerifying solution...")
        if solver.verify_solution():
            print("Solution verified: CORRECT")
        else:
            print("Solution verified: INCORRECT")
            return 1

    return 0


def cmd_estimate(args):
    """Estimate cost and parameters for a MAKER run."""
    print("=" * 60)
    print("MAKER Cost Estimation")
    print("=" * 60)

    num_steps = 2 ** args.disks - 1
    p = 1 - args.error_rate

    print(f"\nPuzzle Configuration:")
    print(f"  Disks: {args.disks}")
    print(f"  Total steps: {num_steps:,}")

    print(f"\nModel Assumptions:")
    print(f"  Per-step error rate: {args.error_rate} ({args.error_rate * 100:.2f}%)")
    print(f"  Per-step success rate: {p}")
    print(f"  Target success probability: {args.target_prob}")

    # Calculate k_min
    try:
        k_min = calculate_k_min(num_steps, p, args.target_prob)
        print(f"\nVoting Parameters:")
        print(f"  Minimum k for {args.target_prob*100:.0f}% success: {k_min}")
    except ValueError as e:
        print(f"\nError: {e}")
        print("Per-step success rate must be > 0.5 for voting to work")
        return 1

    # Estimate cost for different models
    print(f"\nEstimated Costs (target: {args.target_prob*100:.0f}% success):")
    print("-" * 50)

    # Approximate tokens per step
    input_tokens = 1000  # ~1k input tokens per step
    output_tokens = 500  # ~500 output tokens per step

    for model_name, config in MODEL_CONFIGS.items():
        cost_per_sample = (
            (input_tokens / 1_000_000) * config.input_cost_per_mtok +
            (output_tokens / 1_000_000) * config.output_cost_per_mtok
        )
        total_cost = estimate_cost(num_steps, p, cost_per_sample, k_min)
        print(f"  {model_name:20s}: ${total_cost:,.0f}")

    # Time estimate
    print(f"\nTime Estimates (at 10 steps/sec):")
    time_seconds = num_steps / 10
    print(f"  Sequential: {time_seconds/3600:.1f} hours")
    print(f"  With k={k_min} parallel: {time_seconds * k_min / 3600:.1f} hours")

    return 0


def cmd_demo(args):
    """Run a quick demo with mock LLM."""
    print("=" * 60)
    print("MAKER Demo - Mock LLM")
    print("=" * 60)

    print("\nRunning demo with 5 disks (31 steps)...")
    print("Using mock LLM client (no API calls)\n")

    solver = MAKERSolver(
        num_disks=5,
        k=3,
        provider="mock",
        validate=True,
        verbose=True
    )

    solution = solver.solve()

    if solver.verify_solution():
        print("\n*** Demo completed successfully! ***")
        return 0
    else:
        print("\n*** Demo failed! ***")
        return 1


def cmd_benchmark(args):
    """Run benchmarks to estimate error rates."""
    print("=" * 60)
    print("MAKER Benchmark - Error Rate Estimation")
    print("=" * 60)

    # This would run actual LLM calls to estimate error rates
    # For now, show the methodology

    print("\nTo estimate error rates:")
    print("1. Sample N random steps from the full puzzle")
    print("2. For each step, call the LLM and compare to ground truth")
    print("3. Calculate error rate = errors / total_samples")
    print("4. Use this to select optimal k and model")

    print("\nExpected error rates from paper (Figure 6):")
    print("-" * 50)
    print(f"  {'Model':20s} {'Error Rate':>12s} {'k_min (1M steps)':>16s}")
    print("-" * 50)

    error_rates = {
        "gpt-4.1-nano": 0.357,
        "gpt-4.1-mini": 0.002,
        "o3-mini": 0.0017,
        "claude-3.5-haiku": 0.184,
        "qwen-3": 0.234,
        "deepseek-v3.1": 0.057,
    }

    for model, error_rate in error_rates.items():
        p = 1 - error_rate
        if p > 0.5:
            k = calculate_k_min(1_000_000, p, 0.95)
            print(f"  {model:20s} {error_rate:>12.4f} {k:>16d}")
        else:
            print(f"  {model:20s} {error_rate:>12.4f} {'N/A':>16s}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="MAKER - Solving Long-Horizon LLM Tasks with Zero Errors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick demo with mock LLM
  python main.py demo

  # Estimate costs for 20-disk puzzle
  python main.py estimate --disks 20 --error-rate 0.002

  # Solve with OpenAI (requires OPENAI_API_KEY)
  python main.py solve --disks 10 --provider openai --model gpt-4.1-mini

  # Solve with mock LLM (for testing)
  python main.py solve --disks 10 --provider mock
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Solve command
    solve_parser = subparsers.add_parser("solve", help="Solve Towers of Hanoi")
    solve_parser.add_argument("--disks", type=int, default=10,
                              help="Number of disks (default: 10)")
    solve_parser.add_argument("--k", type=int, default=3,
                              help="Voting parameter (default: 3)")
    solve_parser.add_argument("--max-tokens", type=int, default=750,
                              help="Max response tokens (default: 750)")
    solve_parser.add_argument("--provider", type=str, default="mock",
                              choices=["openai", "anthropic", "together", "mock"],
                              help="LLM provider (default: mock)")
    solve_parser.add_argument("--model", type=str, default=None,
                              help="Model name (provider-specific)")
    solve_parser.add_argument("--max-steps", type=int, default=None,
                              help="Maximum steps to execute")
    solve_parser.add_argument("--no-validate", action="store_true",
                              help="Skip validation against ground truth")
    solve_parser.add_argument("--verify", action="store_true",
                              help="Verify solution at the end")
    solve_parser.add_argument("--quiet", action="store_true",
                              help="Reduce output")
    solve_parser.add_argument("--checkpoint-dir", type=str, default=None,
                              help="Directory for checkpoints")
    solve_parser.add_argument("--checkpoint-interval", type=int, default=1000,
                              help="Checkpoint every N steps")

    # Estimate command
    estimate_parser = subparsers.add_parser("estimate", help="Estimate costs")
    estimate_parser.add_argument("--disks", type=int, default=20,
                                 help="Number of disks (default: 20)")
    estimate_parser.add_argument("--error-rate", type=float, default=0.002,
                                 help="Per-step error rate (default: 0.002)")
    estimate_parser.add_argument("--target-prob", type=float, default=0.95,
                                 help="Target success probability (default: 0.95)")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Quick demo")

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmarks")

    args = parser.parse_args()

    if args.command == "solve":
        return cmd_solve(args)
    elif args.command == "estimate":
        return cmd_estimate(args)
    elif args.command == "demo":
        return cmd_demo(args)
    elif args.command == "benchmark":
        return cmd_benchmark(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
