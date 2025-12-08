# MAKER - Solving Long-Horizon LLM Tasks with Zero Errors

Implementation of the **MDAP (Massively Decomposed Agentic Processes)** framework from the paper:

> **"Solving a Million-Step LLM Task with Zero Errors"**
> Meyerson et al., Cognizant AI Lab & UT Austin, 2025
> [arXiv:2511.09030](https://arxiv.org/abs/2511.09030)

## Overview

MAKER demonstrates that LLM-based systems can reliably solve tasks with **over one million steps with zero errors** through:

1. **Maximal Agentic Decomposition (MAD)**: Breaking tasks into minimal subtasks (1 step per agent)
2. **First-to-ahead-by-k Voting**: Error correction through multi-agent consensus
3. **Red-flagging**: Detecting and discarding unreliable responses

```
┌─────────────────────────────────────────────────────────────┐
│                    MAKER Architecture                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Task: Towers of Hanoi (20 disks = 1,048,575 steps)       │
│                           │                                  │
│                           ▼                                  │
│   ┌─────────────────────────────────────────────────────┐  │
│   │          Maximal Agentic Decomposition               │  │
│   │          (1 step per agent call)                     │  │
│   └─────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│   ┌─────────────────────────────────────────────────────┐  │
│   │          First-to-ahead-by-k Voting                  │  │
│   │                                                       │  │
│   │   Sample 1 ──┐                                       │  │
│   │   Sample 2 ──┼──► Vote until k-margin winner         │  │
│   │   Sample 3 ──┘                                       │  │
│   └─────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              Red-flagging Filter                      │  │
│   │                                                       │  │
│   │   ✗ Too long (>750 tokens)                          │  │
│   │   ✗ Invalid format                                   │  │
│   │   ✓ Valid response → Count vote                      │  │
│   └─────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│                   Zero-Error Solution                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd maker

# Install dependencies (optional, for API access)
pip install -r requirements.txt

# Or install specific providers
pip install openai      # For OpenAI models
pip install anthropic   # For Anthropic models
pip install together    # For open-source models via Together.ai
```

## Quick Start

### Demo (No API Required)

```bash
# Run demo with mock LLM (5 disks, 31 steps)
python main.py demo
```

### Cost Estimation

```bash
# Estimate costs for 20-disk puzzle
python main.py estimate --disks 20 --error-rate 0.002

# With different error rate
python main.py estimate --disks 15 --error-rate 0.01
```

### Solve with Real LLM

```bash
# Set API key
export OPENAI_API_KEY=your-key-here

# Solve 10-disk puzzle (1,023 steps)
python main.py solve --disks 10 --provider openai --model gpt-4.1-mini

# Solve with validation and verification
python main.py solve --disks 10 --provider openai --verify
```

## Usage

### Command Line Interface

```
python main.py <command> [options]

Commands:
  demo        Quick demo with mock LLM
  estimate    Estimate costs and parameters
  solve       Solve Towers of Hanoi puzzle
  benchmark   Show error rate benchmarks

Options for 'solve':
  --disks N           Number of disks (default: 10)
  --k N               Voting parameter (default: 3)
  --max-tokens N      Max response tokens (default: 750)
  --provider NAME     LLM provider: openai, anthropic, together, mock
  --model NAME        Model name
  --max-steps N       Maximum steps to execute
  --verify            Verify solution at the end
  --checkpoint-dir    Directory for saving checkpoints
```

### Python API

```python
from src.solver import MAKERSolver

# Create solver
solver = MAKERSolver(
    num_disks=10,        # 2^10 - 1 = 1,023 steps
    k=3,                 # Voting margin
    provider="openai",   # or "anthropic", "together", "mock"
    model="gpt-4.1-mini"
)

# Solve
solution = solver.solve()

# Verify
if solver.verify_solution():
    print("Success! Zero errors.")

# Check statistics
print(solver.stats.to_dict())
```

## Key Results from Paper

| Model | Error Rate | k_min | Est. Cost (1M steps) |
|-------|------------|-------|---------------------|
| gpt-4.1-nano | 35.7% | 29 | $41,900 |
| gpt-4.1-mini | 0.22% | 3 | **$3,500** |
| o3-mini | 0.17% | 3 | $9,400 |
| claude-3.5-haiku | 18.4% | 12 | $71,200 |

**Key insight**: Expensive "reasoning" models are not required. Cost-effective models like `gpt-4.1-mini` work best!

## Scaling Laws

The cost scales **log-linearly** with the number of steps:

```
E[cost] = O(s · ln(s))
```

Where:
- `s` = number of steps
- `k_min = O(ln(s))` = minimum votes needed

This means solving a **billion-step** task is only ~1.5x more expensive per step than a million-step task!

## Project Structure

```
maker/
├── main.py              # CLI entry point
├── requirements.txt     # Dependencies
├── README.md           # This file
└── src/
    ├── __init__.py     # Package init
    ├── prompts.py      # Prompt templates
    ├── parsers.py      # Response parsers & red-flagging
    ├── voting.py       # First-to-ahead-by-k voting
    ├── llm_client.py   # LLM API clients
    └── solver.py       # Main MAKER solver
```

## Algorithm Details

### Algorithm 1: generate_solution
```python
def generate_solution(initial_state, model, k):
    actions = []
    state = initial_state
    for step in range(total_steps):
        action, state = do_voting(state, model, k)
        actions.append(action)
    return actions
```

### Algorithm 2: do_voting
```python
def do_voting(state, model, k):
    votes = defaultdict(int)
    while True:
        response = get_vote(state, model)
        votes[response] += 1
        if votes[response] >= k + max(other_votes):
            return response
```

### Algorithm 3: get_vote
```python
def get_vote(state, model):
    while True:
        response = model.generate(state)
        if no_red_flags(response):
            return parse(response)
```

## Citation

```bibtex
@article{meyerson2025maker,
  title={Solving a Million-Step LLM Task with Zero Errors},
  author={Meyerson, Elliot and Paolo, Giuseppe and Dailey, Roberto and
          Shahrzad, Hormoz and Francon, Olivier and Hayes, Conor F. and
          Qiu, Xin and Hodjat, Babak and Miikkulainen, Risto},
  journal={arXiv preprint arXiv:2511.09030},
  year={2025}
}
```

## License

MIT License - See LICENSE file for details.

## Acknowledgments

This implementation is based on the research by Cognizant AI Lab and UT Austin.
The original paper introduced the concept of Massively Decomposed Agentic Processes (MDAPs)
as an alternative path to scaling AI systems.
