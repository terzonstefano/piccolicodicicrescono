"""
MAKER: Maximal Agentic decomposition, first-to-ahead-by-K Error correction, and Red-flagging

Implementation based on the paper:
"Solving a Million-Step LLM Task with Zero Errors"
by Meyerson et al. (Cognizant AI Lab & UT Austin, 2025)

This framework implements Massively Decomposed Agentic Processes (MDAPs) for
solving long-horizon tasks with zero errors.
"""

__version__ = "1.0.0"
__author__ = "Based on Meyerson et al."

from .voting import do_voting, get_vote
from .solver import MAKERSolver
from .parsers import parse_move_state_repair, parse_move_state_flag
from .prompts import SYSTEM_PROMPT, USER_TEMPLATE
